"""
Dummy piecewise compilation backend -- Stage 3 (GPU emulation via real
Dynamo/torch.compile graph splitting).

Stages 1-2 never let the real model graph execute at all: the fake_forward
hook intercepted BEFORE self.model.forward() was called, so
DummyPiecewiseBackend was wired up (get_piecewise_backend_cls() returns it)
but dormant.

Stage 3 changes that: torch.compile is enabled for real, Dynamo traces the
actual model (on torch.device("meta") tensors, so no real memory is ever
touched), and splits the graph at real SGLang-authored boundaries (e.g.
around attention kernels that aren't Dynamo-traceable). This backend is
registered as the compiled callable for each of those real pieces.

It never invokes the real compiled kernel and never materializes anything
off meta:
  - Every piece's inputs and outputs stay on torch.device("meta"), all the
    way through every piece -- mixing in a real tensor mid-graph breaks
    Dynamo's own fake-tensor propagation for any downstream piece still on
    meta (confirmed empirically: a real CPU tensor feeding a still-meta
    linear layer raises a device-mismatch error during tracing itself, not
    during compilation).
  - On first call for a given instance (each instance == one shape bucket
    for one piece, matching real SGLang's own per-shape piecewise-compile
    capture passes), it estimates a latency from this piece's own FX graph
    via a coarse roofline CostModel, and caches both that latency and the
    piece's output shape/dtype on the instance.
  - Every call (including the first) sleeps for the cached latency and
    returns a freshly-materialized meta tensor (or tuple of them) matching
    the real output's shape/dtype -- so downstream pieces get something
    structurally correct to keep tracing/executing against, without any
    real compute or memory ever happening.

Real, non-meta output values only need to exist at the very end of the whole
forward pass (e.g. logits), and that substitution happens one layer up, in
fake_forward.py's post-forward hook -- NOT here. This backend fakes time;
the forward hook fakes values. Keeping those two concerns separate keeps
this class from needing to know whether it's "the last piece".

BUGFIX (confirmed via a real launch + real traceback): a compiled piece
with MULTIPLE outputs (a tuple, e.g. q/k/v feeding directly into a
downstream split-op like unified_attention_with_output) could silently
materialize as a tuple with one or more None elements inside it, if
shape/dtype extraction failed for just ONE of those outputs while
succeeding for the others. The old __call__ only checked whether the
WHOLE materialized result was None -- a tuple containing a None element
is not None itself, so that check passed, and a tuple like
(real_tensor, real_tensor, None, real_tensor) was returned as-is. That
None then propagated into whatever split-op consumed it (observed:
unified_attention_with_output's `query` argument arriving as None,
crashing with "'NoneType' object is not subscriptable" on
`query[:real_num_tokens]`). Fixed by recursively checking for any None
anywhere inside the materialized structure and treating that the same as
a fully-failed extraction -- falling back to the real
compiled_graph_for_general_shape(*args) call instead.
"""

import logging
import time
from typing import Any, Callable, Optional, Tuple

import torch

from dummy_srt_platform_plugin.cost_model import CostModel

logger = logging.getLogger(__name__)

# Matmul-family op name fragments we recognize when walking the FX graph.
# These dominate transformer FLOPs; everything else (elementwise ops, norms,
# reshapes) is ignored in this coarse first-pass model.
_MATMUL_NAME_FRAGMENTS = ("mm.default", "addmm.default", "bmm.default", "baddbmm.default")


def _node_val_shape_dtype(node: Any) -> Optional[Tuple[torch.Size, torch.dtype]]:
    """Pull (shape, dtype) off the FakeTensor that Dynamo attaches to `node`
    as node.meta['val'] during tracing. Returns None if unavailable."""
    meta = getattr(node, "meta", None)
    if not isinstance(meta, dict):
        return None
    val = meta.get("val")
    if val is None or not hasattr(val, "shape") or not hasattr(val, "dtype"):
        return None
    return val.shape, val.dtype


def _fx_nodes(graph_obj: Any):
    """
    Return the iterable of FX nodes for `graph_obj`.

    In real SGLang, the `graph` this backend receives is a torch.fx.GraphModule,
    not a bare torch.fx.Graph -- confirmed against the real source
    (sglang.srt.compilation.backend.make_backend's own `graph: fx.GraphModule`
    type hint, and its own node-walk idiom at backend.py:232,
    `for node in graph.graph.nodes`). So the real path here is
    `graph_obj.graph.nodes`.

    Falls back to `graph_obj.nodes` for a bare Graph (defensive only; not the
    real-usage path). Raises like the rest of this module's introspection
    helpers if neither shape is present -- callers catch broadly and fall
    back to a zero-cost estimate (this is what makes a mocked `graph` in unit
    tests degrade safely instead of raising)."""
    inner = getattr(graph_obj, "graph", None)
    if inner is not None and hasattr(inner, "nodes"):
        return inner.nodes
    return graph_obj.nodes


def _dtype_size_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def estimate_flops_and_bytes(graph: Any) -> Tuple[float, float]:
    """
    Walk an FX graph's nodes and sum FLOPs/bytes for matmul-family ops, using
    the FakeTensor shape/dtype metadata Dynamo attaches to each node
    (node.meta['val']) during tracing.

    Coarse roofline model: only matmul-family ops (mm, addmm, bmm, baddbmm)
    are counted, since they dominate transformer FLOPs; elementwise/norm ops
    are ignored as a first approximation. A later refinement can walk every
    op via torch.utils.flop_counter for a precise estimate.

    Raises whatever the underlying graph object raises on iteration/attribute
    access (e.g. a mocked graph in a unit test) -- callers are expected to
    catch broadly and fall back to a zero-cost estimate, since this function
    only works on real FX graphs.
    """
    total_flops = 0.0
    total_bytes = 0.0

    for node in _fx_nodes(graph):
        if getattr(node, "op", None) != "call_function":
            continue

        target_str = str(node.target)
        if not any(frag in target_str for frag in _MATMUL_NAME_FRAGMENTS):
            continue

        operand_shapes = []
        dtype = torch.float16
        for arg in node.args:
            info = _node_val_shape_dtype(arg)
            if info is None:
                continue
            shape, dtype = info
            if len(shape) >= 2:
                operand_shapes.append(shape)

        # mm/bmm: args = (A, B). addmm/baddbmm: args = (bias, A, B) -- in
        # both cases the last two >=2D operands are the actual matmul pair.
        if len(operand_shapes) < 2:
            continue
        a_shape, b_shape = operand_shapes[-2], operand_shapes[-1]

        batch = 1
        for d in a_shape[:-2]:
            batch *= int(d)
        m, k = int(a_shape[-2]), int(a_shape[-1])
        k2, n = int(b_shape[-2]), int(b_shape[-1])
        if k != k2:
            # Doesn't match our (M,K) x (K,N) assumption -- skip rather than
            # guess at a possibly-wrong FLOP count.
            continue

        elem_size = _dtype_size_bytes(dtype)
        total_flops += 2.0 * batch * m * k * n
        total_bytes += (batch * m * k + batch * k * n + batch * m * n) * elem_size

    return total_flops, total_bytes


def _output_meta_nodes(graph: Any) -> Optional[Any]:
    """Return the FX graph's output node's args[0] -- a single Node, or a
    tuple/list of Nodes for multi-output pieces. Raises like
    estimate_flops_and_bytes if `graph` isn't a real FX graph."""
    output_node = None
    for node in _fx_nodes(graph):
        if getattr(node, "op", None) == "output":
            output_node = node
    if output_node is None or not output_node.args:
        return None
    return output_node.args[0]


def _materialize_meta_like(val: Any) -> Any:
    """Recursively build fresh torch.device('meta') tensor(s) matching the
    shape/dtype recorded on `val` (a Node, or tuple/list of Nodes).

    Returns None for any single element whose shape/dtype can't be
    extracted -- callers MUST check the result with _contains_none()
    before trusting a tuple/list result, since a tuple with one None
    element is not None itself (see _contains_none's docstring and this
    module's BUGFIX note)."""
    if isinstance(val, (tuple, list)):
        made = [_materialize_meta_like(v) for v in val]
        return tuple(made) if isinstance(val, tuple) else made
    info = _node_val_shape_dtype(val)
    if info is None:
        return None
    shape, dtype = info
    return torch.empty(shape, dtype=dtype, device="meta")


def _contains_none(val: Any) -> bool:
    """True if `val` is None, or is a tuple/list containing None anywhere
    (recursively). Needed because _materialize_meta_like can partially
    succeed on a multi-output piece: the top-level tuple/list itself is
    non-None even if one of its elements failed extraction and came back
    as None -- a plain `is not None` check on the whole structure misses
    that case entirely (confirmed by a real crash: a None element reached
    a downstream split-op as a genuinely None argument instead of a meta
    tensor, see this module's BUGFIX note above)."""
    if val is None:
        return True
    if isinstance(val, (tuple, list)):
        return any(_contains_none(v) for v in val)
    return False


class DummyPiecewiseBackend:
    """
    GPU-emulating piecewise compilation backend.

    Each instance corresponds to one real Dynamo-split piece of the model,
    for one shape bucket (matching real SGLang's own per-shape piecewise
    capture passes). Never runs real compute or touches real memory: stays
    meta-in/meta-out uniformly, and its only real-world side effect is
    time.sleep() for a cached, roofline-estimated latency.
    """

    def __init__(
        self,
        graph: Any,
        compile_config: Any,
        inductor_config: dict,
        graph_pool: Any,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list,
        compiled_graph_for_general_shape: Callable,
        sglang_backend,
    ):
        """
        Initialize the dummy piecewise backend.

        Args:
            graph: The PyTorch FX graph module for this piece.
            compile_config: Compilation configuration.
            inductor_config: Inductor-specific configuration.
            graph_pool: Memory pool for graph operations (unused; no real
                memory is ever allocated by this backend).
            piecewise_compile_index: Index of this graph in the piecewise
                sequence.
            total_piecewise_compiles: Total number of piecewise compiles.
            sym_shape_indices: Indices of symbolic shapes in the input.
            compiled_graph_for_general_shape: Callable for the real
                general-shape compiled graph. Invoked whenever output-spec
                extraction fails (fully, or partially -- see BUGFIX note
                above), so this remains a real, exercised fallback path,
                not just a defensive one kept for mock-based unit tests.
            sglang_backend: Backend instance from SGLang.
        """
        self.graph = graph
        self.compile_config = compile_config
        self.inductor_config = inductor_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.sym_shape_indices = sym_shape_indices
        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape
        self.sglang_backend = sglang_backend

        # Instance-level cache. One shape bucket per instance means a plain
        # instance attribute is sufficient -- deliberately NOT a shared
        # module-level dict keyed by shape, to avoid the cross-instance
        # singleton mutation bug hit earlier in this project.
        self._cached_latency: Optional[float] = None
        self._cached_output_spec: Optional[Any] = None
        self._cost_model: Optional[CostModel] = None

        logger.debug(
            "DummyPiecewiseBackend initialized for graph %d/%d",
            piecewise_compile_index + 1,
            total_piecewise_compiles,
        )

    def _compute_latency_and_output_spec(self) -> None:
        """First-call setup, run exactly once per instance: resolve the GPU
        spec, estimate this piece's latency from its own FX graph, and
        capture its output shape/dtype. Any failure to introspect `graph`
        (e.g. it's a mock, not a real FX graph) degrades gracefully to a
        zero-cost estimate and no output spec, which makes __call__ fall
        back to invoking compiled_graph_for_general_shape directly."""
        self._cost_model = CostModel()  # resolves current_gpu_spec() once

        try:
            flops, bytes_moved = estimate_flops_and_bytes(self.graph)
        except Exception as e:
            logger.debug(
                "FLOPs/bytes estimation failed for piece %d/%d (%s); "
                "defaulting to zero cost",
                self.piecewise_compile_index + 1,
                self.total_piecewise_compiles,
                e,
            )
            flops, bytes_moved = 0.0, 0.0

        try:
            self._cached_output_spec = _output_meta_nodes(self.graph)
        except Exception as e:
            logger.debug(
                "Output-spec extraction failed for piece %d/%d (%s); "
                "will fall back to compiled_graph_for_general_shape",
                self.piecewise_compile_index + 1,
                self.total_piecewise_compiles,
                e,
            )
            self._cached_output_spec = None

        self._cached_latency = self._cost_model.estimate(flops, bytes_moved)

        logger.debug(
            "DummyPiecewiseBackend piece %d/%d: flops=%.3e bytes=%.3e -> latency=%.6fs",
            self.piecewise_compile_index + 1,
            self.total_piecewise_compiles,
            flops,
            bytes_moved,
            self._cached_latency,
        )

    def __call__(self, *args) -> Any:
        """
        Emulate executing this compiled piece.

        Never touches compiled_graph_for_general_shape or real memory in
        normal operation -- EXCEPT when this piece's output-spec
        extraction failed, fully or partially (see BUGFIX note above),
        in which case the real compiled callable is invoked instead of
        returning a broken (possibly None-containing) result.

        Sleeps for a roofline-estimated latency (computed once on first
        call, cached thereafter) and returns fresh meta tensor(s) matching
        this piece's real output shape/dtype.
        """
        if self._cached_latency is None:
            self._compute_latency_and_output_spec()

        if self._cached_latency:
            time.sleep(self._cached_latency)

        if self._cached_output_spec is not None:
            materialized = _materialize_meta_like(self._cached_output_spec)
            if not _contains_none(materialized):
                return materialized
            logger.debug(
                "DummyPiecewiseBackend piece %d/%d: materialized output "
                "contained a None element (partial shape/dtype extraction "
                "failure) -- falling back to compiled_graph_for_general_shape",
                self.piecewise_compile_index + 1,
                self.total_piecewise_compiles,
            )

        # Fallback path: reached when `graph` wasn't a real FX graph we
        # could introspect (e.g. a MagicMock in a unit test), OR when
        # output-spec extraction succeeded at the top level but produced a
        # None inside a multi-output tuple/list (see BUGFIX note above).
        return self.compiled_graph_for_general_shape(*args)