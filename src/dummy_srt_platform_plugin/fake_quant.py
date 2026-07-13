"""
General SGLang plugin: fake FP8-quantized linear and FP8 MoE compute.

Why this exists
----------------
Confirmed via a real launch + real traceback: Fp8LinearMethod.apply()
(sglang.srt.layers.quantization.fp8) dispatches block-FP8 activation
quantization to a Triton kernel (per_token_group_quant_fp8 ->
_per_token_group_quant_8bit) whenever a model's checkpoint carries an
Fp8Config -- which Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 does
unconditionally, --load-format dummy or not (dummy load only skips
*materializing* weights, not which quant_method class gets attached to
each layer -- see fake_load.py's own docstring on this exact distinction).
Triton's runtime tries to resolve an active GPU driver to know which
backend to codegen for; on a CPU-only box there are zero CUDA/HIP/XPU
drivers registered, so triton.runtime.driver._create_driver() raises
"RuntimeError: 0 active drivers ([]). There should only be one." -- fatal,
not a warning.

Why fake_forward.py's ModelRunner.forward hook and fake_attention.py's
DummyNativeAttnBackend hooks don't catch this
-----------------------------------------------------------------------
The crash's own traceback confirms the call chain: it originates inside
TcPiecewiseCudaGraphBackend's compile-pass warmup
(_run_compile_pass -> PrefillCudaGraphRunner._run_dummy_forward), which
calls self.model_runner.model.forward(...) DIRECTLY -- one level below
ModelRunner.forward, so fake_forward.py's hook is bypassed entirely for
this call. fake_attention.py's hooks only target attention
forward_extend/forward_decode on DummyNativeAttnBackend; they have no
reach into linear or MoE layers, where FP8 quantized matmul lives.

Real per-request serving hits the exact same Fp8LinearMethod.apply /
Fp8MoEMethod.apply seam -- LinearBase.forward always calls
self.quant_method.apply(self, input_, bias) (confirmed at linear.py:471),
and FusedMoE.forward_impl -> run_moe_core always calls
self.quant_method.apply(layer=self, dispatch_output=dispatch_output)
(confirmed at fused_moe_triton/layer.py:1285-1290) -- compile-pass warmup
or not. So hooking apply() here, rather than trying to special-case the
cuda-graph-runner internals (whose class/method names and line numbers
already differ from most published SGLang trees, and churn release to
release), is the one seam that covers BOTH the compile-pass warmup AND
real request serving without needing two separate fixes.

Which layers this covers / doesn't
-----------------------------------
- Fp8LinearMethod.apply -- confirmed crash site (this session's actual
  traceback).
- Fp8MoEMethod.apply -- confirmed live via a second real crash + real
  traceback (fused_moe_triton/layer.py:1159 run_moe_core ->
  self.quant_method.apply(...) -> qwen3_moe.py's self.experts(...)),
  during actual request serving (the real event loop, forward_batch_
  generation, well after weight loading and server startup completed --
  NOT during weight/param loading, despite that being an earlier working
  assumption in this project; this fires once per MoE layer, per forward
  pass, for every batch/request).
- Every other quant scheme (AWQ, GPTQ/Marlin, INT8 w8a8, ModelOpt
  fp8/nvfp4, mxfp4, moe_wna16) is explicitly OUT of scope here. Each has
  its own concrete apply() with its own dispatch, and none has been tested
  against this platform yet -- consistent with this project's existing
  posture of failing loudly rather than silently guessing (see
  srt_platform.py's MLA/DSA KV pool NotImplementedError for the same
  posture). Extend this file's register() with an additional
  HookRegistry.register(...) call, following the exact same pattern, if
  and when a model using one of those schemes is actually tested here.

Fp8MoEMethod.apply() call signature -- CONFIRMED, not guessed
-----------------------------------------------------------------
Verified directly against real upstream source
(fused_moe_triton/layer.py, FusedMoE.run_moe_core):

    def run_moe_core(self, dispatch_output: DispatchOutput) -> CombineInput:
        return self.quant_method.apply(
            layer=self,
            dispatch_output=dispatch_output,
        )

Both `layer` and `dispatch_output` are passed as KEYWORD arguments, never
positional. `layer` binds to this hook's own named `layer` parameter (via
AROUND-hook dispatch), so only `dispatch_output` shows up in **kwargs --
confirmed by a real crash log showing args=[], kwargs=['dispatch_output'].

dispatch_output's concrete type depends on get_moe_a2a_backend():
confirmed via a real launch's printed ServerArgs (moe_a2a_backend='none')
plus fused_moe_triton/layer.py's create_moe_dispatcher(), that 'none'
(and also 'megamoe', 'ascend_fuseep') resolves to StandardDispatcher, whose
dispatch() returns a StandardDispatchOutput NamedTuple with a real
`hidden_states: torch.Tensor` field -- and whose combine() expects a
StandardCombineInput NamedTuple with exactly one field, also named
hidden_states, unpacked via `(hidden_states,) = combine_input`. This
hook's StandardCombineInput(hidden_states=fake_output) construction and
dispatch_output.hidden_states read were both confirmed field-for-field
against this real source, not assumed. Other a2a backends (deepep,
mooncake, mori, nixl) resolve to different dispatcher/CombineInput types
this hook has NOT been tested against -- if you switch --moe-a2a-backend
away from "none", re-verify this contract before trusting this hook.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_quant = "dummy_srt_platform_plugin.fake_quant:register"
"""

import logging
import time

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from dummy_srt_platform_plugin.cost_model import CostModel

logger = logging.getLogger(__name__)

# Module-level holder for the resolved CostModel -- same pattern as
# fake_attention.py's _cost_model_holder, for the same reason (avoid the
# per-process singleton mutation bug already hit twice in this project).
_cost_model_holder: dict = {}

# One-time diagnostic flag so the MoE signature-detection branch logs which
# path it took exactly once, not on every call -- lets you confirm (via the
# log line) which branch fired on your actual pinned SGLang without
# spamming logs on every MoE forward.
_moe_signature_logged = {"done": False}


def _get_cost_model() -> CostModel:
    if "model" not in _cost_model_holder:
        _cost_model_holder["model"] = CostModel()
    return _cost_model_holder["model"]


def register():
    """Entry point called by load_plugins()."""
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.apply",
        _fake_fp8_linear_apply,
        HookType.AROUND,
    )
    logger.info("dummy_quant plugin registered (Fp8LinearMethod.apply faked)")

    try:
        HookRegistry.register(
            "sglang.srt.layers.quantization.fp8.Fp8MoEMethod.apply",
            _fake_fp8_moe_apply,
            HookType.AROUND,
        )
        logger.info("dummy_quant plugin registered (Fp8MoEMethod.apply faked)")
    except Exception as e:
        # Fp8MoEMethod's exact class name/location has moved across SGLang
        # releases in the past (see this file's module docstring). Fail
        # loudly in the log, but don't take down the whole plugin load --
        # a model that never reaches an MoE layer (unlikely for this
        # project's actual target models, but not impossible for future
        # ones) would otherwise be blocked by a missing hook it never
        # needed.
        logger.warning(
            "dummy_quant: could not register Fp8MoEMethod.apply hook (%s). "
            "If the target model is MoE, real Triton MoE kernels WILL run "
            "on CPU and WILL crash the same way Fp8LinearMethod did -- "
            "verify Fp8MoEMethod's current location/name before relying on "
            "this platform for an MoE model.",
            e,
        )


def _dtype_size_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _numel_leading(shape) -> int:
    n = 1
    for d in shape[:-1]:
        n *= int(d)
    return n


# ---------------------------------------------------------------------
# Fp8LinearMethod.apply
# ---------------------------------------------------------------------

def _linear_flops_bytes(layer, x: torch.Tensor, out_features: int, elem_size: int):
    """Coarse roofline FLOPs/bytes for one quantized linear call.

    m = number of tokens (product of all leading dims of x, matching how
    SGLang flattens batch/seq before a linear call). k = input features
    (per this rank's shard). n = output features (per this rank's shard,
    i.e. layer.output_size_per_partition -- already TP-sharded at __init__
    time by SGLang's own ColumnParallelLinear/RowParallelLinear, matching
    the same per-rank-correct-by-construction reasoning fake_load.py's
    weight_bytes figure already relies on).
    """
    m = _numel_leading(x.shape)
    k = getattr(layer, "input_size_per_partition", None) or x.shape[-1]
    n = out_features

    flops = 2.0 * m * k * n
    bytes_moved = (m * k + k * n + m * n) * elem_size
    return flops, bytes_moved, m


def _resolve_linear_output_features(layer, x: torch.Tensor) -> int:
    """Prefer layer.output_size_per_partition (the real, TP-sharding-aware
    attribute LinearBase subclasses set at __init__ time). Falls back to
    the quantized weight's own leading shape dim, then to x's own last dim
    (a same-size projection) only as a last resort -- logged loudly since
    that fallback is very unlikely to be shape-correct for a real
    projection layer."""
    n = getattr(layer, "output_size_per_partition", None)
    if n is not None:
        return int(n)

    weight = getattr(layer, "weight", None)
    if weight is not None and weight.dim() >= 1:
        return int(weight.shape[0])

    logger.debug(
        "dummy_quant: Fp8LinearMethod fake could not resolve "
        "output_size_per_partition or weight.shape[0]; falling back to "
        "x.shape[-1] (likely wrong for a real projection layer)"
    )
    return int(x.shape[-1])


def _fake_fp8_linear_apply(original_fn, self, layer, x, bias=None):
    """AROUND hook: never calls original_fn, so
    apply_w8a8_block_fp8_linear / triton_w8a8_block_fp8_linear / the
    per_token_group_quant_fp8 Triton kernel are never reached -- sidesteps
    the "0 active drivers" crash entirely rather than working around
    Triton's driver resolution.

    Returns a tensor shaped and dtyped exactly like what a real
    Fp8LinearMethod.apply() would return: same leading dims as x, last dim
    = this rank's output_size_per_partition, dtype = x.dtype (the
    layer's activation dtype -- bf16/fp16 -- not the fp8 weight dtype;
    quantized linear layers dequantize back to activation dtype on output,
    confirmed by every quant scheme's apply() contract in this codebase)."""
    out_features = _resolve_linear_output_features(layer, x)

    try:
        elem_size = _dtype_size_bytes(x.dtype)
        flops, bytes_moved, _m = _linear_flops_bytes(layer, x, out_features, elem_size)
        latency = _get_cost_model().estimate(flops, bytes_moved)
    except Exception as e:
        logger.debug(
            "dummy_quant: Fp8LinearMethod latency estimation failed (%s); "
            "skipping sleep",
            e,
        )
        latency = 0.0

    if latency:
        time.sleep(latency)

    output = torch.empty(
        (*x.shape[:-1], out_features), dtype=x.dtype, device=x.device
    )
    if bias is not None:
        # Real quant methods that receive a non-None bias are expected to
        # fuse the add into apply() (that's why LinearBase.forward passes
        # bias INTO apply rather than adding it itself) -- matched here for
        # shape/dtype consistency with downstream code, even though the
        # values are meaningless either way.
        output = output + bias
    return output


# ---------------------------------------------------------------------
# Fp8MoEMethod.apply
# ---------------------------------------------------------------------

def _moe_flops_bytes(num_tokens: float, top_k: float, hidden_size: float,
                      intermediate_size: float, elem_size: int):
    """Coarse roofline FLOPs/bytes for one MoE apply() call across all
    activated experts in the batch.

    Each activated (token, expert) pair runs two grouped GEMMs -- gate/up
    projection (hidden -> 2*intermediate, fused gate+up as is standard for
    SwiGLU-style MoE FFNs) and down projection (intermediate -> hidden) --
    same [tokens x features] shape reasoning as _extend_flops_bytes in
    fake_attention.py, just with expert-FFN dimensions instead of
    attention-head dimensions. num_tokens * top_k is the total number of
    activated (token, expert) pairs the batch requires compute for.
    """
    activated_pairs = num_tokens * top_k

    flops = 2.0 * activated_pairs * (
        2.0 * hidden_size * intermediate_size  # gate+up projection
        + intermediate_size * hidden_size       # down projection
    )
    bytes_moved = activated_pairs * (hidden_size + intermediate_size) * elem_size * 2
    return flops, bytes_moved


def _resolve_moe_dims(layer, hidden_states: torch.Tensor):
    """Best-effort attribute resolution across SGLang MoE layer variants.
    Every attribute here is read via getattr with a fallback rather than
    assumed present, since Fp8MoEMethod's host layer's exact attribute
    names have not been independently re-verified against a real cloned
    checkout for this project (see module docstring's VERIFY note for the
    call-signature part, which HAS been verified -- dimension attribute
    names below have not)."""
    top_k = (
        getattr(layer, "top_k", None)
        or getattr(getattr(layer, "moe_runner_config", None), "top_k", None)
        or 1
    )
    hidden_size = hidden_states.shape[-1]
    intermediate_size = (
        getattr(layer, "intermediate_size_per_partition", None)
        or getattr(layer, "moe_intermediate_size", None)
        or hidden_size
    )
    return float(top_k), float(hidden_size), float(intermediate_size)


def _fake_moe_output_like(hidden_states: torch.Tensor) -> torch.Tensor:
    """Real MoE apply() returns combined expert output the same shape/dtype
    as the dispatched hidden states (post-combine, pre-residual-add) --
    matched here regardless of which call-signature branch fired."""
    return torch.empty_like(hidden_states)


def _fake_fp8_moe_apply(original_fn, self, layer, *args, **kwargs):
    """AROUND hook: never calls original_fn, so no real fused-MoE Triton
    grouped-GEMM / block-FP8-quant kernel is ever reached.

    Call-signature detection, in priority order:

    1. kwargs["dispatch_output"] -- the CONFIRMED real call shape for this
       codebase (fused_moe_triton/layer.py's run_moe_core:
       self.quant_method.apply(layer=self, dispatch_output=dispatch_output),
       both keyword args). This is verified against real source, not
       guessed -- see module docstring.
    2. A bare tensor as args[0] -- an older, positional call convention
       (apply(self, layer, x, router_logits, top_k, ...)) that has NOT been
       confirmed against any real call site in this codebase; kept only as
       a defensive fallback in case an older SGLang version is ever
       targeted.
    3. args[0] having a .hidden_states attribute -- dispatch_output passed
       positionally instead of by keyword; not confirmed as a real call
       shape here either, but a cheap, harmless fallback to keep.
    4. kwargs["hidden_states"] directly -- another unconfirmed defensive
       fallback.

    Every branch below (2)-(4) is best-effort; only (1) is verified. If
    none match, this raises loudly rather than silently guessing a shape
    that could corrupt the batch -- consistent with this project's
    established practice of failing loudly on an unverified integration
    point rather than proceeding on an assumption.
    """
    hidden_states = None
    dispatch_output = None

    if "dispatch_output" in kwargs:
        # CONFIRMED real call shape (see module docstring): run_moe_core
        # calls self.quant_method.apply(layer=self, dispatch_output=...).
        dispatch_output = kwargs["dispatch_output"]
        hidden_states = dispatch_output.hidden_states
        branch = "kwarg dispatch_output (CONFIRMED: fused_moe_triton/layer.py run_moe_core)"
    elif args and torch.is_tensor(args[0]):
        # Older convention: apply(self, layer, x, router_logits, top_k, ...)
        # NOT confirmed against any real call site in this codebase.
        hidden_states = args[0]
        branch = "tensor (UNCONFIRMED older convention: apply(layer, x, router_logits, ...))"
    elif args and hasattr(args[0], "hidden_states"):
        # dispatch_output passed positionally instead of by keyword.
        # NOT confirmed as an actual call shape here.
        dispatch_output = args[0]
        hidden_states = dispatch_output.hidden_states
        branch = "positional dispatch_output (UNCONFIRMED)"
    elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
        hidden_states = kwargs["hidden_states"]
        branch = "kwarg hidden_states (UNCONFIRMED)"
    else:
        logger.warning(
            "dummy_quant: Fp8MoEMethod fake could not recognize apply() "
            "call signature (args=%s, kwargs=%s) -- returning a zero-cost, "
            "best-effort fallback. VERIFY this against your pinned "
            "SGLang's real Fp8MoEMethod.apply signature (see fake_quant.py "
            "module docstring).",
            [type(a).__name__ for a in args],
            list(kwargs.keys()),
        )
        # Nothing usable to shape an output against -- re-raise rather than
        # silently return something that will corrupt the forward pass in
        # a way that's much harder to debug than a clear crash here.
        raise RuntimeError(
            "dummy_quant: Fp8MoEMethod.apply call signature not recognized; "
            "see fake_quant.py module docstring's VERIFY note."
        )

    if not _moe_signature_logged["done"]:
        logger.info("dummy_quant: Fp8MoEMethod.apply fake using branch: %s", branch)
        _moe_signature_logged["done"] = True

    try:
        num_tokens = float(_numel_leading((*hidden_states.shape[:-1], 1)))
        top_k, hidden_size, intermediate_size = _resolve_moe_dims(layer, hidden_states)
        elem_size = _dtype_size_bytes(hidden_states.dtype)
        flops, bytes_moved = _moe_flops_bytes(
            num_tokens, top_k, hidden_size, intermediate_size, elem_size
        )
        latency = _get_cost_model().estimate(flops, bytes_moved)
    except Exception as e:
        logger.debug(
            "dummy_quant: Fp8MoEMethod latency estimation failed (%s); "
            "skipping sleep",
            e,
        )
        latency = 0.0

    if latency:
        time.sleep(latency)

    fake_output = _fake_moe_output_like(hidden_states)

    if dispatch_output is not None:
        # CONFIRMED against real source (token_dispatcher/standard.py) for
        # the moe_a2a_backend="none" case actually in play on this
        # platform's tested launches: StandardCombineInput is a NamedTuple
        # with exactly one field, hidden_states, and StandardDispatcher
        # .combine() unpacks it via `(hidden_states,) = combine_input` --
        # matching this construction field-for-field. Other a2a backends
        # (deepep, mooncake, mori, nixl) resolve to different
        # dispatcher/CombineInput types NOT covered here -- re-verify if
        # --moe-a2a-backend is ever changed away from "none".
        try:
            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
            return StandardCombineInput(hidden_states=fake_output)
        except Exception as e:
            logger.warning(
                "dummy_quant: could not construct StandardCombineInput (%s); "
                "returning bare tensor instead -- verify Fp8MoEMethod.apply's "
                "real return type for your pinned SGLang / moe_a2a_backend.",
                e,
            )
            return fake_output

    return fake_output