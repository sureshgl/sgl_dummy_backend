"""
General SGLang plugin: fake the real attention compute, separately from
DummyPiecewiseBackend.

Why this exists as its own hook
--------------------------------
Stage 3 lets the real model.forward() run (see fake_forward.py), traced by
Dynamo/torch.compile on meta tensors, with DummyPiecewiseBackend faking
each *compiled* piece's latency. But attention itself is deliberately NOT
part of any compiled piece: real attention kernels (FlashAttention,
FlashInfer, or -- on this platform -- torch_native's own
scaled_dot_product_attention call) are either custom ops with no
Meta/FakeTensor kernel registered (Dynamo can't symbolically trace through
them at all) or explicitly wrapped so Dynamo is forced to break the graph
there on purpose (data-dependent KV-cache/page-table indexing, and no
fusion benefit from including an already-hand-tuned kernel in the compiled
region). Either way, the real attention call always runs as ordinary eager
Python, in between two DummyPiecewiseBackend-compiled pieces -- it is never
handed to DummyPiecewiseBackend at all.

That matters here because get_mha_kv_pool_cls() (srt_platform.py) returns
NoOpMHATokenToKVPool, whose set_kv_buffer() raises loudly by design --
verified directly against the real TorchNativeAttnBackend.forward_extend /
forward_decode source, both of which unconditionally call
self.token_to_kv_pool.set_kv_buffer(...). Letting the real attention forward
run for real would hit that raise on literally the first real request. This
hook exists specifically to make sure that call never happens: it never
invokes original_fn, so set_kv_buffer / get_key_buffer / get_value_buffer
are never reached.

Why this hooks DummyNativeAttnBackend, not TorchNativeAttnBackend
-------------------------------------------------------------------
Confirmed against a real launch + real source: SGLang unconditionally
disables cuda_graph_config for BOTH decode and prefill whenever
attention_backend == "torch_native" (server_args.py,
_handle_attention_backend_compatibility) -- which silently stomped
DummySRTPlatform's own "tc_piecewise" default and meant Stage 3 never
actually ran, despite loading successfully. That check is string-keyed, not
class-identity-keyed, so DummySRTPlatform.get_default_attention_backend()
now returns "dummy_native" instead, resolving to
dummy_native_backend.DummyNativeAttnBackend -- a zero-behavior-change
subclass of TorchNativeAttnBackend that exists purely so the literal string
comparison never matches. This module's register() is what actually wires
"dummy_native" into SGLang's attention backend registry (there's no other
natural place for that registration to live, since this is the plugin that
owns all attention-related behavior).

q/k/v arrive here as torch.device("meta") tensors -- they're the output of
whatever DummyPiecewiseBackend-compiled linear projection ran immediately
before attention -- so the emulated output stays meta too, matching every
other piece in the forward pass. Real, non-meta values only get substituted
once, at the very end of the whole forward pass, in fake_forward.py's
post-forward hook.

Latency is estimated via the same coarse-roofline CostModel used by
DummyPiecewiseBackend, but with attention-shaped FLOPs/bytes formulas
instead of a matmul-per-node graph walk: real SGLang treats extend
(prefill) and decode as two structurally different costs (extend is
roughly O(context_len) per new token, i.e. quadratic in sequence length
for a full prefill; decode is O(context_len) per single new token, i.e.
linear per step) and this hook keeps that same split, driven directly by
forward_batch.seq_lens / forward_batch.extend_seq_lens -- real, ordinary
(non-meta) control tensors that fake_forward.py's own _hash_logits already
relies on being real, so reading real values off them here is safe and
consistent with the rest of this plugin.

Unlike DummyPiecewiseBackend (one shape bucket per instance, so its latency
is cached once per instance), attention's cost genuinely varies call to
call -- batch composition and per-request context lengths change on every
request -- so there is no natural "compute once" opportunity here. Latency
is estimated fresh on every call; the only thing cached module-level is the
resolved CostModel itself (its GPUSpec lookup), consistent with the
"resolve shared/expensive state once" pattern used throughout this plugin.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_attention = "dummy_srt_platform_plugin.fake_attention:register"
"""

import logging
import time

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from dummy_srt_platform_plugin.cost_model import CostModel

logger = logging.getLogger(__name__)

# Module-level holder for the resolved CostModel -- not a bare module
# global, for the same reason device.py's _resident_weight_bytes_holder and
# gpu_spec.py's own holder aren't: avoids the per-process singleton
# mutation bug already hit twice in this project.
_cost_model_holder: dict = {}


def _get_cost_model() -> CostModel:
    if "model" not in _cost_model_holder:
        _cost_model_holder["model"] = CostModel()
    return _cost_model_holder["model"]


def _register_dummy_native_backend() -> None:
    """
    Register "dummy_native" as a real, selectable attention backend name --
    a thin subclass of TorchNativeAttnBackend (dummy_native_backend.py)
    with zero behavior changes, existing purely so
    server_args.py's `if attention_backend == "torch_native": disable cuda
    graph` check never matches (see dummy_native_backend.py's docstring for
    the full story, confirmed against a real launch + real source).

    Mutates ATTENTION_BACKENDS directly (equivalent to what
    @register_attention_backend("name") does at module-import time) since
    this registration has to happen at plugin-load time, not at import
    time. Also calls the sanctioned add_attention_backend_choices()
    extension point so "dummy_native" is a valid --attention-backend CLI
    choice too, not just an internal default silently smuggled past
    argparse.
    """
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.server_args import add_attention_backend_choices
    from dummy_srt_platform_plugin.dummy_native_backend import DummyNativeAttnBackend

    def _create_dummy_native_backend(runner):
        return DummyNativeAttnBackend(runner)

    ATTENTION_BACKENDS["dummy_native"] = _create_dummy_native_backend
    add_attention_backend_choices(["dummy_native"])
    logger.info("Registered 'dummy_native' attention backend (DummyNativeAttnBackend)")


def register():
    """Entry point called by load_plugins()."""
    _register_dummy_native_backend()

    HookRegistry.register(
        "dummy_srt_platform_plugin.dummy_native_backend.DummyNativeAttnBackend.forward_extend",
        _fake_forward_extend,
        HookType.AROUND,
    )
    HookRegistry.register(
        "dummy_srt_platform_plugin.dummy_native_backend.DummyNativeAttnBackend.forward_decode",
        _fake_forward_decode,
        HookType.AROUND,
    )
    logger.info("dummy_attention plugin registered (dummy_native attention emulated)")


def _fake_attention_output(q: torch.Tensor, layer) -> torch.Tensor:
    """Build the emulated output tensor -- same shape-derivation logic as
    the real TorchNativeAttnBackend.forward_extend/forward_decode (verified
    against source), but never runs scaled_dot_product_attention or touches
    the KV pool. q is meta, so the result is meta too."""
    if layer.qk_head_dim != layer.v_head_dim:
        return q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
    return torch.empty_like(q)


def _dtype_size_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _extend_flops_bytes(layer, forward_batch, elem_size: int):
    """Coarse roofline FLOPs/bytes for a prefill (extend) attention call.

    Q@K^T and the P@V weighted sum are both
    [num_new_tokens_i x context_len_i x head_dim] per request, summed over
    the batch -- forward_batch.extend_seq_lens is the per-request new-token
    (query) length, forward_batch.seq_lens is the per-request total context
    length (prefix + new tokens) the real _run_sdpa_forward_extend loop
    already uses identically.
    """
    seq_lens = forward_batch.seq_lens.to(torch.int64)
    extend_lens = forward_batch.extend_seq_lens.to(torch.int64)

    qk_context_products = float((extend_lens * seq_lens).sum())
    num_new_tokens = float(extend_lens.sum())
    total_context_tokens = float(seq_lens.sum())

    num_q_heads = layer.tp_q_head_num
    num_kv_heads = layer.tp_k_head_num
    qk_head_dim = layer.qk_head_dim
    v_head_dim = layer.v_head_dim

    # Q@K^T scores + P@V weighted sum -- same [tokens x context] shape,
    # different head_dim each.
    flops = 2.0 * num_q_heads * qk_context_products * (qk_head_dim + v_head_dim)

    q_bytes = num_new_tokens * num_q_heads * qk_head_dim * elem_size
    kv_bytes = total_context_tokens * num_kv_heads * (qk_head_dim + v_head_dim) * elem_size
    out_bytes = num_new_tokens * num_q_heads * v_head_dim * elem_size

    return flops, q_bytes + kv_bytes + out_bytes


def _decode_flops_bytes(layer, forward_batch, elem_size: int):
    """Coarse roofline FLOPs/bytes for a decode attention call: query length
    is always 1 per request, so cost is linear in context length rather
    than quadratic -- forward_batch.seq_lens is the per-request context
    length so far, same field the real _run_sdpa_forward_decode loop uses.
    """
    seq_lens = forward_batch.seq_lens.to(torch.int64)

    total_context_tokens = float(seq_lens.sum())
    num_requests = float(seq_lens.shape[0])

    num_q_heads = layer.tp_q_head_num
    num_kv_heads = layer.tp_k_head_num
    qk_head_dim = layer.qk_head_dim
    v_head_dim = layer.v_head_dim

    flops = 2.0 * num_q_heads * total_context_tokens * (qk_head_dim + v_head_dim)

    q_bytes = num_requests * num_q_heads * qk_head_dim * elem_size
    kv_bytes = total_context_tokens * num_kv_heads * (qk_head_dim + v_head_dim) * elem_size
    out_bytes = num_requests * num_q_heads * v_head_dim * elem_size

    return flops, q_bytes + kv_bytes + out_bytes


def _sleep_for(flops_bytes_fn, layer, forward_batch, dtype: torch.dtype) -> None:
    """Shared plumbing for both hooks: estimate latency, sleep, never raise
    (a latency-estimation failure should never take down a real request)."""
    try:
        elem_size = _dtype_size_bytes(dtype)
        flops, bytes_moved = flops_bytes_fn(layer, forward_batch, elem_size)
        latency = _get_cost_model().estimate(flops, bytes_moved)
    except Exception as e:
        logger.debug("Attention latency estimation failed (%s); skipping sleep", e)
        return
    if latency:
        time.sleep(latency)


def _fake_forward_extend(original_fn, self, q, k, v, layer, forward_batch, save_kv_cache=True):
    """AROUND hook: never calls original_fn, so
    self.token_to_kv_pool.set_kv_buffer(...) (and get_key_buffer /
    get_value_buffer) are never reached -- sidesteps NoOpMHATokenToKVPool's
    raise entirely rather than working around it."""
    _sleep_for(_extend_flops_bytes, layer, forward_batch, q.dtype)
    return _fake_attention_output(q, layer)


def _fake_forward_decode(original_fn, self, q, k, v, layer, forward_batch, save_kv_cache=True):
    """AROUND hook: same reasoning as _fake_forward_extend, decode-shaped
    cost formula."""
    _sleep_for(_decode_flops_bytes, layer, forward_batch, q.dtype)
    return _fake_attention_output(q, layer)