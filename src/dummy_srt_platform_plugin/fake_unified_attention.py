"""
General SGLang plugin: give sglang::unified_attention_with_output a real
kernel reachable from this platform's real tensors.

Why this exists
----------------
Confirmed via real launches + real tracebacks, then pinned down against
real source across four files (radix_attention.py, custom_op.py,
utils/common.py, kernel_api_logging.py):

1. radix_attention.py's unified_attention_with_output() (decorated with
   @register_custom_op(mutates_args=["output"]) @register_split_op()) is
   the real split-op tc_piecewise uses for attention -- registered so
   Dynamo treats it as an opaque node rather than tracing inside real
   attention kernels. Its own body does nothing CUDA-specific: it slices
   query/key/value to real_num_tokens and calls
   get_attn_backend().forward(query, key, value, attention_layer,
   forward_batch, save_kv_cache, **kwargs) -- exactly the call chain that
   reaches DummyNativeAttnBackend.forward_extend/forward_decode, already
   hooked by fake_attention.py.

2. utils/common.py's direct_register_custom_op() registers this real
   function under "CUDA" only (with NPU/XPU/MUSA carve-outs, no CPU/OOT
   branch), plus a Meta fake_impl via _register_fake -- confirmed by a
   real RuntimeError when an earlier version of this plugin tried to
   also claim "Meta": "already a kernel registered from python...".

3. ATTEMPT 1 (registered under "Meta"): blocked by (2) above, never took
   effect -- load_plugins() caught the RuntimeError and logged "Failed to
   execute general plugin", so the server ran with no fix at all.

4. ATTEMPT 2 (registered under "CompositeExplicitAutograd", pointing at
   the module-level name `unified_attention_with_output` imported from
   radix_attention.py): registered successfully this time, but caused
   infinite recursion -- confirmed by a real traceback showing
   torch/_ops.py OpOverload.__call__ calling itself thousands of times.
   Root cause, confirmed against kernel_api_logging.py's debug_torch_op():

       impl = getattr(getattr(torch.ops, namespace), op_name)
       if _KERNEL_API_LOG_LEVEL == 0:
           return impl

   With logging disabled (the default), debug_torch_op just returns
   torch.ops.sglang.unified_attention_with_output ITSELF -- meaning the
   module-level name `unified_attention_with_output` in radix_attention.py
   IS the dispatcher entry point, not the raw Python function. Registering
   it as a kernel for its own op means "the kernel for this op is: call
   this op" -- guaranteed infinite self-recursion regardless of which
   dispatch key it's registered under.

THIS FIX: rather than trying to extract the raw function from behind the
decorator (not exposed by any public name once module-level decoration
has run), this defines a direct, line-for-line copy of the real,
already-verified function body here, importing its actual dependencies
(get_tc_piecewise_forward_context, get_attn_backend, _is_hip,
_zero_padded_pcg_tail) directly from their own un-decorated source
modules/names -- none of those are wrapped by register_custom_op, only
unified_attention_with_output (and unified_sparse_attention_with_output,
untouched here) are. This copy is registered as the
"CompositeExplicitAutograd" kernel -- PyTorch's catch-all key, consulted
whenever no more specific backend kernel (CUDA, Meta, ...) matches, and
never touched by direct_register_custom_op, so it does not conflict with
the CUDA/Meta kernels the real launch always registers first.

VERIFY: if a future SGLang version changes unified_attention_with_output's
real body, this copy will silently drift out of sync. Worth periodically
diffing this function against radix_attention.py's real source.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_unified_attention = "dummy_srt_platform_plugin.fake_unified_attention:register"
"""

import logging
from typing import Optional

import torch

from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)

logger = logging.getLogger(__name__)

_OP_QUALNAME = "sglang::unified_attention_with_output"

try:
    from sglang.srt.layers.radix_attention import _is_hip
except ImportError:
    # Same defensive posture as _zero_padded_pcg_tail below: this name has
    # already drifted once against a real pinned checkout. _is_hip only
    # gates a HIP-specific MLA companion-layer swap -- never relevant on
    # this CPU platform regardless of its value, so False is always a
    # safe default here.
    _is_hip = False
    logger.info(
        "dummy_unified_attention plugin: _is_hip not found in this "
        "SGLang revision's radix_attention.py; defaulting to False (a "
        "HIP-specific branch, never relevant on this CPU platform)"
    )

try:
    from sglang.srt.layers.radix_attention import _zero_padded_pcg_tail
except ImportError:
    # Not present in every SGLang revision (confirmed: this plugin's
    # module-load failed against a real pinned checkout where it doesn't
    # exist, even though it was present when this file was first written
    # against a different commit of radix_attention.py -- this file
    # churns quickly). Safe to no-op rather than hard-require it: its only
    # job upstream is zeroing REAL torch.empty garbage (possible NaN/Inf)
    # in a padded tail so it doesn't propagate through residual/MoE
    # routing/allreduce. `output` here is always torch.device("meta") --
    # no real backing storage, so there is no real garbage to zero in the
    # first place, on this platform, regardless of whether the real helper
    # exists in your pinned SGLang version.
    def _zero_padded_pcg_tail(buf, context) -> None:
        return None

    logger.info(
        "dummy_unified_attention plugin: _zero_padded_pcg_tail not found in "
        "this SGLang revision's radix_attention.py; using a no-op instead "
        "(safe here since output is always a meta tensor with no real "
        "garbage to zero)"
    )


def _real_unified_attention_with_output_body(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    is_neox: Optional[bool] = None,
    llama_4_scaling: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
) -> None:
    """Line-for-line copy of radix_attention.py's real
    unified_attention_with_output body (verified against real source --
    see module docstring). Not a fake: this is the actual real logic,
    given a real (non-recursive) home as a kernel."""
    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    query = query[:real_num_tokens]
    if key is not None:
        key = key[:real_num_tokens]
    if value is not None:
        value = value[:real_num_tokens]

    if _is_hip and not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):
        attention_layer = attention_layer._pcg_mha_companion

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope[:real_num_tokens]
    if k_rope is not None:
        kwargs["k_rope"] = k_rope[:real_num_tokens]
    if sinks is not None:
        kwargs["sinks"] = sinks
    if cos_sin_cache is not None:
        kwargs["cos_sin_cache"] = cos_sin_cache
    if is_neox is not None:
        kwargs["is_neox"] = is_neox
    if llama_4_scaling is not None:
        kwargs["llama_4_scaling"] = llama_4_scaling
    if topk_indices is not None:
        kwargs["topk_indices"] = topk_indices[:real_num_tokens]

    original_out_cache_loc = forward_batch.out_cache_loc
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]

    forward_batch._attn_output = output[:real_num_tokens]

    ret = get_attn_backend().forward(
        query,
        key,
        value,
        attention_layer,
        forward_batch,
        save_kv_cache,
        **kwargs,
    )
    forward_batch.out_cache_loc = original_out_cache_loc

    if ret.data_ptr() != output.data_ptr():
        output[:real_num_tokens].view(ret.shape).copy_(ret)

    _zero_padded_pcg_tail(output, context)
    return


def register():
    """Entry point called by load_plugins()."""
    # Import triggers radix_attention.py's module-level decoration, which
    # (eager=True) immediately calls direct_register_custom_op() and
    # defines/CUDA-registers the op (plus its Meta fake_impl) if it isn't
    # already. Must happen before we can add our own kernel for the same
    # op name.
    import sglang.srt.layers.radix_attention  # noqa: F401

    try:
        torch.library.impl(_OP_QUALNAME, "CompositeExplicitAutograd")(
            _real_unified_attention_with_output_body
        )
        logger.info(
            "dummy_unified_attention plugin registered (%s given a real "
            "'CompositeExplicitAutograd' kernel -- a verified copy of the "
            "real logic, which reaches fake_attention.py's hooks via "
            "get_attn_backend().forward())",
            _OP_QUALNAME,
        )
    except RuntimeError as e:
        if "Tried to register an operator" in str(e) and "multiple times" in str(e):
            logger.info(
                "dummy_unified_attention plugin: %s already has a "
                "'CompositeExplicitAutograd' kernel registered, not "
                "overriding",
                _OP_QUALNAME,
            )
        else:
            raise