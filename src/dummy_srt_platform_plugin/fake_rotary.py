"""
General SGLang plugin: fake rotary position embedding.

Why this exists
----------------
Confirmed via a real launch + real traceback: with support_piecewise_cuda_graph()
returning True, tracing now gets past VocabParallelEmbedding.forward()
(fake_embedding.py) and into the first decoder layer's rotary embedding
call. Same root cause as fake_embedding.py, a different tensor:

    RotaryEmbedding.forward_native() (rotary_embedding/base.py:254):
        cos_sin = self.cos_sin_cache.index_select(0, positions)

self.cos_sin_cache is a registered buffer, built torch.device("meta") by
fake_load.py's meta-device model construction. positions is real, ordinary
"cpu" -- part of the dummy warmup batch, never faked, exactly like
input_ids was for the embedding table. Dynamo's fake-tensor propagation
cannot unify "meta" and "cpu" for aten.index_select.default, so tracing
hard-fails at the same "Unhandled FakeTensor Device Propagation... found
two different devices meta, cpu" error, one layer further into the model.

forward_cpu() was checked directly (rotary_embedding/base.py:334): it only
takes an AMX-kernel path when _is_cpu_amx_available; otherwise it falls
straight back into forward_native() anyway. So hooking forward_cpu alone
would not reliably avoid this -- the fix targets the outer dispatch entry
point instead, RotaryEmbedding.forward() (inherited from MultiPlatformOp,
multi_platform.py:83's self._forward_method(*args, **kwargs) is what
routes to forward_native/forward_cpu/forward_cuda/etc. depending on
platform) -- same "hook the whole method, not the real sub-call inside
it" principle fake_embedding.py already uses for
VocabParallelEmbedding.forward().

Unlike the embedding fix, this one needs no shape/dtype reconstruction at
all: by the time this call happens, query and key are already
torch.device("meta") -- they are the direct output of qkv_proj's
Fp8LinearMethod.apply(), already faked by fake_quant.py. Rotary embedding
doesn't change shape or dtype (it rotates values in place, conceptually),
so handing back exactly what was received is both the simplest and the
most correct fake here -- no new tensor needs to be materialized.

Rotary embedding's own cost (a handful of elementwise multiplies/adds per
element) is not matmul-shaped and is not roofline-estimated here, same
posture as fake_embedding.py's embedding lookup: real transformer FLOPs
are dominated by the matmuls (linear/MoE, already faked in fake_quant.py)
and attention (fake_attention.py), not this elementwise step.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_rotary = "dummy_srt_platform_plugin.fake_rotary:register"
"""

import logging

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def register():
    """Entry point called by load_plugins()."""
    HookRegistry.register(
        "sglang.srt.layers.rotary_embedding.base.RotaryEmbedding.forward",
        _fake_rotary_forward,
        HookType.AROUND,
    )
    logger.info("dummy_rotary plugin registered (RotaryEmbedding.forward faked)")


def _fake_rotary_forward(
    original_fn,
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets=None,
    fused_set_kv_buffer_arg=None,
):
    """AROUND hook: never calls _forward_method (forward_native/forward_cpu/
    forward_cuda/etc.), so self.cos_sin_cache (meta) never meets positions
    (real cpu) under Dynamo's fake-tensor propagation.

    Real RotaryEmbedding.forward() returns (query, key) with the SAME
    shape and dtype as the inputs (confirmed against forward_native's real
    source: it reshapes internally but returns via
    torch.cat(...).reshape(query_shape) / (key_shape) -- output shape
    always matches input shape). query and key are already
    torch.device("meta") by this point (output of qkv_proj's faked
    Fp8LinearMethod.apply(), see fake_quant.py), so returning them
    directly -- rather than materializing new tensors -- is both correct
    and the cheapest possible implementation.
    """
    return query, key