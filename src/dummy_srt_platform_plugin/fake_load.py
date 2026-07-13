"""
General SGLang plugin: skip real weight allocation, not just real compute.

Even with --load-format dummy, DummyModelLoader.load_model() still:
  1. Builds the *real* nn.Module graph on the *real* target device -- every
     Linear/Embedding/etc. gets a real, fully-sized parameter tensor.
  2. Calls initialize_dummy_weights(model), writing random values into
     every one of those real tensors.

Step 1 is the expensive one for large models: a 70B-parameter model still
needs on the order of 140GB of real memory for its randomly-initialized
weights, even though no checkpoint is ever read. Step 2 additionally spends
real CPU/GPU time writing to every element.

Since fake_forward.py's ModelRunner.forward hook never calls self.model(...)
-- it returns synthetic logits before the real model would ever run --
nothing downstream ever reads these parameter values. So there is no reason
to give them real storage at all.

This hook rebuilds DummyModelLoader.load_model() to construct the model
under torch.device("meta") instead of the real device. Meta tensors carry
correct shape/dtype/device metadata -- so KV-cache sizing, num-params
logging, tied-weight assignment, etc. all keep working -- but allocate zero
real bytes of storage and cost zero time to "fill." initialize_dummy_weights
and process_weights_after_loading are skipped entirely: both would try to
write real values into tensors that have no storage to write into, and
neither is needed since the values are never read.

It also computes what those weights WOULD have cost in real VRAM -- sum of
numel * element_size across every parameter, which meta tensors report
correctly even with no real storage behind them -- and stashes that number
directly on current_platform, where fake_memory.py's device.py-backed
get_current_memory_usage() reads it back. Because SGLang's own
ColumnParallelLinear/RowParallelLinear etc. compute their LOCAL shard size
(divide(full_size, tp_size)) at __init__ time, this figure is automatically
correct per-rank under tensor/expert parallelism too -- no extra sharding
logic needed here.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_load = "dummy_srt_platform_plugin.fake_load:register"

Requires: launch with --load-format dummy (this hook only patches
DummyModelLoader; other load formats are untouched).

What's actually incompatible, verified against the source (not assumed):
  - --model-checksum: not affected at all. Checksum verification lives in
    _prepare_weights(), which belongs to the real download-and-read loader
    and explicitly raises if ever called under LoadFormat.DUMMY. It was
    already unreachable under plain --load-format dummy, meta device or not.
  - --kv-cache-dtype fp8_e4m3 + --quantization-param-path: verified SAFE.
    model.load_kv_cache_scales() (opt.py, llama.py, and every other in-tree
    model that implements it) writes the loaded value via plain Python
    attribute reassignment -- `layer.attn.k_scale = scaling_factor`, a bare
    float -- onto RadixAttention.k_scale, which starts as a plain `None`
    attribute, not an nn.Parameter or registered buffer. Bare `=`
    reassignment doesn't care what was there before, so this works
    identically whether the surrounding model is real or meta.
  - Quantization calibration (process_weights_after_loading): for an
    ordinary unquantized checkpoint this is already a no-op UNLESS the CPU
    has Intel AMX support, in which case it repacks weights into VNNI
    layout using real values. Since this platform targets CPU, that's the
    one CPU-specific crash risk if this call were made -- which is exactly
    why it's skipped unconditionally rather than only for quantized models.
    This is also where the *other* KV-cache-scale mechanism lives --
    BaseKVCacheMethod (layers/quantization/kv_cache.py), used when a
    checkpoint's own quantization config wires up k_scale/v_scale as real
    nn.Parameter tensors -- its process_weights_after_loading() calls
    .tolist() on them, which would fail on a meta tensor. Already covered
    by skipping process_weights_after_loading unconditionally; no separate
    guard needed. This is also the code path a "Detected fp8 checkpoint"
    (pre-quantized Hub repo, auto-detected from its own config.json) hits.
  - Speculative-decoding draft models and LoRA adapters load through
    separate paths (speculative_draft_load_format, lora_paths) that this
    hook does not touch -- they would still allocate real memory.
"""

import logging

import torch

from sglang.srt.model_loader.loader import (
    _get_quantization_config,
    _initialize_model,
    _post_load_weights,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def register():
    """Entry point called by load_plugins()."""
    HookRegistry.register(
        "sglang.srt.model_loader.loader.DummyModelLoader.load_model",
        _fake_load_model,
        HookType.AROUND,
    )
    logger.info("dummy_load plugin registered (meta-device model construction)")


def _fake_load_model(original_fn, self, *, model_config, device_config):
    """AROUND hook: never calls original_fn, so no real-sized weight tensor
    is ever allocated on the real device."""

    quant_config = _get_quantization_config(model_config, self.load_config)
    if quant_config is not None:
        # Not fatal -- construction still succeeds and nothing ever reads
        # these layers -- just worth knowing when it happens, since you're
        # pulling arbitrary checkpoints and some Hub repos are pre-quantized
        # (AWQ/GPTQ/FP8) without you passing --quantization explicitly.
        logger.warning(
            "%s resolved to quant_config=%s (likely auto-detected from the "
            "checkpoint's own config.json). Quantization-specific weight "
            "post-processing will be skipped since weights are never "
            "materialized under dummy_load.",
            model_config.model_path,
            type(quant_config).__name__,
        )

    with set_default_torch_dtype(model_config.dtype):
        with torch.device("meta"):
            model = _initialize_model(model_config, self.load_config, quant_config)

        # Deliberately NOT called, unlike the real DummyModelLoader:
        #   initialize_dummy_weights(model)     -- no storage to write into,
        #                                           and nothing ever reads it
        #   process_weights_after_loading(...)  -- needs real values, would
        #                                           raise on a meta tensor
        #                                           (e.g. AMX repack on CPU)
        _post_load_weights(model)  # cheap flag-setting fixups; safe on meta

    # Meta tensors have no data but correctly report shape/dtype -- this is
    # exactly what real weights (in the real target dtype -- fp8, bf16,
    # whatever the checkpoint actually is) would cost in VRAM, computed
    # without ever allocating it. For TP/EP > 1, `model` here already holds
    # only this rank's local shard (ColumnParallelLinear etc. size
    # themselves at __init__ time), so this is automatically a per-rank
    # figure, not the full model's.
    from sglang.srt.platforms import current_platform

    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    from dummy_srt_platform_plugin.device import _resident_weight_bytes_holder
    _resident_weight_bytes_holder["value"] = weight_bytes
    from collections import Counter

    bytes_by_dtype = Counter()
    for p in model.parameters():
        bytes_by_dtype[str(p.dtype)] += p.numel() * p.element_size()

    for dtype, b in sorted(bytes_by_dtype.items(), key=lambda x: -x[1]):
        logger.info("dummy_load: %s: %.2f GB", dtype, b / (1 << 30))

    logger.info(
        "dummy_load: this rank's (fake) weights would occupy %.2f GB of real VRAM",
        weight_bytes / (1 << 30),
    )

    return model.eval()