"""
General SGLang plugin: teach support_triton() that "dummy_native" is not
Triton-capable, same as "torch_native" already is.

Why this exists
----------------
sglang.srt.utils.common.support_triton is a plain string allowlist:

    def support_triton(backend: str) -> bool:
        return backend not in ["torch_native", "intel_amx"]

write_cache_indices (sglang.srt.mem_cache.common) uses this exact check to
decide whether to call the real write_req_to_token_pool_triton Triton
kernel, or fall back to a plain Python/torch loop
(req_to_token_pool.write(...), no Triton at all) -- confirmed via a real
crash + real source read (mem_cache/common.py:122-160): the fallback
branch already exists and is fully CPU-safe.

"torch_native" is correctly recognized by this check. But this platform's
attention backend is registered as "dummy_native" specifically to sidestep
a DIFFERENT hardcoded string check
(_handle_attention_backend_compatibility's `== "torch_native"`, see
dummy_native_backend.py) -- and support_triton has no idea "dummy_native"
is a torch_native-derived backend. "dummy_native" not in
["torch_native", "intel_amx"] evaluates True, so write_cache_indices takes
the Triton branch and crashes with "0 active drivers" on a CPU-only box,
same failure shape as the earlier FP8 Triton crash, just in scheduler
bookkeeping instead of model compute.

This is scheduler-internal, model-agnostic code -- it runs on the very
first prefill batch, before any model forward pass, and would crash
identically regardless of which model or quant scheme is loaded.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_support_triton = "dummy_srt_platform_plugin.fake_support_triton:register"
"""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def register():
    """Entry point called by load_plugins()."""
    HookRegistry.register(
        "sglang.srt.utils.common.support_triton",
        _fake_support_triton,
        HookType.AROUND,
    )
    logger.info("dummy_support_triton plugin registered (dummy_native treated as non-Triton)")


def _fake_support_triton(original_fn, backend: str) -> bool:
    """AROUND hook: treat "dummy_native" exactly like "torch_native" --
    both are TorchNativeAttnBackend-family backends with no real Triton
    kernel usage anywhere in their own forward_extend/forward_decode
    (confirmed: DummyNativeAttnBackend is a zero-behavior-change subclass
    of TorchNativeAttnBackend, and fake_attention.py's hooks never call
    Triton either). Every other backend name is left completely untouched."""
    if backend == "dummy_native":
        return False
    return original_fn(backend)