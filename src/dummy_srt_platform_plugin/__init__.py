"""
Dummy SRT Platform Plugin

Provides a CPU-compatible dummy platform for testing and development.
Register via setuptools entry_points under sglang.srt.platforms.
"""

import logging
import sys

logger = logging.getLogger(__name__)


def _stub_weak_ref_tensor_module() -> None:
    """
    Pre-empt sglang.srt.compilation.weak_ref_tensor's own import-time crash
    on CPU, BEFORE anything can import sglang.srt.compilation.backend.

    Confirmed against real source: weak_ref_tensor.py is a compiled
    sgl-kernel custom op with CUDA/NPU-only implementations -- it raises
    NotImplementedError at import time on any other device. backend.py
    unconditionally imports cuda_piecewise_backend.py at its own module top
    level, which in turn unconditionally imports weak_ref_tensor at ITS top
    level. So merely importing backend.py -- which we need for real,
    since SGLangBackend / split_graph / PiecewiseCompileInterpreter are all
    genuinely CUDA-agnostic and are exactly what "mimic everything except
    compute" wants to exercise for real -- crashes before
    current_platform.get_piecewise_backend_cls() is ever consulted, entirely
    independent of which piecewise backend class ends up selected.

    The real weak_ref_tensors(tensors) converts real CUDA tensor(s) to weak
    references so a captured CUDA graph's memory pool can reclaim them --
    a real-memory-management optimization with nothing to reclaim here,
    since DummyPiecewiseBackend (get_piecewise_backend_cls() below, via
    srt_platform.py) never enters CUDAPiecewiseBackend.__call__ at all --
    the whole class is substituted, not patched. So this stub's
    identity-function body should never actually run in practice; it only
    needs to exist so the import chain succeeds. If it somehow WERE called,
    identity is still correct: meta tensors have no real storage for a weak
    reference to ever meaningfully manage.

    Must be called from activate() -- the earliest hook SGLang's plugin
    discovery calls -- not from a HookRegistry-based general plugin
    (sglang.srt.plugins entry points load later, via load_plugins(), by
    which point something else may already have imported backend.py).
    """
    import types

    module_name = "sglang.srt.compilation.weak_ref_tensor"
    if module_name in sys.modules:
        # Already imported (real module, or a stub from a prior activate()
        # call in this same process) -- don't clobber either case.
        return

    stub = types.ModuleType(module_name)

    def weak_ref_tensors(tensors):
        """Identity stand-in. See _stub_weak_ref_tensor_module's docstring:
        this should never actually be invoked, since DummyPiecewiseBackend
        never enters the code path that would call it -- kept as identity
        rather than raising, in case some other real SGLang code path ever
        imports this name directly for an unrelated reason."""
        return tensors

    stub.weak_ref_tensors = weak_ref_tensors
    sys.modules[module_name] = stub
    logger.info(
        "dummy_srt_platform_plugin: stubbed %s (no CPU/OOT kernel upstream) "
        "before any import of sglang.srt.compilation.backend",
        module_name,
    )


def activate():
    """
    Activation function for the dummy platform plugin.

    Called by the plugin discovery system to determine if this plugin should be
    activated on this machine. Always returns the plugin class name since the
    dummy platform runs on CPU.

    Returns:
        str: Fully-qualified class name for the dummy platform, or None if
            the hardware is not available (never happens for dummy CPU platform).
    """
    logger.info("Activating dummy SRT platform plugin")

    # Must run before ANY code path (ours or SGLang's own) can import
    # sglang.srt.compilation.backend -- see docstring above. activate() is
    # the earliest point plugin discovery calls into this package, so this
    # is the right place, not a general (sglang.srt.plugins) plugin's
    # register(), which loads later.
    _stub_weak_ref_tensor_module()

    # sgl_kernel ships CUDA-linked binaries only; on a CPU dummy platform
    # every consumer already treats it as optional, so short-circuit the
    # import instead of letting it re-probe GPU architecture every time.
    sys.modules.setdefault("sgl_kernel", None)
    return "dummy_srt_platform_plugin.srt_platform:DummySRTPlatform"
