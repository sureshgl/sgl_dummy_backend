"""
General SGLang plugin: give torch.cpu a graph_pool_handle() no-op.

Why this exists
----------------
TcPiecewiseCudaGraphBackend._run_compile_pass (SGLang's own tc_piecewise
compile-pass setup, not anything in this project's plugins) unconditionally
calls self._device_module.graph_pool_handle() -- confirmed via a real
launch traceback. self._device_module resolves to torch.get_device_module(
self.device), which for this platform's device="cpu" is the real torch.cpu
module. torch.cpu has no graph_pool_handle -- that API is CUDA-specific
(used to let multiple captured CUDA graphs share a memory arena), and has
no CPU equivalent because CPU has no such concept.

This is a genuine gap in SGLang's own tc_piecewise setup for OOT platforms
that opt into support_piecewise_cuda_graph()=True while
support_cuda_graph()=False (this platform's exact combination) -- upstream
appears to assume any platform requesting tc_piecewise also has a device
module exposing this CUDA-graph-pooling primitive.

Since DummyPiecewiseBackend (piecewise_backend.py) never performs real
CUDA graph capture -- every piece stays meta-in/meta-out and the only real
side effect is time.sleep() -- the VALUE returned by graph_pool_handle()
is never meaningfully consumed for actual memory-pooling purposes on this
platform; self._pool just needs to hold *something* so downstream
attribute access doesn't fail.

CONFIRMED WRONG, via a real launch traceback: returning None is NOT a safe
stand-in. get_or_create_global_graph_memory_pool() caches whatever this
returns and threads it straight into
install_torch_compiled()'s backend_factory, which constructs
SGLangBackend(compile_config, graph_pool) -- and SGLangBackend.__init__
(compilation/backend.py:387) unconditionally asserts
`assert graph_pool is not None`. Returning None satisfies "give me
something" for the missing-method crash but fails this separate assertion
one call later:

    AssertionError (compilation/backend.py:387, assert graph_pool is not None)

Fixed by returning a plain sentinel object instead -- anything that is not
None satisfies the assert, and nothing downstream ever dereferences it for
real CUDA-pool semantics: it's just stored as self.graph_pool and threaded
through make_backend()/DummyPiecewiseBackend.__init__ as an opaque value,
never read again (confirmed by grepping backend.py, tc_piecewise_cuda_
graph_backend.py, and piecewise_backend.py for any other use of
graph_pool/self._pool besides storage).

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_graph_pool = "dummy_srt_platform_plugin.fake_graph_pool:register"
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Fixed sentinel, not a fresh object() per call -- get_or_create_global_
# graph_memory_pool() only calls graph_pool_handle() once and caches the
# result (resources.graph_memory_pool), but returning the same sentinel
# every call is one less thing to reason about if anything ever calls
# this more than once, and is just as cheap.
_POOL_SENTINEL = object()


def register():
    """Entry point called by load_plugins()."""
    if not hasattr(torch.cpu, "graph_pool_handle"):
        torch.cpu.graph_pool_handle = lambda: _POOL_SENTINEL
        logger.info(
            "dummy_graph_pool plugin registered (torch.cpu.graph_pool_handle "
            "no-op added, returns a non-None sentinel)"
        )
    else:
        logger.info(
            "dummy_graph_pool plugin: torch.cpu.graph_pool_handle already "
            "exists, not overriding"
        )