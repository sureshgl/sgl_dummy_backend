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
attribute access doesn't fail. Returning None is intentionally the
simplest possible stand-in.

VERIFY BEFORE RELYING ON THIS: whether self._pool (the attribute this
sets) is used ANYWHERE ELSE in tc_piecewise_cuda_graph_backend.py besides
being stored -- e.g. passed into a later torch.cuda.graph(pool=self._pool)
call, which would ALSO need patching/faking for a CPU device, and which
this plugin does not address. Run:

    import inspect
    from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
        TcPiecewiseCudaGraphBackend,
    )
    print(inspect.getsource(TcPiecewiseCudaGraphBackend))

and grep the output for every other use of `self._pool` before assuming
this one-line patch is sufficient to get past compile-pass setup entirely.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_graph_pool = "dummy_srt_platform_plugin.fake_graph_pool:register"
"""

import logging

import torch

logger = logging.getLogger(__name__)


def register():
    """Entry point called by load_plugins()."""
    if not hasattr(torch.cpu, "graph_pool_handle"):
        torch.cpu.graph_pool_handle = lambda: None
        logger.info(
            "dummy_graph_pool plugin registered (torch.cpu.graph_pool_handle "
            "no-op added)"
        )
    else:
        logger.info(
            "dummy_graph_pool plugin: torch.cpu.graph_pool_handle already "
            "exists, not overriding"
        )