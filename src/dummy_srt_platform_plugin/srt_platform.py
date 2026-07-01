"""
Dummy SRT Platform implementation.

Provides a CPU-compatible platform that reuses existing CPU-friendly SGLang
components (graph runner, KV pool, allocator) for testing and development.
"""

import logging

from sglang.srt.platforms.interface import SRTPlatform
from dummy_srt_platform_plugin.device import DummyDeviceMixin

logger = logging.getLogger(__name__)


class DummySRTPlatform(SRTPlatform, DummyDeviceMixin):
    """
    CPU-compatible dummy SRT platform.

    Reuses existing SGLang CPU components (CPUGraphRunner, MHATokenToKVPool, etc.)
    for a minimal, functional platform.
    """

    def get_default_attention_backend(self) -> str:
        """Return the default attention backend for this platform."""
        # Use torch_native for CPU; it's the safest CPU-compatible option
        return "torch_native"

    def get_graph_runner_cls(self) -> type:
        """Return the graph runner class for this platform."""
        # Use existing CPU graph runner
        from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
        return CPUGraphRunner

    def get_mha_kv_pool_cls(self) -> type:
        """Return the MHA KV pool class for this platform."""
        # Use existing CPU-compatible MHA pool
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        return MHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        """Return the MLA KV pool class for this platform."""
        # Use existing CPU-compatible MLA pool
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        return MLATokenToKVPool

    def get_dsa_kv_pool_cls(self) -> type:
        """Return the DSA KV pool class for this platform."""
        # Use existing CPU-compatible DSA pool
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        return DSATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        """Return the paged allocator class for this platform."""
        # Use existing paged allocator
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        return PagedTokenToKVPoolAllocator

    def get_piecewise_backend_cls(self) -> type:
        """Return the piecewise compilation backend class for this platform."""
        # Use our custom dummy piecewise backend that works on CPU
        from dummy_srt_platform_plugin.piecewise_backend import DummyPiecewiseBackend
        return DummyPiecewiseBackend

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply platform-specific default values to server arguments."""
        # Force CPU device
        logger.info("Applying dummy platform defaults: device=cpu")
        if not hasattr(server_args, 'device') or server_args.device != "cpu":
            server_args.device = "cpu"

    def init_backend(self) -> None:
        """One-time backend initialization."""
        logger.info("Dummy platform backend initialized")

    def support_cuda_graph(self) -> bool:
        """Whether this platform supports CUDA graph capture."""
        # CPU uses torch.compile instead of CUDA graphs
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        """Whether this platform supports piecewise CUDA graph."""
        # Dummy platform can support piecewise if torch.compile is enabled
        return False
