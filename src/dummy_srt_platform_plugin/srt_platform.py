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
        """Return the MHA KV pool class for this platform.

        Returns NoOpMHATokenToKVPool rather than the real MHATokenToKVPool.
        The real pool allocates k_buffer/v_buffer sized by
        max_total_num_tokens -- which, once DUMMY_GPU (device.py) makes
        current_platform.get_device_total_memory() report an arbitrarily
        large emulated VRAM figure, gets sized assuming that much memory is
        genuinely available. That's real torch.zeros(...) on the real "cpu"
        device: telling the scheduler "you have 141GB" doesn't summon 141GB,
        it just makes the real allocation attempt bigger, and it's exactly
        what triggered the SIGKILL/OOM this platform hit in practice.

        NoOpMHATokenToKVPool (sglang.srt.mem_cache.memory_pool) is an
        in-tree class built for a different purpose (embedding-mode
        prefill-only workloads that skip KV cache entirely) but the
        contract is exactly what we need here: self.size (the logical
        capacity used for scheduling/admission) is set in the KVCache base
        class before _create_buffers() ever runs, so admission decisions
        still see the full (emulated) capacity -- while the actual
        k_buffer/v_buffer are tiny, constant-size KB-scale placeholders
        regardless of size. set_kv_buffer() raises loudly if ever called;
        get_key_buffer/get_value_buffer return the placeholder rather than
        raising. Since fake_forward.py guarantees the real model (and
        therefore the real attention layers that would call these) never
        runs, neither path is ever exercised in practice -- the raise is a
        safety net, not an expected code path. get_kv_size_bytes() also
        returns (0, 0), so downstream memory-accounting log lines report
        zero KV-cache usage honestly instead of a fake GB figure.

        Caveat: the "KV Cache is allocated. #tokens: N, K size: X GB, V
        size: Y GB" log line a real deployment prints does not appear in
        this form -- NoOpMHATokenToKVPool logs its own
        "KV Cache skipped (no-op pool)" line instead. Matching a real
        log's exact KV-allocation line text is not possible while also
        not allocating the real memory; this is the honest tradeoff.
        """
        from sglang.srt.mem_cache.memory_pool import NoOpMHATokenToKVPool
        return NoOpMHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        """Return the MLA KV pool class for this platform.

        No no-op MLA pool exists in-tree yet (unlike MHA). Returning the
        real MLATokenToKVPool here would silently reproduce the exact same
        real-memory-allocation OOM that get_mha_kv_pool_cls() above exists
        to avoid, just for MLA-family models (e.g. DeepSeek) instead of
        MHA ones. Failing loudly here -- at pool-class-selection time --
        is far better than a confusing SIGKILL deep into scheduler
        initialization. Extending this to a real NoOpMLATokenToKVPool
        would follow the exact same pattern as NoOpMHATokenToKVPool
        (override _create_buffers() to allocate tiny placeholders instead
        of size-scaled buffers), but MLA's buffer shape depends on
        kv_lora_rank/qk_rope_head_dim/use_dsa/dsa_kv_cache_store_fp8, which
        needs verifying against an actual MLA model before writing it --
        not done here since none has been tested against this platform yet.
        """
        raise NotImplementedError(
            "DummySRTPlatform has no no-op MLA KV pool yet -- the real "
            "MLATokenToKVPool would allocate real memory sized by the "
            "emulated DUMMY_GPU VRAM, reproducing the same OOM "
            "get_mha_kv_pool_cls() avoids for MHA models. Add a "
            "NoOpMLATokenToKVPool (mirroring NoOpMHATokenToKVPool's "
            "_create_buffers() override) before using this platform with "
            "an MLA-family model."
        )

    def get_dsa_kv_pool_cls(self) -> type:
        """Return the DSA KV pool class for this platform.

        Same reasoning and same gap as get_mla_kv_pool_cls() above -- no
        in-tree no-op DSA pool exists yet, so fail loudly rather than
        silently reproduce the real-memory OOM for DSA-family models.
        """
        raise NotImplementedError(
            "DummySRTPlatform has no no-op DSA KV pool yet -- see "
            "get_mla_kv_pool_cls()'s docstring for why this raises instead "
            "of returning the real (memory-allocating) DSATokenToKVPool."
        )

    def get_paged_allocator_cls(self) -> type:
        """Return the paged allocator class for this platform.

        Unlike the KV pools above, the real PagedTokenToKVPoolAllocator is
        safe to use unmodified: it only tracks free/used page indices
        (O(size / page_size) bookkeeping, plus a fixed 1024-element warmup
        tensor) -- nowhere near the scale of a real K/V buffer, regardless
        of how large DUMMY_GPU makes size. No no-op variant needed.
        """
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