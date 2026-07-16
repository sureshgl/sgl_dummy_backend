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
        """Return the default attention backend for this platform.

        Returns "dummy_native", NOT "torch_native" -- confirmed against a
        real launch + real source that this matters, not just a naming
        preference. server_args.py's _handle_attention_backend_compatibility
        unconditionally disables BOTH cuda_graph_config.decode.backend and
        cuda_graph_config.prefill.backend whenever attention_backend
        resolves to the literal string "torch_native":

            if attention_backend == "torch_native":
                logger.warning("Cuda graph is disabled because of using torch native attention backend")
                self.cuda_graph_config.decode.backend = Backend.DISABLED
                self.cuda_graph_config.prefill.backend = Backend.DISABLED

        "dummy_native" resolves to DummyNativeAttnBackend
        (dummy_native_backend.py) -- a zero-behavior-change subclass of
        TorchNativeAttnBackend, registered by fake_attention.py's register()
        via the sanctioned add_attention_backend_choices() extension point.
        The check above is string-keyed, not class-identity-keyed (verified:
        TorchNativeAttnBackend is not special-cased anywhere else in the
        codebase), so this sidesteps it entirely with no other behavior
        change. fake_attention.py's hooks target DummyNativeAttnBackend
        specifically, not TorchNativeAttnBackend.
        """
        logger.info("DummySRTPlatform default attention backend: dummy_native")
        return "dummy_native"

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
        raising. get_kv_size_bytes() also returns (0, 0), so downstream
        memory-accounting log lines report zero KV-cache usage honestly
        instead of a fake GB figure.

        fake_forward.py's ModelRunner.forward hook never calls
        self.model(...) for real compute to produce values that would need
        real KV storage -- and fake_attention.py's hooks on
        DummyNativeAttnBackend.forward_extend/forward_decode never call
        set_kv_buffer/get_key_buffer/get_value_buffer either. So this
        pool's raise-on-real-use guarantee holds: neither path is ever
        exercised in practice, the raise is a safety net, not an expected
        code path.

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
        """Return the piecewise compilation backend class for this platform.

        Reachable again. Confirmed against real upstream source that
        make_backend() (sglang.srt.compilation.backend, the function
        PiecewiseCompileInterpreter.call_module() calls to construct each
        per-subgraph wrapper) already dispatches through this exact method
        for any current_platform.is_out_of_tree() platform:

            if current_platform.is_out_of_tree():
                backend_cls = current_platform.get_piecewise_backend_cls()
            elif is_xpu(): ... elif is_npu(): ... else: backend_cls = CUDAPiecewiseBackend

        There is no hardcoding to work around here -- this was previously
        marked unreachable because support_piecewise_cuda_graph() returned
        False, which meant tc_piecewise was never selected in the first
        place (a separate gate, upstream in server_args.py), not because
        this dispatch site itself was broken. See
        support_piecewise_cuda_graph()'s docstring for what actually
        blocked reachability, and why both blockers are now resolved rather
        than worked around.
        """
        from dummy_srt_platform_plugin.piecewise_backend import DummyPiecewiseBackend
        return DummyPiecewiseBackend

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply platform-specific default values to server arguments."""
        # Force CPU device
        logger.info("Applying dummy platform defaults: device=cpu")
        if not hasattr(server_args, 'device') or server_args.device != "cpu":
            server_args.device = "cpu"

        # Explicitly force the attention backend to "dummy_native".
        # get_default_attention_backend() returning "dummy_native" is NOT
        # sufficient on its own -- confirmed across multiple real launches
        # that attention_backend still resolves to SGLang's own generic
        # "torch_native" default unless this is set explicitly here.
        # Without this, TorchNativeAttnBackend (not DummyNativeAttnBackend)
        # is what actually gets instantiated, fake_attention.py's hooks
        # target a class that's never constructed, and the real
        # forward_extend/forward_decode run for real -- hitting
        # NoOpMHATokenToKVPool's intentional raise on the very first real
        # request. Treats "torch_native" as the sentinel meaning "not
        # deliberately chosen by the user", same as this platform already
        # treats it as the trigger string worth avoiding elsewhere
        # (dummy_native_backend.py). This can't perfectly distinguish an
        # explicit user "--attention-backend torch_native" from SGLang's
        # own un-set dataclass default -- both look identical here -- but
        # since this platform categorically cannot support real
        # torch_native behavior anyway (no GPU), overriding both cases to
        # dummy_native is the safe choice.
        if getattr(server_args, "attention_backend", None) in (None, "torch_native"):
            server_args.attention_backend = self.get_default_attention_backend()

        # NOTE: tc_piecewise / cuda_graph_tc_compiler defaults are still not
        # force-set here. support_piecewise_cuda_graph() below now returns
        # True again -- but this time because the two concrete blockers
        # that caused the earlier revert (weak_ref_tensor's import-time
        # crash, and real torch.cuda.graph() capture with no CPU
        # equivalent) both have real fixes in place rather than being
        # worked around -- see that method's docstring for the full chain
        # of reasoning. Since tc_piecewise is documented as SGLang's own
        # enabled-by-default behavior once no compatibility rule vetoes it,
        # and no rule does for this platform (see below), no explicit
        # backend default should be needed here either -- but this has NOT
        # yet been confirmed against a real launch. VERIFY: does the
        # resolved cuda_graph_config.prefill.backend actually come back as
        # "tc_piecewise" on an actual launch with this platform, or does
        # some other rule still silently reset it to "disabled"? If so, set
        # server_args.cuda_graph_config.prefill.backend explicitly here.

    def init_backend(self) -> None:
        """One-time backend initialization."""
        logger.info("Dummy platform backend initialized")

    def support_cuda_graph(self) -> bool:
        """Whether this platform supports CUDA graph capture."""
        # CPU has no real CUDA graph capture capability.
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        """Whether this platform supports piecewise CUDA graph.

        Returns True. Reversed again from the prior revert-to-False state --
        this time because the two concrete blockers that caused that revert
        each have a real fix in place, not a workaround:

        1. weak_ref_tensor.py raising NotImplementedError at import time.
           Fixed: __init__.py's activate() now stubs
           sglang.srt.compilation.weak_ref_tensor in sys.modules before
           anything can import sglang.srt.compilation.backend (which
           unconditionally imports cuda_piecewise_backend.py, which
           unconditionally imports weak_ref_tensor.py). Confirmed this
           import chain was the ENTIRE crash -- get_piecewise_backend_cls()
           was never even reached, so no per-class dispatch logic was ever
           actually at fault.

        2. Real torch.cuda.graph()-style CUDA graph capture, with no CPU
           equivalent. Confirmed by reading CUDAPiecewiseBackend's real
           source directly: torch.cuda.CUDAGraph() and the
           torch.cuda.graph(...) context manager live ENTIRELY inside
           CUDAPiecewiseBackend.__call__, and nothing outside that class --
           not SGLangBackend, not TcPiecewiseCudaGraphBackend, not
           PrefillCudaGraphRunner -- ever inspects a piecewise-backend
           instance's internal capture state (ConcreteSizeEntry.cudagraph /
           .output / .concrete_size_entries). The only contract any caller
           relies on is "call this object with args, get back a
           tensor/tuple matching shape/dtype". get_piecewise_backend_cls()
           below already substitutes DummyPiecewiseBackend for the whole
           class, so real capture is never faked -- it's bypassed by
           construction, and confirmed safe to bypass.

        The remaining CUDA-graph-adjacent primitives this path touches are
        each already handled, for reasons that don't require faking real
        driver behavior:
          - device_module.graph_pool_handle() -- stubbed as a no-op on
            torch.cpu by fake_graph_pool.py.
          - the CUDA stream object threaded through capture_session() --
            never dereferenced for real stream semantics once (2) above is
            bypassed; any opaque handle satisfies the bookkeeping.
          - set_graph_pool_id() (pynccl_allocator) -- a bare Python global
            assignment, not a driver call; harmless on any device.
          - self._device_module.synchronize() -- generic across
            torch.get_device_module(), already a no-op on CPU.

        Also confirmed: the OTHER server_args.py auto-disable rule for this
        feature -- "non-CUDA hardware (HIP/NPU/CPU/MPS/XPU)" -- does not
        fire for this platform either. That rule's is_cpu()
        (sglang.srt.utils.common.is_cpu) is gated on the
        SGLANG_USE_CPU_ENGINE=1 environment variable, not on host
        architecture -- confirmed by reading the function directly. It
        targets SGLang's own in-tree CPU engine, not an OOT platform that
        happens to run on CPU hardware; this platform never sets that env
        var, so is_hip()/is_npu()/is_cpu()/is_mps()/is_xpu() all evaluate
        False here regardless of the real underlying device.

        NOT resolved by this change -- deliberately out of scope here:
        fake_attention.py's and fake_quant.py's hooks are plain Python
        monkeypatches, not registered custom ops. A standalone Dynamo probe
        already confirmed these break fullgraph=True tracing (install_
        compile()'s hard requirement) rather than gracefully graph-breaking
        around them. Returning True here makes tc_piecewise SELECTABLE
        again and unblocks it at the platform-capability layer; it does
        NOT by itself make a real forward pass through
        Qwen3-Coder-480B-A35B trace successfully under fullgraph=True.
        Expect a Dynamo trace-time failure at the first untraceable hook
        until those are converted to register_custom_op-wrapped ops -- a
        separate, larger piece of follow-up work, not bundled into this
        change.
        """
        return True