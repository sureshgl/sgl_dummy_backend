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

        Kept as a real, working implementation (DummyPiecewiseBackend) even
        though it is currently UNREACHABLE via SGLang's real tc_piecewise
        pipeline -- confirmed against current upstream source that
        PiecewiseCompileInterpreter (inside SGLangBackend, the torch.compile
        backend tc_piecewise installs) hardcodes CUDAPiecewiseBackend by
        name with no platform-lookup indirection at all, so this method is
        never actually consulted by that code path. support_piecewise_
        cuda_graph() below returns False specifically so tc_piecewise is
        never selected in the first place, avoiding that dead-end entirely
        (see that method's docstring for the full chain of reasoning:
        weak_ref_tensor is a compiled CUDA/NPU-only sgl-kernel op with no
        CPU path, and a Dynamo probe confirmed fake_attention.py's plain
        Python hooks are not compile-safe under fullgraph=True without
        first being wrapped as proper custom ops -- a separate, larger
        piece of follow-up work, not something to bolt on here).

        Left in place (rather than deleted) because the FX-graph cost-
        modeling logic in piecewise_backend.py is exactly what a future,
        properly-custom-op-wrapped compile path would want to reuse; only
        the SGLang-side wiring to reach it is currently absent.
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

        # NOTE: tc_piecewise / cuda_graph_tc_compiler defaults were removed
        # here (previously: cuda_graph_backend_prefill = "tc_piecewise",
        # cuda_graph_tc_compiler = "eager"). Confirmed via a real crash
        # chain (weak_ref_tensor.py raising NotImplementedError at import
        # time on CPU) plus source-level tracing that tc_piecewise entails
        # real CUDA graph capture end-to-end -- it is not separable into a
        # "just get Dynamo-split FX graphs" half and a "real CUDA graph
        # replay" half. support_piecewise_cuda_graph() below now returns
        # False, so server_args.py's own OOT-platform compatibility check
        # ("OOT platform without piecewise support") disables tc_piecewise
        # automatically; no explicit backend default is needed or wanted
        # here. Prefill now runs through the plain eager path
        # (EagerRunner), which fake_forward.py / fake_attention.py /
        # fake_quant.py already hook correctly.

    def init_backend(self) -> None:
        """One-time backend initialization."""
        logger.info("Dummy platform backend initialized")

    def support_cuda_graph(self) -> bool:
        """Whether this platform supports CUDA graph capture."""
        # CPU has no real CUDA graph capture capability.
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        """Whether this platform supports piecewise CUDA graph.

        Returns False. Reverted from an earlier Stage 3 attempt that
        returned True to unlock tc_piecewise, on the premise that
        DummyPiecewiseBackend (get_piecewise_backend_cls() above) would be
        consulted as the per-piece compiled callable. That premise does not
        hold against current upstream SGLang source:

        - PiecewiseCompileInterpreter (inside SGLangBackend, the
          torch.compile backend tc_piecewise's install_compile() wires up)
          hardcodes CUDAPiecewiseBackend by class name, with no platform
          hook or factory indirection at that point -- confirmed via
          inspection and via a real crash (weak_ref_tensor.py raising
          NotImplementedError at import time, deep inside
          CUDAPiecewiseBackend's own import chain, before any per-piece
          class selection could even matter).
        - Beyond the import-time failure, tc_piecewise's Capture phase does
          genuine torch.cuda.graph()-style CUDA graph recording -- a real
          hardware capability with no CPU equivalent to fall back to,
          confirmed against SGLang's own published PCG documentation
          ("captured as a separate CUDA graph", "output tensors of the
          last subgraph are stored as weak references to maximize memory
          reuse" -- weak_ref_tensor was itself migrated into sgl-kernel as
          a compiled CUDA/NPU-only op, per PR #12505). Upstream's own PCG
          compatibility list already auto-disables "Non-CUDA hardware (AMD
          ROCm, Ascend NPU)" for exactly this structural reason; CPU was
          never a candidate either.
        - A standalone Dynamo probe (torch.compile(model, backend=...) with
          a plain Python function containing time.sleep() spliced into the
          forward path, mirroring fake_attention.py's actual hook shape)
          confirmed that under permissive (non-fullgraph) compilation,
          Dynamo graph-breaks around the untraceable call -- but SGLang's
          real install_compile() uses fullgraph=True, under which an
          unregistered, untraceable call like this would hard-fail at
          trace time instead of gracefully splitting. Making
          fake_attention.py's hooks compile-safe under fullgraph=True would
          require converting them into properly declared custom ops
          (register_custom_op, with explicit split-point registration) --
          a real, separate piece of follow-up work, not something to bolt
          on to unblock this platform today.

        Returning False here means server_args.py's own compatibility rule
        ("OOT platform without piecewise support",
         lambda: current_platform.is_out_of_tree()
                 and not current_platform.support_piecewise_cuda_graph())
        disables tc_piecewise automatically for this platform, and prefill
        falls back to the plain eager path -- which fake_forward.py's
        ModelRunner.forward hook, fake_attention.py's DummyNativeAttnBackend
        hooks, and fake_quant.py's Fp8LinearMethod/Fp8MoEMethod hooks all
        already handle correctly, with no dependency on Dynamo tracing or
        CUDA graph capture at all.
        """
        return False