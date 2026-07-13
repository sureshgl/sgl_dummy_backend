"""
Thin subclass of TorchNativeAttnBackend, registered under a new attention
backend name ("dummy_native") rather than reusing "torch_native".

Why this exists
----------------
Confirmed against real SGLang source (server_args.py,
_handle_attention_backend_compatibility): whenever attention_backend
resolves to the literal string "torch_native", SGLang unconditionally
disables BOTH cuda_graph_config.decode.backend and
cuda_graph_config.prefill.backend, with no override flag anywhere in that
path:

    if attention_backend == "torch_native":
        logger.warning("Cuda graph is disabled because of using torch native attention backend")
        self.cuda_graph_config.decode.backend = Backend.DISABLED
        self.cuda_graph_config.prefill.backend = Backend.DISABLED

This runs AFTER current_platform.apply_server_args_defaults() (confirmed by
call order: apply_server_args_defaults at server_args.py:2786,
_handle_attention_backend_compatibility at line 2802), so it silently stomps
DummySRTPlatform's own "tc_piecewise" default back to "disabled" every time
-- which is exactly what an actual launch showed: ServerArgs printed
cuda_graph_backend_prefill='tc_piecewise', but the final resolved
CudaGraphConfig showed prefill.backend='disabled', and the whole Stage 3
torch.compile/DummyPiecewiseBackend/fake_attention pipeline never actually
ran.

The check is keyed on the literal string "torch_native", not on class
identity -- confirmed by grepping the whole codebase for
TorchNativeAttnBackend and "torch_native": the ONLY place that string drives
this behavior is that one comparison. So a subclass registered under a
different name sidesteps it entirely, with zero other behavior change --
this class inherits forward_extend/forward_decode/support_triton/etc.
unmodified from TorchNativeAttnBackend. Both forward methods are faked
anyway by fake_attention.py's hooks, which target THIS class specifically
(not TorchNativeAttnBackend), so nothing about this class's own logic is
ever actually exercised at runtime regardless.

Registered into sglang.srt.layers.attention.attention_registry.ATTENTION_BACKENDS
under the name "dummy_native" by fake_attention.py's register(), using the
sanctioned add_attention_backend_choices() extension point
(server_args.py) so "dummy_native" is also a valid --attention-backend CLI
choice, not just an internal default.
"""

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend


class DummyNativeAttnBackend(TorchNativeAttnBackend):
    """Identical to TorchNativeAttnBackend in every respect except its class
    identity -- which is the entire point. See module docstring."""

    pass