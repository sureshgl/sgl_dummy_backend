"""
General SGLang plugin: fake the token-embedding lookup.

Why this exists
----------------
Confirmed via two real launches + real tracebacks: with
support_piecewise_cuda_graph() returning True, TcPiecewiseCudaGraphBackend's
compile-pass warmup (_run_compile_pass -> PrefillCudaGraphRunner.
_run_dummy_forward) calls self.model_runner.model.forward(...) DIRECTLY --
one level below ModelRunner.forward, so fake_forward.py's hook is bypassed
entirely for this call, exactly as fake_quant.py's docstring already
documented for Fp8LinearMethod/Fp8MoEMethod.

That direct call traces the real model with Dynamo. The very first op it
hits -- before any split point, before DummyPiecewiseBackend is ever
reached -- is VocabParallelEmbedding.forward() (vocab_parallel_embedding.py:
501). Confirmed this needed TWO rounds, not one, because the same root
cause (a real, ordinary "cpu" tensor meeting a torch.device("meta") tensor
under Dynamo's fake-tensor propagation, which cannot unify the two devices
for a single op) recurs at successive lines inside that one method:

  Round 1 (fixed by hooking UnquantizedEmbeddingMethod.embedding() alone):
    output_parallel = self.quant_method.embedding(self, masked_input.long())
    -- layer.weight is meta (fake_load.py); masked_input is real cpu
    (derived from real input_ids). Crashed inside F.embedding itself.

  Round 2 (this real launch): with quant_method.embedding() now faked and
    returning a meta tensor, tracing got one line further and crashed on
    the VERY NEXT real line in the same method:
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    output_parallel is now meta (round 1's fake); input_mask is real cpu
    (built by get_masked_input_and_mask() from the real input_ids). Same
    "Unhandled FakeTensor Device Propagation... found two different
    devices meta, cpu" failure, one line later.

Rather than continue patching line-by-line (the method also ends with
tensor_model_parallel_all_reduce(output_parallel) / attn_tp_all_reduce(...)
for tp_size > 1 -- a real collective on what would again be a meta tensor,
almost certainly the NEXT thing to crash if only masked_fill_ were
patched), this hook replaces the ENTIRE VocabParallelEmbedding.forward()
method instead -- consistent with how fake_attention.py hooks whole
forward_extend/forward_decode methods and fake_quant.py hooks whole
apply() methods, rather than patching individual real sub-calls inside
them. Nothing inside forward() -- maybe_detect_oob's real assertion,
get_masked_input_and_mask's real mask construction, the quant_method call,
the masking, the TP all-reduce -- ever runs under this hook; the real
input_ device is irrelevant.

The original inner hook (on UnquantizedEmbeddingMethod.embedding) is left
registered too, as defense in depth for any other real call site that
might invoke quant_method.embedding() directly without going through
VocabParallelEmbedding.forward() -- not confirmed to exist, but harmless
to leave in place since it will simply never fire if forward() is already
replaced.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_embedding = "dummy_srt_platform_plugin.fake_embedding:register"
"""

import logging

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def register():
    """Entry point called by load_plugins()."""
    HookRegistry.register(
        "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding.forward",
        _fake_vocab_parallel_embedding_forward,
        HookType.AROUND,
    )
    logger.info("dummy_embedding plugin registered (VocabParallelEmbedding.forward faked)")

    HookRegistry.register(
        "sglang.srt.layers.quantization.unquant.UnquantizedEmbeddingMethod.embedding",
        _fake_embedding,
        HookType.AROUND,
    )
    logger.info("dummy_embedding plugin registered (UnquantizedEmbeddingMethod.embedding faked, defense in depth)")


def _fake_vocab_parallel_embedding_forward(original_fn, self, input_: torch.Tensor) -> torch.Tensor:
    """AROUND hook: never calls maybe_detect_oob, get_masked_input_and_mask,
    quant_method.embedding, masked_fill_, or the TP all-reduce -- so no real
    tensor derived from input_ ever meets a meta tensor under Dynamo tracing
    at any point in this method.

    Real VocabParallelEmbedding.forward() returns shape
    (*input_.shape, self.embedding_dim) regardless of tp_size (masking
    zeroes rows in place, and the TP all-reduce sums across ranks -- neither
    changes shape), dtype = self.weight.dtype (confirmed against real
    source: quant_method.embedding's real F.embedding(input_, layer.weight)
    output dtype follows the weight, and this class's own extra_repr()
    confirms self.embedding_dim is the un-sharded per-row size). Always
    returned on torch.device("meta"), consistent with every other piece in
    this pipeline staying meta-in/meta-out.
    """
    return torch.empty(
        (*input_.shape, self.embedding_dim), dtype=self.weight.dtype, device="meta"
    )


def _fake_embedding(original_fn, self, layer, input_: torch.Tensor) -> torch.Tensor:
    """AROUND hook: defense in depth only -- see module docstring. Never
    calls F.embedding, so the meta-weight / real-input device mismatch
    never reaches Dynamo's fake-tensor propagation, for any call site that
    reaches this method directly rather than through
    VocabParallelEmbedding.forward().

    Real UnquantizedEmbeddingMethod.embedding() returns shape
    (*input_.shape, embedding_dim), dtype = layer.weight.dtype (confirmed
    against real source: F.embedding(input_, layer.weight) -- embedding_dim
    is layer.weight's last dim, this rank's shard per-partition size,
    already TP-sharded at __init__ time same as every other layer this
    project's hooks rely on). Always returned on torch.device("meta").
    """
    embedding_dim = layer.weight.shape[-1]
    dtype = layer.weight.dtype
    return torch.empty((*input_.shape, embedding_dim), dtype=dtype, device="meta")