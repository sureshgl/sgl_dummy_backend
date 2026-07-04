"""
General SGLang plugin: skip real model compute, return synthetic logits.

This is a *general plugin* (entry-point group "sglang.srt.plugins"),
separate from the hardware-platform plugin (entry-point group
"sglang.srt.platforms") that DummySRTPlatform already registers.

The platform interface (SRTPlatform / DeviceMixin) only controls WHICH
classes handle graph running, KV pools, and allocators -- it never touches
the forward-pass math itself. To skip the actual matmuls, hook the seam
where ModelRunner.forward() would normally call into the real model.

Everything upstream and downstream of this hook stays real: tokenization,
scheduling/batching, KV-cache bookkeeping, sampling, detokenization, and
the HTTP/OpenAI-compatible response path all run exactly as they would in
production. Only the transformer math is faked.

Coverage across execution modes
--------------------------------
TP (tensor parallel):   No extra code needed. Each TP rank runs in its own
    scheduler subprocess (engine.py spawns one mp.Process per tp_rank), and
    load_plugins() runs independently in every subprocess, so this hook is
    applied on every rank automatically. Correctness caveat: real TP relies
    on collectives to keep logits identical across ranks before sampling.
    Since this hook skips all real compute (including those collectives),
    only use DUMMY_FORWARD_MODE="hash" or "fixed" when tp_size > 1 --
    both are derived from batch metadata that is already identical across
    ranks. DUMMY_FORWARD_MODE="random" draws independently per process and
    WILL desync ranks (each would sample a different "next" token, corrupting
    KV-cache consistency across ranks).

DP (data parallel):    No extra code needed. Each DP replica is also an
    independent run_scheduler_process subprocess with its own plugin load,
    and replicas never need to agree with each other (they serve disjoint
    requests), so "random" mode is safe here even though it isn't for TP.

PP (pipeline parallel): Handled explicitly below. Only the last PP stage
    produces LogitsProcessorOutput; every earlier stage produces
    PPProxyTensors (a dict of hidden-state tensors) to hand to the next
    stage. A hook that always returns fake logits would break intermediate
    stages, so this hook checks self.pp_group.is_last_rank and returns a
    zero-filled PPProxyTensors({"hidden_states": ..., "residual": ...}) on
    non-last stages instead.

dllm (diffusion LLM):  No extra hook needed -- the dllm algorithms
    (sglang/srt/dllm/algorithm/*.py) call model_runner.forward() directly,
    in a loop, so this hook intercepts every one of those calls too.
    The only wrinkle: dllm reads logits from `logits_output.full_logits`
    (shape [num_tokens_in_block, vocab_size]), not `next_token_logits`
    (shape [batch_size, vocab_size]) which the normal AR sampler uses.
    This hook fills both fields so it works whether or not dllm is active,
    with no mode detection required.

Install alongside dummy_srt_platform_plugin, add the entry point below,
and launch with --load-format dummy so no real checkpoint needs to be
downloaded or read (the model config is still used, so vocab size /
hidden size / layer counts / KV pool sizing stay correct).

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_forward = "dummy_srt_platform_plugin.fake_forward:register"

Launch example (single stage, CPU, TP or DP is transparent to this hook):

    SGLANG_PLATFORM=dummy python -m sglang.launch_server \
        --model-path <hf-model-id-or-local-config-dir> \
        --load-format dummy \
        --device cpu

DUMMY_FORWARD_MODE controls what the fake logits look like:
    "hash"   (default) -- deterministic, derived from batch metadata that
              is identical across TP ranks. Safe for TP, DP, and PP.
    "fixed"  -- always favors DUMMY_FORWARD_TOKEN_ID (or token 0 if unset).
              Every request converges almost immediately -- useful for
              pure throughput/latency load testing.
    "random" -- uniform random logits. Cheapest, but only safe when
              tp_size == 1 (see TP note above).
"""

import logging
import os

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunnerOutput
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)

_MODE = os.environ.get("DUMMY_FORWARD_MODE", "hash")
_FIXED_TOKEN_ID = os.environ.get("DUMMY_FORWARD_TOKEN_ID")


def register():
    """Entry point called by load_plugins()."""
    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.forward",
        _fake_forward,
        HookType.AROUND,
    )
    logger.info("dummy_forward plugin registered (mode=%s)", _MODE)


def _fake_forward(original_fn, self, forward_batch, *args, **kwargs):
    """AROUND hook: never calls original_fn, so the real model.forward()
    (and every matmul inside it, on every PP stage) never runs."""
    logger.info("dummy_forward plugin _fake_forward invoked, original_fn=%s, forward_batch=%s", original_fn, forward_batch)
    device = self.device
    num_tokens = forward_batch.input_ids.shape[0]

    if not self.pp_group.is_last_rank:
        # Intermediate PP stage: downstream code expects hidden states to
        # forward to the next stage, not logits. hidden_size covers the
        # common "hidden_states" (+ "residual") convention used by most
        # PP-enabled model implementations; harmless if the next stage
        # doesn't look up "residual".
        hidden_size = self.model_config.hidden_size
        zeros = torch.zeros(num_tokens, hidden_size, device=device)
        proxy = PPProxyTensors({"hidden_states": zeros, "residual": zeros.clone()})
        return ModelRunnerOutput(logits_output=proxy, can_run_graph=False)

    vocab_size = self.model_config.vocab_size
    logits_output = _fake_logits_output(forward_batch, num_tokens, vocab_size, device)
    return ModelRunnerOutput(logits_output=logits_output, can_run_graph=False)


def _fake_logits_output(forward_batch, num_tokens, vocab_size, device):
    batch_size = forward_batch.batch_size

    if _MODE == "random":
        next_token_logits = torch.randn(batch_size, vocab_size, device=device)
        full_logits = torch.randn(num_tokens, vocab_size, device=device)
    elif _MODE == "fixed":
        token_id = int(_FIXED_TOKEN_ID) if _FIXED_TOKEN_ID else 0
        next_token_logits = _one_hot(batch_size, token_id, vocab_size, device)
        full_logits = _one_hot(num_tokens, token_id, vocab_size, device)
    else:
        # "hash": deterministic per-call, identical across TP ranks since
        # both seq_lens and input_ids are shared batch metadata, not
        # independently generated per rank.
        next_token_logits = _hash_logits(forward_batch.seq_lens, vocab_size, device)
        full_logits = _hash_logits(forward_batch.input_ids, vocab_size, device)

    return LogitsProcessorOutput(
        next_token_logits=next_token_logits,
        full_logits=full_logits,
    )


def _hash_logits(values: torch.Tensor, vocab_size: int, device) -> torch.Tensor:
    token_ids = (values.to(torch.int64) * 2654435761) % vocab_size  # Knuth hash
    n = token_ids.shape[0]
    logits = torch.full((n, vocab_size), -10.0, device=device)
    logits[torch.arange(n, device=device), token_ids] = 10.0
    return logits


def _one_hot(n: int, token_id: int, vocab_size: int, device) -> torch.Tensor:
    logits = torch.full((n, vocab_size), -10.0, device=device)
    logits[:, token_id] = 10.0
    return logits