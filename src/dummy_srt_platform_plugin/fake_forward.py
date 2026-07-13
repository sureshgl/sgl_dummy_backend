"""
General SGLang plugin: let the real forward pass run (on meta tensors,
through torch.compile), then substitute real synthetic logits only at the
very end.

Stage 1/2 vs Stage 3
--------------------
Stage 1/2: this hook intercepted ModelRunner.forward() BEFORE
self.model.forward() was ever called -- the real model graph never
executed, so DummyPiecewiseBackend (wired up via
get_piecewise_backend_cls()) sat dormant, and there was no way to produce
per-piece, per-shape latency: only one undifferentiated sleep for the
whole forward pass.

Stage 3 changes the posture from "skip the call" to "let it run, replace
only the output": this hook now calls original_fn for real. That lets
Dynamo actually trace the real model (on torch.device("meta") tensors, so
no real memory is ever touched), hit real SGLang-authored split
boundaries, and invoke DummyPiecewiseBackend per compiled piece and
fake_attention.py's hooks for the real (always-eager, never-compiled)
attention calls in between. Every piece -- including attention -- stays
meta-in/meta-out, so the real forward pass produces a structurally correct
but valueless (meta) result.

That meta result isn't usable by the sampler, which needs real numbers.
So after original_fn returns, this hook substitutes the same hash-based
synthetic logits Stage 1/2 already used (_fake_logits_output, unchanged)
in place of whatever the (meta) real forward produced -- but only on the
PP rank that actually produces logits. Every other field on the real
output (can_run_graph, expert_distribution_metrics,
routed_experts_output, indexer_topk_output) is left exactly as the real
forward pass set it, since Stage 3 no longer needs to reconstruct those by
hand: the real code path already populated them correctly.

PP (pipeline parallel) is simpler under Stage 3 than Stage 1/2
-----------------------------------------------------------------
Stage 1/2 hand-built a zero-filled PPProxyTensors for intermediate PP
stages, because the real model never ran and nothing else would have
produced hidden states to hand to the next stage. Stage 3 doesn't need
that anymore: the real (meta) forward pass already produces a real
PPProxyTensors object with meta hidden-state tensors in it, via the same
real code path a genuine GPU deployment would use -- and the next PP
stage's real (meta) forward consumes those exactly as it would on a real
GPU. So intermediate PP stages need no special-casing at all here: only
the last-rank stage's logits get substituted with real (non-meta) values,
since sampling -- the first place real numbers are actually required --
only happens after the last PP stage.

dllm (diffusion LLM): unchanged from Stage 1/2 -- this hook still
intercepts every model_runner.forward() call the dllm algorithms make in
their loop, and _fake_logits_output still fills both full_logits and
next_token_logits so it works whether or not dllm is active.

Install alongside dummy_srt_platform_plugin, add the entry point below,
and launch with --load-format dummy so no real checkpoint needs to be
downloaded or read (the model config is still used, so vocab size /
hidden size / layer counts / KV pool sizing stay correct). For Stage 3
latency numbers to reflect real per-piece/per-attention-call timing,
also launch with --cuda-graph-backend-prefill tc_piecewise (defaulted by
DummySRTPlatform.apply_server_args_defaults, see srt_platform.py) so the
real torch.compile/Dynamo piecewise pipeline actually runs.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_forward = "dummy_srt_platform_plugin.fake_forward:register"

DUMMY_FORWARD_MODE controls what the fake logits look like (unchanged from
Stage 1/2):
    "hash"   (default) -- deterministic, derived from batch metadata that
              is identical across TP ranks. Safe for TP, DP, and PP.
    "fixed"  -- always favors DUMMY_FORWARD_TOKEN_ID (or token 0 if unset).
              Every request converges almost immediately -- useful for
              pure throughput/latency load testing.
    "random" -- uniform random logits. Cheapest, but only safe when
              tp_size == 1 (see TP note below).

TP (tensor parallel): No extra code needed. Each TP rank runs in its own
    scheduler subprocess, and load_plugins() runs independently in every
    subprocess, so this hook is applied on every rank automatically.
    Correctness caveat unchanged from Stage 1/2: real TP relies on
    collectives to keep logits identical across ranks before sampling.
    This hook's own logits substitution skips those collectives, so only
    use DUMMY_FORWARD_MODE="hash" or "fixed" when tp_size > 1 -- both are
    derived from batch metadata already identical across ranks. "random"
    draws independently per process and WILL desync ranks under TP.
"""

import logging
import os

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
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
    logger.info("dummy_forward plugin registered (mode=%s, Stage 3 post-forward substitution)", _MODE)


def _fake_forward(original_fn, self, forward_batch, *args, **kwargs):
    """AROUND hook: DOES call original_fn now (Stage 3) -- the real
    model.forward() runs, on meta tensors, through torch.compile/Dynamo,
    hitting DummyPiecewiseBackend at every real compiled-piece boundary and
    fake_attention.py's hooks for the real (eager, never-compiled)
    attention calls in between. Only the final logits value is replaced,
    and only on the PP rank that actually produces them."""
    output = original_fn(self, forward_batch, *args, **kwargs)

    if not self.pp_group.is_last_rank:
        # Intermediate PP stage: the real (meta) forward pass already built
        # a correct PPProxyTensors with meta hidden-state tensors in it --
        # nothing to substitute here. The next PP stage's real (meta)
        # forward consumes it exactly as it would on a real GPU.
        return output

    device = self.device
    num_tokens = forward_batch.input_ids.shape[0]
    vocab_size = self.model_config.vocab_size

    # Replace only the logits value; every other field on `output`
    # (can_run_graph, expert_distribution_metrics, routed_experts_output,
    # indexer_topk_output) was already set correctly by the real forward
    # pass that just ran, so it's left untouched rather than reconstructed.
    output.logits_output = _fake_logits_output(forward_batch, num_tokens, vocab_size, device)
    return output


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