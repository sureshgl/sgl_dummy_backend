"""
Dummy decode graph runner -- Option B (bucket-cached latency, no real
capture).

Why this exists
----------------
Confirmed against real SGLang source at commit
bf231f01a3b3ae3a9c1dc942bb4c52343e81f26b (sgl-project/sglang, main), across
a multi-step investigation -- summarized here since none of it is visible
from any single file:

1. support_cuda_graph() (srt_platform.py) is NOT the actual gate on decode
   graph usage. Repo-wide grep finds only two consumers, both in
   model_runner.py, both OR'd with a hardcoded device-string check
   (`self.device in ("cuda", "musa", "cpu", "npu")`, model_runner.py:918,
   1814-1819) that already matches this platform's device_type="cpu"
   regardless of that flag's value. Flipping it does nothing on its own.

2. Because DummySRTPlatform is OOT (_enum = PlatformEnum.OOT),
   init_decode_cuda_graph() (model_runner.py:2575-2590) always takes the
   `current_platform.get_graph_runner_cls()` branch, never the real
   `DecodeCudaGraphRunner` / `FullCudaGraphBackend` (real
   torch.cuda.CUDAGraph()) path used for genuine GPU deployments. That real
   path is unreachable from this platform by construction, so its
   "torch.cuda.graph() has no CPU equivalent" problem is moot here --
   irrelevant, not solved.

3. get_graph_runner_cls() previously returned the real, in-tree
   CPUGraphRunner (srt_platform.py). Confirmed unsound against a
   meta-device model: CPUGraphRunner.__init__ (cpu_graph_runner.py:
   632-649) allocates REAL (non-meta) static buffers on the real device,
   then calls capture() synchronously at startup, which calls
   self.model_runner.model.forward(...) DIRECTLY (bypassing
   ModelRunner.forward / fake_forward.py's hook entirely) on those real
   buffers -- an eager (non-Dynamo-traced) call. Real "cpu" input tensors
   reaching this plugin's meta-device weights (fake_load.py) in an
   ordinary eager op is a hard device-mismatch RuntimeError, not a
   graceful degrade: CPUGraphRunner cannot succeed against this platform's
   model, unlike prefill's tc_piecewise pipeline, which stays inside
   Dynamo's FakeTensorMode the whole time (see piecewise_backend.py).

4. Real DecodeCudaGraphRunner + FullCudaGraphBackend (the actual GPU
   mechanism, unreachable here per (2)) captures the ENTIRE decode step as
   one torch.cuda.CUDAGraph per shape bucket -- confirmed via
   full_cuda_graph_backend.py's own docstring, and via
   runner_backend/utils.py's resolve_decode_backend(), which explicitly
   falls back from Backend.TC_PIECEWISE to FullCudaGraphBackend with
   "cuda_graph_config decode='tc_piecewise' is not yet implemented" --
   i.e. even real SGLang has no piecewise decode capture today. So
   "one decode step is one monolithic, unsplit call" is a property of
   SGLang's own current design, on real GPUs, not something specific to
   this plugin's emulation.

5. EagerRunner._execute_decode (eager_runner.py:239) -- what this
   platform actually runs today whenever no decode graph runner exists --
   ALSO calls self.model_runner.model.forward(...) directly, uncompiled,
   every single decode token: no torch.compile/Dynamo boundary at all for
   decode, unlike prefill. That real per-layer eager Python dispatch,
   uncosted by any CostModel today, is the actual ~27 tok/s bottleneck
   this file addresses.

6. The exact call contract this class must satisfy (model_runner.py:
   3011-3031): only `can_run_graph(forward_batch) -> bool` and
   `execute(forward_batch, pp_proxy_tensors=None, **kwargs) -> ret` are
   ever called from outside, where `ret` becomes `ModelRunnerOutput(
   logits_output=ret, can_run_graph=True)` directly. `load_batch` is an
   internal-only helper on other runners, never invoked by _forward_raw
   itself -- not needed here.

7. Confirmed safe to fake: `ModelRunnerOutput.logits_output` is
   unconditionally overwritten by fake_forward.py's AROUND hook right
   after this whole chain returns (output.logits_output =
   _fake_logits_output(...)), regardless of what execute() produced --
   so a cache-hit's return value never needs real content, only a type
   that survives until then. The other three ModelRunnerOutput fields
   (expert_distribution_metrics, routed_experts_output,
   indexer_topk_output) are populated by separate, process-global
   recorder/capturer singletons INSIDE ModelRunner.forward(), driven by
   hooks into the model's real ops (model_runner.py:2938-2977) -- not by
   this class's return value at all, and None-gated
   (`if capturer is not None`) exactly as a real GPU's own CUDA-graph
   decode replay already leaves them on every graphed step. A cache-hit
   here is exactly as "incomplete" on these fields as a real GPU replay
   step already is; the one honest limitation is that expert-distribution
   / indexer-capture profiling (off by default, orthogonal to this
   plugin's purpose) will undercount on cache-hit steps.

8. TorchNativeAttnBackend.init_forward_metadata (inherited unmodified by
   DummyNativeAttnBackend, see dummy_native_backend.py) only reads
   forward_batch.out_cache_loc and self.token_to_kv_pool -- confirmed
   against real source (torch_native_backend.py:50-59) to have zero
   dependency on q/k/v or any model output. Safe to call on every
   execute() call, hit or miss.

Design (Option B, as agreed): unlike prefill's DummyPiecewiseBackend --
which fakes only the COMPUTE inside a real Dynamo capture/split, because
Dynamo's FakeTensorMode gives it a safe symbolic space to do real tracing
in -- decode has no such structure to preserve: it is one monolithic,
never-split call, and there is no FX graph here to walk for a roofline
estimate the way piecewise_backend.py's estimate_flops_and_bytes() does.
So this class does no real capture at all. Instead, per batch-size
bucket:
  - First call (cache MISS): run the real, uncompiled
    model_runner.model.forward(...) for real -- the same call
    EagerRunner already makes today -- letting every existing hook
    (fake_attention.py, fake_quant.py, fake_embedding.py, fake_rotary.py,
    fake_unified_attention.py) fire normally, staying meta-in/meta-out.
    Measure this call's real wall-clock latency (time.perf_counter()) and
    cache it under this bucket. This captures BOTH fake_attention.py's
    own roofline-estimated attention sleeps AND the real (uncosted,
    unavoidable) per-op Python dispatch overhead of every surrounding
    linear/MoE/norm layer, empirically -- there is no FX graph available
    here to derive an equivalent number analytically, unlike prefill.
  - Every subsequent call for that bucket (cache HIT), for the rest of the
    server's life: skip the real forward entirely, sleep the cached
    latency, and return a cheap meta-device stub -- eliminating exactly
    the real per-op dispatch overhead that is today's decode bottleneck.

Scope limits (mirrors CPUGraphRunner's own explicit assert list,
cpu_graph_runner.py:586-606, narrowed further since this runner does no
real capture at all): LoRA, two-batch-overlap, speculative decoding,
pipeline parallelism, encoder-decoder models, and elastic-EP rebalancing
are all asserted unsupported at construction/call time rather than
silently mishandled.

pyproject.toml: no separate entry point needed -- this class is wired in
directly by DummySRTPlatform.get_graph_runner_cls() (srt_platform.py),
the same way get_piecewise_backend_cls() already returns
DummyPiecewiseBackend directly rather than via a HookRegistry entry.
"""

import bisect
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _stub_logits_output(forward_batch: "ForwardBatch", vocab_size: int) -> LogitsProcessorOutput:
    """Cheap placeholder returned on a cache-HIT ("replay") call.

    Never read for its content -- see this module's docstring point (7):
    fake_forward.py's AROUND hook on ModelRunner.forward unconditionally
    overwrites output.logits_output right after this whole call chain
    returns, regardless of what's here. Built on torch.device("meta") so
    returning it never allocates real memory, matching this plugin's
    meta-in/meta-out convention everywhere else. Sized to the ACTUAL
    current batch (not the bucket) since that's what a correctly-shaped
    (if disposable) LogitsProcessorOutput requires.
    """
    batch_size = forward_batch.batch_size
    return LogitsProcessorOutput(
        next_token_logits=torch.empty(
            (batch_size, vocab_size), dtype=torch.float32, device="meta"
        ),
    )


class DummyDecodeGraphRunner:
    """
    Decode-side counterpart to DummyPiecewiseBackend, structurally
    DIFFERENT by design -- see module docstring point (8)/Design section
    above: no real capture, no FX graph, no compiled artifact. Just a
    per-batch-size-bucket cache of one real call's measured latency,
    replayed by sleeping thereafter.

    Not a BaseCudaGraphRunner subclass (that ABC's capture()/
    capture_prepare()/capture_one_shape() machinery is for the real
    tc_piecewise-family runners -- irrelevant here, same reason
    CPUGraphRunner itself is also a bare, non-inheriting class). Only
    `can_run_graph` and `execute` are ever called externally (see module
    docstring point (6)).
    """

    def __init__(self, model_runner: "ModelRunner") -> None:
        self.model_runner = model_runner
        self.device = model_runner.device
        sa = model_runner.server_args

        # Scope guards, mirroring CPUGraphRunner's own explicit assert
        # list (cpu_graph_runner.py:586-606) -- narrower here since this
        # runner does no real capture at all, so anything that would need
        # graph-capture-aware handling (PP proxy tensors, speculative
        # verify shapes, two-batch-overlap splitting, LoRA batch prep,
        # elastic-EP rebalancing, encoder-decoder cross-attention) is out
        # of scope rather than silently mishandled.
        assert not sa.enable_lora, "DummyDecodeGraphRunner does not support LoRA yet."
        assert not sa.enable_two_batch_overlap, (
            "DummyDecodeGraphRunner does not support two-batch overlap yet."
        )
        assert model_runner.spec_algorithm.is_none(), (
            "DummyDecodeGraphRunner does not support speculative decoding yet."
        )
        assert sa.pp_size == 1, "DummyDecodeGraphRunner does not support pipeline parallelism yet."
        assert not model_runner.model_config.is_encoder_decoder, (
            "DummyDecodeGraphRunner does not support encoder-decoder models yet."
        )
        assert not model_runner.enable_elastic_ep, (
            "DummyDecodeGraphRunner does not support elastic-EP rebalancing yet."
        )

        # Reuse real SGLang's own bucket-selection helper (base_cuda_graph_
        # runner.py) rather than reimplementing it -- same buckets a real
        # GPU deployment's DecodeCudaGraphRunner would capture for this
        # server_args.cuda_graph_config.decode.bs / max-running-requests
        # combination.
        self.capture_bs, _ = get_batch_sizes_to_capture(model_runner, num_tokens_per_bs=1)
        self.max_bs = max(self.capture_bs)

        # bucket -> latency (seconds), measured from the one real MISS
        # call made for that bucket. Instance-level dict, not a module-
        # level holder -- this project has already hit the cross-instance
        # singleton mutation bug twice (see gpu_spec.py / device.py's
        # holder-dict notes); one ModelRunner owns one
        # DummyDecodeGraphRunner instance for its whole lifetime, so
        # instance-level state is both sufficient and correctly scoped.
        self._bucket_latency: Dict[int, float] = {}

        logger.info(
            "DummyDecodeGraphRunner initialized (Option B: no real capture, "
            "bucket-cached latency only -- see module docstring); "
            "capture_bs=%s",
            self.capture_bs,
        )

    def _pad_to_bucket(self, raw_size: int) -> int:
        """Return the smallest captured bucket >= raw_size. can_run_graph
        must reject raw_size > max_bs before this is ever called (mirrors
        base_cuda_graph_runner.py's own _pad_to_bucket contract)."""
        assert raw_size <= self.max_bs, (
            f"size {raw_size} exceeds max bucket {self.max_bs}; "
            f"can_run_graph should have rejected this batch"
        )
        index = bisect.bisect_left(self.capture_bs, raw_size)
        return self.capture_bs[index]

    def can_run_graph(self, forward_batch: "ForwardBatch") -> bool:
        """Called from _forward_raw only after forward_batch.forward_mode.
        is_cpu_graph() (== is_decode(), model_runner.py:3006-3010) has
        already been confirmed True -- no need to re-check forward_mode
        here, matching CPUGraphRunner.can_run_graph's own posture."""
        return forward_batch.batch_size <= self.max_bs

    def execute(
        self,
        forward_batch: "ForwardBatch",
        pp_proxy_tensors: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """The only other externally-called method (see module docstring
        point (6)). Returns whatever should become
        ModelRunnerOutput.logits_output -- disposable on a cache hit, real
        on a cache miss (see _stub_logits_output's docstring and module
        docstring point (7))."""
        assert pp_proxy_tensors is None, (
            "DummyDecodeGraphRunner does not support pipeline parallelism yet."
        )

        # Pure pre-forward bookkeeping (out_cache_loc / SWA translation),
        # confirmed to have zero dependency on q/k/v or any model output
        # -- see module docstring point (8). Safe on every call, hit or
        # miss, exactly as EagerRunner._execute_decode already calls it
        # unconditionally.
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        bucket = self._pad_to_bucket(forward_batch.batch_size)

        if bucket not in self._bucket_latency:
            return self._run_real_and_cache(forward_batch, bucket)

        latency = self._bucket_latency[bucket]
        if latency:
            time.sleep(latency)

        return _stub_logits_output(
            forward_batch, self.model_runner.model_config.vocab_size
        )

    def _run_real_and_cache(self, forward_batch: "ForwardBatch", bucket: int) -> Any:
        """Cache MISS path: the one real, uncompiled forward call for this
        bucket -- identical in shape to EagerRunner._execute_decode's own
        call (eager_runner.py:239), so every existing per-layer hook fires
        exactly as it already does today. Measures real wall-clock time
        rather than deriving a roofline estimate analytically, since
        decode has no FX graph to walk the way piecewise_backend.py's
        estimate_flops_and_bytes() does for prefill (see module docstring
        Design section) -- this empirical measurement already includes
        fake_attention.py's own roofline-estimated attention sleeps plus
        the real (otherwise-uncosted) per-op dispatch overhead of every
        surrounding layer."""
        start = time.perf_counter()
        ret = self.model_runner.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )
        elapsed = time.perf_counter() - start
        self._bucket_latency[bucket] = elapsed

        logger.info(
            "DummyDecodeGraphRunner: bucket=%d first (real) call measured "
            "%.6fs; caching for future replay",
            bucket,
            elapsed,
        )
        return ret