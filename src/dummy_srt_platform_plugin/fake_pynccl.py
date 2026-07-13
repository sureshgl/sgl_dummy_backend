"""
General SGLang plugin: skip PyNCCL communicator setup for the gloo backend.

init_model_parallel_group() (sglang.srt.distributed.parallel_state) decides
whether to attempt a PyNcclCommunicator like this:

    use_pynccl=(
        not (_is_npu or _is_xpu or backend == "mooncake")
        if use_pynccl is None else use_pynccl
    )

There's a carve-out for NPU, XPU, and the "mooncake" backend, but none for
plain CPU/gloo. PyNcclCommunicator.__init__ unconditionally does
`torch.cuda.device(device)` -- so the moment tp_size > 1 needs a real TP
GroupCoordinator, this crashes immediately with
"ValueError: Expected a cuda device, but got: cpu", during
GroupCoordinator construction, entirely before dummy_forward.py's hook (or
any forward pass) is even reachable. tp_size == 1 never hits this, because
GroupCoordinator only attempts it when `self.world_size > 1` -- which is
exactly why this never showed up until tp_size=2 was tried.

This hook forces use_pynccl=False whenever backend == "gloo", for every
group init_model_parallel_group is called for (world, tp, pp, dp, ...).
GroupCoordinator already handles use_pynccl=False correctly -- that's the
exact documented behavior for NPU/XPU -- so pynccl_comm simply stays None,
and any real collective (SGLang's own cross-rank token-id sync, or
fake_memory.py's own all_reduce for KV-cache sizing under TP) falls back to
plain torch.distributed over gloo, which is already proven to work
end-to-end in this project.

Gating on backend == "gloo" rather than checking current_platform means
this is correct for any gloo-backed platform, not just this one specific
dummy platform -- pynccl is inherently CUDA-NCCL-specific, so it can never
be valid to attempt it under gloo regardless of which platform selected
gloo as its backend.

init_model_parallel_group is only ever called from within
sglang.srt.distributed.parallel_state itself (verified against source --
not name-imported into any other module), so hooking it directly here is
not subject to the name-import-binding pitfall fake_memory.py's target
(get_available_gpu_memory) ran into.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_pynccl = "dummy_srt_platform_plugin.fake_pynccl:register"
"""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def register():
    """Entry point called by load_plugins()."""
    HookRegistry.register(
        "sglang.srt.distributed.parallel_state.init_model_parallel_group",
        _fake_init_model_parallel_group,
        HookType.AROUND,
    )
    print("dummy_pynccl plugin registered (pynccl disabled for gloo backend)")


def _fake_init_model_parallel_group(original_fn, group_ranks, local_rank, backend, **kwargs):
    """AROUND hook: previously gated on backend == "gloo" so this stayed
    correct for any gloo-backed platform, not just this one. Loosened to
    unconditional after a real launch showed PyNcclCommunicator still being
    constructed despite backend presumably being "gloo" for the world
    group -- meaning either a different subgroup receives a different
    backend string, or GroupCoordinator's own use_pynccl resolution no
    longer trusts this kwarg the way it did when this hook was written.
    Forcing use_pynccl=False unconditionally is safe here specifically
    because this hook only ever runs at all on a plugin-loaded process,
    and no legitimate real-GPU deployment would ever load this plugin."""
    logger.info("dummy_pynccl plugin: forcing use_pynccl=False for init_model_parallel_group (backend=%s)", backend)
    kwargs["use_pynccl"] = False
    return original_fn(group_ranks, local_rank, backend, **kwargs)