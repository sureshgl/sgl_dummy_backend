"""
General SGLang plugin: make the "cpu" dummy device report GPU-shaped
available memory, for whichever GPU DUMMY_GPU (see device.py) names.

sglang.srt.utils.common.get_available_gpu_memory() is what SGLang core
calls everywhere it needs a live "how much memory is free right now"
number -- weight-loading logs, KV-cache sizing (_profile_available_bytes,
which is what max_total_num_tokens and chunked_prefill_size's GPU-memory
tiering ultimately trace back to), CUDA-graph-capture logs, and the final
scheduler startup summary line. For device == "cpu" it calls
psutil.virtual_memory() directly -- it never consults current_platform at
all. So without this hook, none of device.py's GPU-table work would
actually be seen by any of these call sites; the table would just sit there
unread.

This hook makes device == "cpu" report

    current_platform.get_device_total_memory() - current_platform.get_current_memory_usage()

instead -- the emulated GPU's total VRAM minus whatever fake_load.py
determined the (fake) model's real weight footprint would be. Every other
device string (a real "cuda"/"rocm"/etc. deployment) falls straight through
to the real function, untouched.

For distributed=True (used by _profile_available_bytes under TP > 1, to
take the minimum available memory across ranks so KV-cache sizing is safe
even if ranks' shards differ slightly), this hook performs the identical
all_reduce(MIN) over the real gloo cpu_group the real function would use --
same mechanism, just fed emulated per-rank numbers instead of real
torch.cuda.mem_get_info() ones.

pyproject.toml addition:

    [project.entry-points."sglang.srt.plugins"]
    dummy_memory = "dummy_srt_platform_plugin.fake_memory:register"

Depends on device.py's get_device_total_memory/get_current_memory_usage
already being table-driven, and (for a non-zero "used" figure) on
fake_load.py's weight-byte stash having already run.
"""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def register():
    """Entry point called by load_plugins()."""
    HookRegistry.register(
        "sglang.srt.utils.common.get_available_gpu_memory",
        _fake_available_memory,
        HookType.AROUND,
    )
    logger.info("dummy_memory plugin registered (GPU-emulated memory queries)")


def _fake_available_memory(
    original_fn, device, gpu_id, distributed=False, empty_cache=True, cpu_group=None
):
    """AROUND hook: only device == "cpu" (our dummy platform) is emulated;
    every other device string is left completely untouched."""

    if device != "cpu":
        return original_fn(
            device,
            gpu_id,
            distributed=distributed,
            empty_cache=empty_cache,
            cpu_group=cpu_group,
        )

    from sglang.srt.platforms import current_platform

    total = current_platform.get_device_total_memory()
    used = current_platform.get_current_memory_usage()
    available_gb = (total - used) / (1 << 30)  # bytes -> GB, matching original_fn's units

    if distributed and cpu_group is not None:
        import torch

        tensor = torch.tensor(available_gb, dtype=torch.float32)
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
        available_gb = tensor.item()

    return available_gb