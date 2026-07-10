"""
Dummy device mixin for CPU-backed dummy platform.
"""

import logging
import os
import platform as _platform
from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import DeviceMixin, DeviceCapability, PlatformEnum

logger = logging.getLogger(__name__)

# GPU name -> total VRAM, in bytes. Selected via the DUMMY_GPU environment
# variable (plugins can't register new SGLang CLI flags -- see fake_memory.py
# for why current_platform is the seam every consumer of this table goes
# through instead of reading DUMMY_GPU directly).
_GPU_VRAM_BYTES: dict[str, int] = {
    "T4-16GB": 16 * 1024**3,
    "A10-24GB": 24 * 1024**3,
    "L4-24GB": 24 * 1024**3,
    "L40-48GB": 48 * 1024**3,
    "A100-40GB": 40 * 1024**3,
    "A100-80GB": 80 * 1024**3,
    "H100-80GB": 80 * 1024**3,
    "H200-141GB": 138 * 1024**3,
    "B200-192GB": 192 * 1024**3,
}
_DEFAULT_GPU_NAME = "A100-80GB"


 # Module-level, not on any instance -- immune to whatever is re-resolving
# current_platform during startup. Both fake_load.py's writer and
# get_current_memory_usage()'s reader share this dict by importing the
# same module-level name, not by going through the platform singleton.
_resident_weight_bytes_holder = {"value": 0.0}


class DummyDeviceMixin(DeviceMixin):
    """
    CPU-compatible device mixin for the dummy platform.

    Provides device operations on CPU tensors with minimal overhead.
    """

    _enum = PlatformEnum.OOT
    device_name = "dummy"
    device_type = "cpu"

   

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # How much of the emulated VRAM the (fake) model weights are using,
        # in bytes. Set here in __init__ -- not as a class-level default --
        # so it is unambiguously object data: each instance owns its own
        # value in its own __dict__ from the moment it's constructed, rather
        # than instances silently sharing one mutable value declared on the
        # class until something happens to assign an instance-level
        # override. Stays 0.0 until dummy_srt_platform_plugin.fake_load
        # builds a model and stashes the real per-rank weight footprint
        # here directly on current_platform (this same instance) -- see
        # fake_load.py.
        print(f"dummy_device_mixin DIAGNOSTIC: __init__ called pid=",os.getpid())

    # ------------------------------------------------------------------
    # Active methods (called by SGLang core)
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """[Active] Get the emulated GPU's total VRAM, in bytes.

        Looks up DUMMY_GPU (falls back to _DEFAULT_GPU_NAME if unset) in the
        _GPU_VRAM_BYTES table above. This is the actual seam SGLang core
        consults for GPU-memory-tiered decisions on an OOT platform --
        sglang.srt.utils.common.get_device_memory_capacity() calls this
        method directly whenever current_platform.is_out_of_tree() is True,
        which is what ultimately drives ServerArgs' chunked_prefill_size
        GPU-tier selection. Raises rather than silently substituting a
        default on an unrecognized name -- a wrong "GPU" should never be
        mistaken for a real memory figure.
        """
        gpu_name = os.environ.get("DUMMY_GPU", _DEFAULT_GPU_NAME)
        try:
            return _GPU_VRAM_BYTES[gpu_name]
        except KeyError:
            raise ValueError(
                f"Unknown DUMMY_GPU={gpu_name!r}. Supported names: "
                f"{sorted(_GPU_VRAM_BYTES)}"
            )

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        """[Active] Get how much of the emulated VRAM the (fake) model
        weights are using, in bytes. Defaults to 0.0 before any model has
        been "loaded" -- see fake_load.py's _fake_load_model, which computes
        the meta-device model's real weight footprint (sum of
        numel * element_size across every parameter, i.e. exactly what real
        weights would cost in the real target dtype) and stashes it directly
        on this instance once construction finishes.
        """
        return float(_resident_weight_bytes_holder["value"])

    # ------------------------------------------------------------------
    # Planned methods (reserved interface, not yet called by core)
    # ------------------------------------------------------------------

    def get_device(self, device_id: int = 0) -> torch.device:
        """[Planned] Return torch.device for the given device id."""
        # CPU has no per-rank device; always return torch.device("cpu")
        return torch.device("cpu")

    def set_device(self, device: torch.device) -> None:
        """[Planned] Set the current device."""
        # CPU is always the default; this is a no-op for symmetry with CudaDeviceMixin.
        pass

    def get_device_name(self, device_id: int = 0) -> str:
        """[Planned] Get human-readable device name."""
        machine = _platform.machine()
        return f"Dummy CPU ({machine})"

    def get_device_uuid(self, device_id: int = 0) -> str:
        """[Planned] Get unique device identifier."""
        return _platform.machine()

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        """[Planned] Get device compute capability. None for CPU."""
        return None

    def empty_cache(self) -> None:
        """[Planned] Release cached device memory."""
        import gc
        gc.collect()

    def synchronize(self) -> None:
        """[Planned] Synchronize device operations."""
        # CPU is always synchronous; no-op.
        pass

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        """[Planned] Return (available_bytes, total_bytes) for the emulated
        GPU -- kept consistent with get_device_total_memory /
        get_current_memory_usage above rather than querying real host RAM,
        so this doesn't silently disagree with them if core ever calls it."""
        total = self.get_device_total_memory(device_id)
        used = self.get_current_memory_usage()
        return (max(int(total - used), 0), total)

    def get_torch_distributed_backend_str(self) -> str:
        """Return the torch.distributed backend string."""
        return "gloo"

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """[Planned] Set random seeds for reproducibility."""
        import random
        import numpy as np
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def get_dispatch_key_name(self) -> str:
        """Return the dispatch key name for MultiPlatformOp."""
        # Return "cpu" so existing CPU implementations are used
        return "cpu"