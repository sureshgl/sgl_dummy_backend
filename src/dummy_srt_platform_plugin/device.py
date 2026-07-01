"""
Dummy device mixin for CPU-backed dummy platform.
"""

import logging
import platform as _platform
from typing import Optional

import psutil
import torch

from sglang.srt.platforms.device_mixin import DeviceMixin, DeviceCapability, PlatformEnum

logger = logging.getLogger(__name__)


class DummyDeviceMixin(DeviceMixin):
    """
    CPU-compatible device mixin for the dummy platform.

    Provides device operations on CPU tensors with minimal overhead.
    """

    _enum = PlatformEnum.OOT
    device_name = "dummy"
    device_type = "cpu"

    # ------------------------------------------------------------------
    # Active methods (called by SGLang core)
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """[Active] Get total system memory in bytes."""
        try:
            return psutil.virtual_memory().total
        except Exception as e:
            logger.warning("Failed to get total memory: %s", e)
            return 16 * 1024 * 1024 * 1024  # Fallback: 16 GB

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        """[Active] Get current system memory usage (total - available) in bytes."""
        try:
            vm = psutil.virtual_memory()
            return float(vm.total - vm.available)
        except Exception as e:
            logger.warning("Failed to get current memory usage: %s", e)
            return 0.0

    # ------------------------------------------------------------------
    # Planned methods (reserved interface, not yet called by core)
    # ------------------------------------------------------------------

    def get_device(self, device_id: int = 0) -> torch.device:
        # """[Planned] Return torch.device for the given device id."""
        # # CPU has no per-rank device; always return torch.device("cpu")
        # return torch.device("cpu")
        """
        Return device string for this platform.
        
        NOTE: Returns str (not torch.device) to match the legacy
        utils/common.py:get_device() fallback path, which expects a string.
        The DeviceMixin interface defines torch.device as the planned return
        type, but the core hasn't migrated to that yet.
        """
        return "cpu"

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
        """[Planned] Return (available_bytes, total_bytes)."""
        try:
            vm = psutil.virtual_memory()
            return (vm.available, vm.total)
        except Exception as e:
            logger.warning("Failed to get available memory: %s", e)
            total = 16 * 1024 * 1024 * 1024
            return (total // 2, total)

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
