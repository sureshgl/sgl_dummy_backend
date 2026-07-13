"""
Single source of truth for emulated GPU specs.

Previously this was split across two places:
  - device.py's _GPU_VRAM_BYTES (memory capacity), resolved via DUMMY_GPU,
    keyed by descriptive names ("H200-141GB").
  - cost_model.py's _GPU_SPECS (peak FLOPS/bandwidth), resolved via a
    SEPARATE env var, SGLANG_DUMMY_GPU, keyed by short names ("h200").

That split was a real correctness gap, not just duplication: setting
DUMMY_GPU alone (the documented, original Stage 1/2 variable) silently left
the CostModel on whatever SGLANG_DUMMY_GPU's default GPU was -- memory
emulation and latency emulation could disagree about which GPU was even
being emulated.

Consolidated here: one GPUSpec dataclass carrying both memory and compute
figures, one table, one env var (DUMMY_GPU -- kept over SGLANG_DUMMY_GPU
since it's the original, already-documented name), resolved once via the
same module-level holder-dict pattern used throughout this plugin.

BREAKING CHANGE vs Stage 1/2: DUMMY_GPU now takes short keys ("h200",
"a100-80", ...) instead of descriptive ones ("H200-141GB", "A100-80GB",
...). See _GPU_SPECS below for the full mapping.

device.py and cost_model.py both import current_gpu_spec() from here;
neither imports the other.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUSpec:
    """Static published specs for one GPU SKU.

    vram_bytes: total emulated VRAM, in bytes. Consumed by device.py's
        get_device_total_memory().
    peak_flops: dense (bf16/fp16) peak FLOPS/sec -- theoretical tensor-core
        peak, not achievable sustained throughput. Consumed by
        cost_model.py's CostModel.
    peak_bandwidth: peak HBM/GDDR bandwidth in bytes/sec. Consumed by
        cost_model.py's CostModel.
    """

    name: str
    vram_bytes: int
    peak_flops: float
    peak_bandwidth: float


# Published specs per GPU (dense bf16/fp16 tensor-core throughput, peak
# memory bandwidth). VRAM figures preserved exactly from Stage 1/2's
# device.py table. a100-80 / h100 / h200 FLOPS+bandwidth figures were
# already established and used earlier in this project; t4 / a10 / l4 /
# l40 / a100-40 / b200 are new here -- reasonable published figures, but
# worth double-checking against current datasheets rather than treating
# them as independently re-verified.
_GPU_SPECS: dict[str, GPUSpec] = {
    "t4": GPUSpec(name="T4-16GB", vram_bytes=16 * 1024**3, peak_flops=65e12, peak_bandwidth=0.320e12),
    "a10": GPUSpec(name="A10-24GB", vram_bytes=24 * 1024**3, peak_flops=125e12, peak_bandwidth=0.600e12),
    "l4": GPUSpec(name="L4-24GB", vram_bytes=24 * 1024**3, peak_flops=121e12, peak_bandwidth=0.300e12),
    "l40": GPUSpec(name="L40-48GB", vram_bytes=48 * 1024**3, peak_flops=181e12, peak_bandwidth=0.864e12),
    # A100 has two VRAM variants with identical compute spec -- kept as two
    # keys rather than collapsing to one "a100", to avoid silently losing
    # the 40GB variant that device.py's original table distinguished.
    "a100-40": GPUSpec(name="A100-40GB", vram_bytes=40 * 1024**3, peak_flops=312e12, peak_bandwidth=2.039e12),
    "a100-80": GPUSpec(name="A100-80GB", vram_bytes=80 * 1024**3, peak_flops=312e12, peak_bandwidth=2.039e12),
    "h100": GPUSpec(name="H100-80GB", vram_bytes=80 * 1024**3, peak_flops=989e12, peak_bandwidth=3.35e12),
    "h200": GPUSpec(name="H200-141GB", vram_bytes=138 * 1024**3, peak_flops=989e12, peak_bandwidth=4.8e12),
    "b200": GPUSpec(name="B200-192GB", vram_bytes=192 * 1024**3, peak_flops=2250e12, peak_bandwidth=8.0e12),
}

_DEFAULT_GPU = "a100-80"

# Module-level holder-dict singleton -- not a bare module global or an
# instance attribute, consistent with every other shared-state pattern in
# this plugin (device.py's _resident_weight_bytes_holder, the original
# per-CostModel-instance caching, etc.), to avoid the per-process singleton
# mutation bug already hit twice in this project.
_holder: dict = {}
_holder_lock = threading.Lock()


def current_gpu_spec(gpu_name: Optional[str] = None) -> GPUSpec:
    """
    Resolve and cache the active GPUSpec.

    Resolution order: explicit `gpu_name` argument, then the DUMMY_GPU
    environment variable, then _DEFAULT_GPU. Only the first call actually
    resolves anything; every subsequent call returns the same cached
    GPUSpec regardless of arguments -- "resolve once" is deliberate here,
    matching how the rest of this plugin treats process-lifetime config.

    Raises immediately on an unrecognized name -- matching device.py's
    original strict behavior for get_device_total_memory(). A wrong "GPU"
    should never be silently mistaken for a real spec, for either the
    memory or the compute numbers.
    """
    with _holder_lock:
        if "spec" not in _holder:
            name = (gpu_name or os.environ.get("DUMMY_GPU", _DEFAULT_GPU)).lower()
            try:
                _holder["spec"] = _GPU_SPECS[name]
            except KeyError:
                raise ValueError(
                    f"Unknown DUMMY_GPU={name!r}. Supported names: {sorted(_GPU_SPECS)}"
                )
            logger.info("Resolved dummy GPU spec: %s", _holder["spec"].name)
        return _holder["spec"]


def reset_gpu_spec() -> None:
    """Clear the cached GPUSpec. Test helper only -- not used in normal
    operation, since production code should resolve the spec exactly once
    per process."""
    with _holder_lock:
        _holder.pop("spec", None)