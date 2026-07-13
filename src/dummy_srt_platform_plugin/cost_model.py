"""
Coarse roofline cost model for the dummy GPU-emulation platform.

Stage 3 needs a way to turn "this FX-graph piece (or attention call) does X
FLOPs and moves Y bytes" into a plausible latency number, without actually
running any real compute.

GPU spec resolution (peak FLOPS, peak bandwidth) now lives in gpu_spec.py --
a single source of truth shared with device.py's memory-capacity emulation,
resolved once via the DUMMY_GPU environment variable. This module used to
carry its own separate GPUSpec table and its own env var
(SGLANG_DUMMY_GPU); that's been removed in favor of importing
gpu_spec.current_gpu_spec() directly, so memory emulation and
latency emulation can never disagree about which GPU is being emulated.

Explicitly out of scope here (per the Stage 3 agreement): TP
collective-communication cost (all-reduce overhead between ranks). This
model only estimates each rank's local compute/memory cost.
"""

import logging
from typing import Optional

from dummy_srt_platform_plugin.gpu_spec import GPUSpec, current_gpu_spec

logger = logging.getLogger(__name__)


class CostModel:
    """
    Coarse roofline latency estimator.

    latency = max(flops / peak_flops, bytes_moved / peak_bandwidth)

    Resolves current_gpu_spec() once at construction time (unless a spec is
    passed explicitly) and holds onto it for the lifetime of the instance --
    consistent with DummyPiecewiseBackend resolving its CostModel once per
    instance and caching it alongside the latency estimate.
    """

    def __init__(self, gpu_spec: Optional[GPUSpec] = None):
        self.gpu_spec = gpu_spec if gpu_spec is not None else current_gpu_spec()

    def estimate(self, flops: float, bytes_moved: float) -> float:
        """Return an estimated latency in seconds for the given FLOPs and
        bytes-moved. Returns 0.0 for a zero-cost (e.g. no matmul-family ops
        found) piece rather than raising."""
        if flops <= 0 and bytes_moved <= 0:
            return 0.0
        compute_time = flops / self.gpu_spec.peak_flops
        memory_time = bytes_moved / self.gpu_spec.peak_bandwidth
        return max(compute_time, memory_time)