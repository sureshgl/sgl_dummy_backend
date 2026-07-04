"""
Dummy SRT Platform Plugin

Provides a CPU-compatible dummy platform for testing and development.
Register via setuptools entry_points under sglang.srt.platforms.
"""

import logging
import sys

logger = logging.getLogger(__name__)


def activate():
    """
    Activation function for the dummy platform plugin.

    Called by the plugin discovery system to determine if this plugin should be
    activated on this machine. Always returns the plugin class name since the
    dummy platform runs on CPU.

    Returns:
        str: Fully-qualified class name for the dummy platform, or None if
            the hardware is not available (never happens for dummy CPU platform).
    """
    logger.info("Activating dummy SRT platform plugin")
    # sgl_kernel ships CUDA-linked binaries only; on a CPU dummy platform
    # every consumer already treats it as optional, so short-circuit the
    # import instead of letting it re-probe GPU architecture every time.
    sys.modules.setdefault("sgl_kernel", None)
    return "dummy_srt_platform_plugin.srt_platform:DummySRTPlatform"
