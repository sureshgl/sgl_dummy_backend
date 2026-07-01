"""
Dummy SRT Platform Plugin

Provides a CPU-compatible dummy platform for testing and development.
Register via setuptools entry_points under sglang.srt.platforms.
"""

import logging

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
    return "dummy_srt_platform_plugin.srt_platform:DummySRTPlatform"
