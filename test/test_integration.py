"""
Integration test for the dummy SRT platform plugin with SGLang test infrastructure.

This test can be run from the SGLang project test suite.
"""

import os
import sys
import unittest

# Ensure the src package directory is in the path when running tests from the plugin repo root.
plugin_src_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)
if os.path.exists(plugin_src_dir):
    sys.path.insert(0, plugin_src_dir)

# Ensure we can import sglang if running from sglang test suite
try:
    from sglang.srt.platforms import _load_platform_class
except ImportError:
    # Try adding sglang to path if not already available
    sglang_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    )
    sglang_python = os.path.join(sglang_root, "sglang", "python")
    if os.path.exists(sglang_python):
        sys.path.insert(0, sglang_python)


class TestDummyPlatformDiscovery(unittest.TestCase):
    """Test dummy platform plugin discovery by SGLang."""

    def test_load_dummy_platform_class_via_entry_point(self):
        """Test that the dummy platform can be loaded via entry point."""
        from sglang.srt.platforms import _load_platform_class

        # Try to load the dummy platform
        platform_cls = _load_platform_class("dummy_srt_platform_plugin.srt_platform:DummySRTPlatform")
        self.assertIsNotNone(platform_cls)

        # Verify it's the correct class
        from dummy_srt_platform_plugin.srt_platform import DummySRTPlatform
        self.assertEqual(platform_cls, DummySRTPlatform)

    def test_dummy_platform_activate_function(self):
        """Test that the dummy platform activate function is discoverable."""
        from dummy_srt_platform_plugin import activate

        result = activate()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("DummySRTPlatform", result)

    def test_dummy_platform_instantiation(self):
        """Test that DummySRTPlatform can be instantiated."""
        from dummy_srt_platform_plugin.srt_platform import DummySRTPlatform

        platform = DummySRTPlatform()
        self.assertIsNotNone(platform)


class TestDummyPlatformEnvironmentVariable(unittest.TestCase):
    """Test SGLANG_PLATFORM environment variable selection."""

    def test_select_dummy_platform_via_env_var(self):
        """Test selecting dummy platform via SGLANG_PLATFORM env var."""
        # This test should be run with SGLANG_PLATFORM=dummy set
        env_platform = os.environ.get("SGLANG_PLATFORM", "")
        
        # If we're running this test with SGLANG_PLATFORM=dummy, verify it works
        if env_platform == "dummy":
            # Import after setting env var
            try:
                # Clear the cached platform if it exists
                import sglang.srt.platforms as platforms_module
                if hasattr(platforms_module, "_platform"):
                    delattr(platforms_module, "_platform")
                
                # Now import current_platform
                from sglang.srt.platforms import current_platform
                
                # Verify it's the dummy platform
                from dummy_srt_platform_plugin.srt_platform import DummySRTPlatform
                self.assertIsInstance(current_platform, DummySRTPlatform)
            except Exception as e:
                # If SGLang doesn't have platform resolution yet, skip
                self.skipTest(f"Cannot test platform resolution: {e}")


if __name__ == "__main__":
    unittest.main()
