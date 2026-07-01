"""
Unit tests for the dummy SRT platform plugin.

Tests plugin discovery, activation, and platform interface implementation.
"""

import os
import sys
import unittest
from unittest import mock

# Ensure the src package directory is in the path when running tests from the plugin repo root.
plugin_src_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)
sys.path.insert(0, plugin_src_dir)


class TestDummyPluginActivation(unittest.TestCase):
    """Test plugin activation and discovery."""

    def test_activate_function_returns_platform_class_name(self):
        """Test that activate() returns the correct platform class path."""
        from dummy_srt_platform_plugin import activate
        result = activate()
        self.assertEqual(result, "dummy_srt_platform_plugin.srt_platform:DummySRTPlatform")

    def test_activate_function_is_callable(self):
        """Test that activate() is callable."""
        from dummy_srt_platform_plugin import activate
        self.assertTrue(callable(activate))


class TestDummySRTPlatform(unittest.TestCase):
    """Test DummySRTPlatform implementation."""

    def setUp(self):
        """Set up test fixtures."""
        from dummy_srt_platform_plugin.srt_platform import DummySRTPlatform
        self.platform_cls = DummySRTPlatform
        self.platform = DummySRTPlatform()

    def test_platform_inherits_from_srt_platform(self):
        """Test that DummySRTPlatform inherits from SRTPlatform."""
        from sglang.srt.platforms.interface import SRTPlatform
        self.assertIsInstance(self.platform, SRTPlatform)

    def test_platform_inherits_from_device_mixin(self):
        """Test that DummySRTPlatform inherits from DummyDeviceMixin."""
        from dummy_srt_platform_plugin.device import DummyDeviceMixin
        self.assertIsInstance(self.platform, DummyDeviceMixin)

    def test_get_default_attention_backend(self):
        """Test get_default_attention_backend() returns torch_native."""
        backend = self.platform.get_default_attention_backend()
        self.assertEqual(backend, "torch_native")

    def test_get_graph_runner_cls(self):
        """Test get_graph_runner_cls() returns CPUGraphRunner."""
        runner_cls = self.platform.get_graph_runner_cls()
        from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
        self.assertEqual(runner_cls, CPUGraphRunner)

    def test_get_mha_kv_pool_cls(self):
        """Test get_mha_kv_pool_cls() returns MHATokenToKVPool."""
        pool_cls = self.platform.get_mha_kv_pool_cls()
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        self.assertEqual(pool_cls, MHATokenToKVPool)

    def test_get_mla_kv_pool_cls(self):
        """Test get_mla_kv_pool_cls() returns MLATokenToKVPool."""
        pool_cls = self.platform.get_mla_kv_pool_cls()
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        self.assertEqual(pool_cls, MLATokenToKVPool)

    def test_get_dsa_kv_pool_cls(self):
        """Test get_dsa_kv_pool_cls() returns DSATokenToKVPool."""
        pool_cls = self.platform.get_dsa_kv_pool_cls()
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        self.assertEqual(pool_cls, DSATokenToKVPool)

    def test_get_paged_allocator_cls(self):
        """Test get_paged_allocator_cls() returns PagedTokenToKVPoolAllocator."""
        allocator_cls = self.platform.get_paged_allocator_cls()
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        self.assertEqual(allocator_cls, PagedTokenToKVPoolAllocator)

    def test_get_piecewise_backend_cls(self):
        """Test get_piecewise_backend_cls() returns DummyPiecewiseBackend."""
        backend_cls = self.platform.get_piecewise_backend_cls()
        from dummy_srt_platform_plugin.piecewise_backend import DummyPiecewiseBackend
        self.assertEqual(backend_cls, DummyPiecewiseBackend)

    def test_support_cuda_graph(self):
        """Test support_cuda_graph() returns False."""
        self.assertFalse(self.platform.support_cuda_graph())

    def test_support_piecewise_cuda_graph(self):
        """Test support_piecewise_cuda_graph() returns False."""
        self.assertFalse(self.platform.support_piecewise_cuda_graph())


class TestDummyDeviceMixin(unittest.TestCase):
    """Test DummyDeviceMixin implementation."""

    def setUp(self):
        """Set up test fixtures."""
        from dummy_srt_platform_plugin.device import DummyDeviceMixin
        self.device_mixin = DummyDeviceMixin()

    def test_device_name_is_dummy(self):
        """Test that device_name is 'dummy'."""
        self.assertEqual(self.device_mixin.device_name, "dummy")

    def test_device_type_is_cpu(self):
        """Test that device_type is 'cpu'."""
        self.assertEqual(self.device_mixin.device_type, "cpu")

    def test_is_out_of_tree(self):
        """Test that is_out_of_tree() returns True."""
        self.assertTrue(self.device_mixin.is_out_of_tree())

    def test_is_cpu(self):
        """Test that is_cpu() returns False (we are OOT, not CPU)."""
        # The dummy platform is OOT, not the built-in CPU platform
        self.assertFalse(self.device_mixin.is_cpu())

    def test_is_cuda(self):
        """Test that is_cuda() returns False."""
        self.assertFalse(self.device_mixin.is_cuda())

    def test_get_device_total_memory(self):
        """Test get_device_total_memory() returns a positive integer."""
        total_memory = self.device_mixin.get_device_total_memory()
        self.assertIsInstance(total_memory, int)
        self.assertGreater(total_memory, 0)

    def test_get_current_memory_usage(self):
        """Test get_current_memory_usage() returns a non-negative float."""
        current_usage = self.device_mixin.get_current_memory_usage()
        self.assertIsInstance(current_usage, float)
        self.assertGreaterEqual(current_usage, 0)

    def test_get_dispatch_key_name(self):
        """Test get_dispatch_key_name() returns 'cpu'."""
        key = self.device_mixin.get_dispatch_key_name()
        self.assertEqual(key, "cpu")

    def test_get_torch_distributed_backend_str(self):
        """Test get_torch_distributed_backend_str() returns 'gloo'."""
        backend = self.device_mixin.get_torch_distributed_backend_str()
        self.assertEqual(backend, "gloo")

    def test_synchronize_does_not_raise(self):
        """Test that synchronize() does not raise an exception."""
        try:
            self.device_mixin.synchronize()
        except Exception as e:
            self.fail(f"synchronize() raised {type(e).__name__}: {e}")

    def test_empty_cache_does_not_raise(self):
        """Test that empty_cache() does not raise an exception."""
        try:
            self.device_mixin.empty_cache()
        except Exception as e:
            self.fail(f"empty_cache() raised {type(e).__name__}: {e}")


class TestDummyPiecewiseBackend(unittest.TestCase):
    """Test DummyPiecewiseBackend implementation."""

    def setUp(self):
        """Set up test fixtures."""
        from dummy_srt_platform_plugin.piecewise_backend import DummyPiecewiseBackend
        import torch

        # Create a simple mock graph and general-shape callable
        self.mock_graph = mock.MagicMock()
        self.mock_output = torch.randn(4, 8)

        def mock_compiled_graph(*args):
            return self.mock_output

        self.backend = DummyPiecewiseBackend(
            graph=self.mock_graph,
            compile_config={},
            inductor_config={},
            graph_pool=None,
            piecewise_compile_index=0,
            total_piecewise_compiles=1,
            sym_shape_indices=[],
            compiled_graph_for_general_shape=mock_compiled_graph,
            sglang_backend=None,
        )

    def test_backend_is_callable(self):
        """Test that DummyPiecewiseBackend is callable."""
        self.assertTrue(callable(self.backend))

    def test_backend_returns_graph_output(self):
        """Test that calling backend returns the compiled graph output."""
        import torch

        # Create mock input
        mock_input = torch.randn(4, 8)
        result = self.backend(mock_input)

        # Verify output matches expected
        self.assertTrue(torch.equal(result, self.mock_output))

    def test_backend_forwards_to_general_shape_graph(self):
        """Test that backend forwards to compiled_graph_for_general_shape."""
        import torch

        # Create a mock that tracks calls
        call_count = [0]

        def tracked_graph(*args):
            call_count[0] += 1
            return self.mock_output

        backend = self.backend.__class__(
            graph=self.mock_graph,
            compile_config={},
            inductor_config={},
            graph_pool=None,
            piecewise_compile_index=0,
            total_piecewise_compiles=1,
            sym_shape_indices=[],
            compiled_graph_for_general_shape=tracked_graph,
            sglang_backend=None,
        )

        mock_input = torch.randn(4, 8)
        backend(mock_input)

        self.assertEqual(call_count[0], 1)


if __name__ == "__main__":
    unittest.main()
