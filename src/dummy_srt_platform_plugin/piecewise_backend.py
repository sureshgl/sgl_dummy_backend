"""
Dummy piecewise compilation backend for CPU.

Provides a minimal piecewise backend that runs eagerly without CUDA graph capture.
"""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class DummyPiecewiseBackend:
    """
    CPU-compatible piecewise compilation backend.

    Runs the compiled graph eagerly without CUDA graph capture (since CPU doesn't
    support CUDA graphs). This is used by torch.compile for prefill operations.
    """

    def __init__(
        self,
        graph: Any,
        compile_config: Any,
        inductor_config: dict,
        graph_pool: Any,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list,
        compiled_graph_for_general_shape: Callable,
        sglang_backend,
    ):
        """
        Initialize the dummy piecewise backend.

        Args:
            graph: The PyTorch FX graph module.
            compile_config: Compilation configuration.
            inductor_config: Inductor-specific configuration.
            graph_pool: Memory pool for graph operations (unused on CPU).
            piecewise_compile_index: Index of this graph in the piecewise sequence.
            total_piecewise_compiles: Total number of piecewise compiles.
            sym_shape_indices: Indices of symbolic shapes in the input.
            compiled_graph_for_general_shape: Callable for the general-shape graph.
            sglang_backend: Backend instance from SGLang.
        """
        self.graph = graph
        self.compile_config = compile_config
        self.inductor_config = inductor_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.sym_shape_indices = sym_shape_indices
        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape
        self.sglang_backend = sglang_backend

        logger.debug(
            "DummyPiecewiseBackend initialized for graph %d/%d",
            piecewise_compile_index + 1,
            total_piecewise_compiles,
        )

    def __call__(self, *args) -> Any:
        """
        Execute the compiled graph.

        On CPU, we always use the general-shape compiled graph without
        shape-specific optimization or CUDA graph capture.

        Args:
            *args: Input arguments to the compiled graph.

        Returns:
            Output tensor(s) from the graph execution.
        """
        return self.compiled_graph_for_general_shape(*args)
