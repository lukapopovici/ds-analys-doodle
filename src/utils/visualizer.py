"""Thin wrapper that re-exports the refactored Visualizer implementation.

The heavy implementation now lives in `utils.visualization`.
"""
from .visualization import Visualizer

__all__ = ["Visualizer"]