from abc import ABC, abstractmethod
from enum import Enum
import multiprocessing as mp
import pandas as pd


class DownsampleMethod(Enum):
    """Enumeration of downsampling strategies"""
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    NONE = "none"


class RenderingStrategy(ABC):
    """
    Abstract base class for rendering strategies.
    Each strategy implements different optimization approaches.
    """

    @abstractmethod
    def should_optimize(self, df, **kwargs):
        """Determine if this strategy should be applied"""
        pass

    @abstractmethod
    def prepare_data(self, df, **kwargs):
        """Prepare data according to strategy"""
        pass

    @abstractmethod
    def get_title_suffix(self, original_len, processed_len):
        """Generate informative title suffix"""
        pass


class SequentialStrategy(RenderingStrategy):
    """No optimization - render all data sequentially"""

    def should_optimize(self, df, **kwargs):
        return False

    def prepare_data(self, df, **kwargs):
        return df, {}

    def get_title_suffix(self, original_len, processed_len):
        return ""


class DownsampleStrategy(RenderingStrategy):
    """Downsample data for faster rendering"""

    def __init__(self, max_points=50000, method=DownsampleMethod.RANDOM, threshold=10000):
        self.max_points = max_points
        self.method = method
        self.threshold = threshold

    def should_optimize(self, df, **kwargs):
        return len(df) > self.threshold

    def prepare_data(self, df, **kwargs):
        if len(df) <= self.max_points:
            return df, {'downsampled': False}

        if self.method == DownsampleMethod.RANDOM:
            result = df.sample(n=self.max_points, random_state=42)
        elif self.method == DownsampleMethod.SYSTEMATIC:
            step = len(df) // self.max_points
            result = df.iloc[::step]
        else:
            result = df

        return result, {'downsampled': True, 'original_len': len(df)}

    def get_title_suffix(self, original_len, processed_len):
        if original_len > processed_len:
            return f" (showing {processed_len:,} of {original_len:,} points)"
        return ""


class ParallelStrategy(RenderingStrategy):
    """Use parallel processing for computations"""

    def __init__(self, max_workers=None, threshold=10000):
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        self.threshold = threshold

    def should_optimize(self, df, **kwargs):
        return len(df) >= self.threshold or kwargs.get('force_parallel', False)

    def prepare_data(self, df, **kwargs):
        # Parallel strategy doesn't transform data, just signals to use parallel execution
        return df, {'use_parallel': True, 'max_workers': self.max_workers}

    def get_title_suffix(self, original_len, processed_len):
        return ""
