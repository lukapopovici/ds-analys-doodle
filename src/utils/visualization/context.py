from typing import List, Tuple


class StrategyContext:
    """
    Context class that uses strategies to optimize rendering.
    Handles strategy selection and coordination.
    """

    def __init__(self, strategies: List = None):
        """Initialize with a list of strategies (applied in order)"""
        self.strategies = strategies or []

    def add_strategy(self, strategy):
        """Add a new strategy to the context"""
        self.strategies.append(strategy)

    def apply_strategies(self, df, **kwargs) -> Tuple:
        """Apply all applicable strategies and return optimized data"""
        processed_df = df
        metadata = {}
        title_parts = []
        original_len = len(df)

        for strategy in self.strategies:
            if strategy.should_optimize(processed_df, **kwargs):
                processed_df, strategy_meta = strategy.prepare_data(processed_df, **kwargs)
                metadata.update(strategy_meta)

                suffix = strategy.get_title_suffix(original_len, len(processed_df))
                if suffix:
                    title_parts.append(suffix)

        title_suffix = "".join(title_parts)
        return processed_df, metadata, title_suffix
