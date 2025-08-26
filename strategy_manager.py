import pandas as pd
import logging
import importlib
import strategies  # Import as module

logger = logging.getLogger(__name__)

class StrategyManager:
    def __init__(self, active_strategies: list, weights: dict = None):
        """Initialize with active strategies and optional weights."""
        self._load_strategies()  # Load strategies dynamically
        self.active_strategies = list(set(active_strategies))  # Remove duplicates
        if not all(strat in strategies.strategies for strat in self.active_strategies):
            raise ValueError(f"Invalid strategies: {[s for s in active_strategies if s not in strategies.strategies]}")
        self.weights = weights if weights else {strat: 1.0 for strat in self.active_strategies}
        if set(self.weights.keys()) != set(self.active_strategies):
            raise ValueError("Weights must match active strategies")

    def _load_strategies(self):
        """Dynamically reload strategies module."""
        importlib.reload(strategies)  # Reload the module

    def apply_strategies(self, df: pd.DataFrame, multi_data=None, sym=None, news_bias=0, threshold=0.5) -> pd.DataFrame:
        """Apply active strategies and aggregate signals."""
        self._load_strategies()  # Reload before applying
        signals = pd.DataFrame(index=df.index)
        for strat in self.active_strategies:
            try:
                signals[strat] = strategies.generate_signals(df, strat, multi_data=multi_data, sym=sym, news_bias=news_bias)
            except Exception as e:
                logger.error(f"Error in strategy {strat}: {e}")
                signals[strat] = 0  # Default to hold on error
        weighted_signals = pd.DataFrame(index=df.index)
        for strat in self.active_strategies:
            weighted_signals[strat] = signals[strat] * self.weights[strat]
        signals['aggregate'] = weighted_signals.mean(axis=1).apply(
            lambda x: 1 if x > threshold else -1 if x < -threshold else 0
        )
        return signals