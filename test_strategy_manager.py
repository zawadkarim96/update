import pandas as pd
import numpy as np
from mt5_trading_bot.strategy_manager import StrategyManager
from mt5_trading_bot.strategies import generate_signals
import pytest

@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2025-01-01", periods=1000, freq='T')
    return pd.DataFrame({
        'Open': np.random.rand(1000),
        'High': np.random.rand(1000) * 1.1,
        'Low': np.random.rand(1000) * 0.9,
        'Close': np.random.rand(1000),
        'Volume': np.random.randint(100, 1000, 1000),
        'sma_50': np.random.rand(1000),  # Mock indicator
        'sma_200': np.random.rand(1000)
    }, index=dates)

def test_apply_strategies(sample_df):
    strategies = ['moving_average_crossover']
    weights = {'moving_average_crossover': 1.0}
    sm = StrategyManager(strategies, weights)
    signals = sm.apply_strategies(sample_df)
    assert 'moving_average_crossover' in signals.columns
    assert 'aggregate' in signals.columns
    assert signals['aggregate'].isin([1, -1, 0]).all()

def test_invalid_strategy(sample_df):
    with pytest.raises(ValueError):
        sm = StrategyManager(['invalid_strategy'])
        sm.apply_strategies(sample_df)

def test_weight_mismatch(sample_df):
    with pytest.raises(ValueError):
        sm = StrategyManager(['moving_average_crossover'], {'other': 1.0})
        sm.apply_strategies(sample_df)