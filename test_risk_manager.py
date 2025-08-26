import pandas as pd
import numpy as np
from mt5_trading_bot.risk_manager import RiskManager
import pytest
import MetaTrader5 as mt5

@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2025-01-01", periods=1000, freq='T')
    return pd.DataFrame({
        'High': np.random.rand(1000) * 1.1,
        'Low': np.random.rand(1000) * 0.9,
        'Close': np.random.rand(1000)
    }, index=dates)

def test_calculate_position_size(sample_df, mocker):
    mt5.initialize()
    mocker.patch('MetaTrader5.symbol_info', return_value=mt5.SymbolInfo(symbol="EURUSD"))
    rm = RiskManager(10000)
    size = rm.calculate_position_size(1.1000, 1.0950, "EURUSD")
    assert size > 0.01 and size <= 10.0  # Reasonable range

def test_set_sl_tp(sample_df):
    rm = RiskManager()
    sl, tp = rm.set_sl_tp(sample_df, 1.1000, 1)
    assert sl < 1.1000 and tp > 1.1000
    sl, tp = rm.set_sl_tp(sample_df, 1.1000, -1)
    assert sl > 1.1000 and tp < 1.1000

def test_trailing_stop(sample_df):
    rm = RiskManager()
    trail = rm.trailing_stop(1.1200, 1.1000, 1, df=sample_df)
    assert trail is not None and trail < 1.1200