import pandas as pd
import numpy as np
from mt5_trading_bot.indicators import get_all_indicators
import pytest

# Sample data
@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2025-01-01", periods=1000, freq='T')
    return pd.DataFrame({
        'Open': np.random.rand(1000),
        'High': np.random.rand(1000) * 1.1,
        'Low': np.random.rand(1000) * 0.9,
        'Close': np.random.rand(1000),
        'Volume': np.random.randint(100, 1000, 1000)
    }, index=dates)

def test_get_all_indicators(sample_df):
    df_ind = get_all_indicators(sample_df, include_price=True)
    assert all(col in df_ind.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    assert 'TA_RSI_14' in df_ind.columns  # Example TA-Lib indicator
    assert 'PTA_SMA_10' in df_ind.columns  # Example pandas_ta indicator
    assert 'PTA_PSARI_002_02' in df_ind.columns  # PSAR long should be present
    assert not df_ind.isnull().all().any()  # No all-NaN columns
    assert len(df_ind) == len(sample_df)

def test_missing_columns(sample_df):
    df_missing = sample_df.drop('Volume', axis=1)
    with pytest.raises(ValueError):
        get_all_indicators(df_missing)
