# tests/test_strategies.py
import pandas as pd
import numpy as np
from strategies import generate_signals

def test_moving_average_crossover():
    df = pd.DataFrame({
        'Open': np.random.rand(1000),
        'High': np.random.rand(1000) * 1.1,
        'Low': np.random.rand(1000) * 0.9,
        'Close': np.random.rand(1000),
        'Volume': np.random.randint(100, 1000, 1000)
    }, index=pd.date_range('2025-01-01', periods=1000, freq='T'))
    signals = generate_signals(df, 'moving_average_crossover')
    assert signals.isin([1, -1, 0]).all()