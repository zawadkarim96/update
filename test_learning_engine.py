import pytest
from learning_engine import LearningEngine
import pandas as pd

def test_self_reflect(mocker):
    df = pd.DataFrame({'Close': range(2000)})
    mocker.patch('backtester.backtest', return_value={'equity_curve': pd.Series(range(2000)), 'sharpe': 1.0})
    le = LearningEngine()
    le.self_reflect(df, threshold_win_rate=0.5)  # Trigger optimization