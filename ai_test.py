"""Lightweight integration test for the AI modules.

The original script expected a real MetaTrader5 connection.  For unit
testing we instead exercise the pipeline using :class:`MockBroker` which
generates synthetic data.  This keeps the test fast and deterministic
while still providing coverage across the main components.
"""

import logging
import datetime
import pandas as pd

from mt5_trading_bot.broker_interface import MockBroker
from mt5_trading_bot.indicators import get_all_indicators
from mt5_trading_bot.learning_engine import LearningEngine
from mt5_trading_bot.config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ai_integration():
    broker = MockBroker()
    df = broker.get_historical_data(DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, 500)
    df = get_all_indicators(df)
    le = LearningEngine()
    le.optimize_strategies(df)  # Simulate nightly run
    broker.close()


if __name__ == "__main__":  # pragma: no cover - manual execution
    # Simulate nightly run at 02:51 PM +06
    if datetime.datetime.now().hour == 14 and datetime.datetime.now().minute == 51:
        test_ai_integration()
    else:
        logger.info("Running AI test outside scheduled time for demo.")
        test_ai_integration()