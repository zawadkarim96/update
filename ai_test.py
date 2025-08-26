from mt5_trading_bot.broker_interface import MT5Broker
from mt5_trading_bot.indicators import get_all_indicators
from mt5_trading_bot.learning_engine import LearningEngine
from mt5_trading_bot.config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME
import pandas as pd
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ai_integration():
    broker = MT5Broker()
    df = broker.get_historical_data(DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, 5000)
    df = get_all_indicators(df)
    le = LearningEngine()
    le.optimize_strategies(df)  # Simulate nightly run
    broker.close()

if __name__ == "__main__":
    # Simulate nightly run at 02:51 PM +06
    if datetime.datetime.now().hour == 14 and datetime.datetime.now().minute == 51:
        test_ai_integration()
    else:
        logger.info("Running AI test outside scheduled time for demo.")
        test_ai_integration()