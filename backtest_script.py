import pandas as pd
from mt5_trading_bot.broker_interface import MT5Broker
from mt5_trading_bot.indicators import get_all_indicators
from mt5_trading_bot.strategy_manager import StrategyManager
from mt5_trading_bot.backtester import backtest
from mt5_trading_bot.config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_backtest(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, start_date="2024-01-01", end_date="2025-08-25"):
    broker = MT5Broker()
    # Fetch historical data
    df = broker.get_historical_data(symbol, timeframe, count=10000)
    df = df[df.index.date >= pd.to_datetime(start_date).date()]
    df = df[df.index.date <= pd.to_datetime(end_date).date()]
    df = get_all_indicators(df)
    
    # Backtest with multiple strategies
    strategies = ['moving_average_crossover', 'rsi_overbought_oversold', 'macd_crossover']
    sm = StrategyManager(strategies)
    signals = sm.apply_strategies(df)
    result = backtest(df, strategies[0])  # Test first strategy
    
    logger.info(f"Backtest Results - Sharpe: {result['sharpe']:.2f}, "
                f"Final Equity: {result['equity_curve'].iloc[-1]:.2f}, "
                f"Cumulative Returns: {result['cum_returns'].iloc[-1]:.2%}")
    broker.close()
    return result

if __name__ == "__main__":
    run_backtest()