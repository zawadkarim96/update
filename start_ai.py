"""Utility script to run multi-timeframe backtests and prepare an RL environment.

The script loads up to five years of historical data from a CSV file, computes a
large set of indicators, applies up to 300 available strategies, and reports the
Sharpe ratio for each strategy across multiple timeframes. It also instantiates
the reinforcement learning environment defined in ``ai_module`` so further
training can be layered on top. The heavy lifting of computing hundreds of
indicators and strategy variations is handled by ``indicators.py`` and
``strategies.py``.

Usage:

"""

from __future__ import annotations

import logging


import pandas as pd

import strategies
from indicators import get_all_indicators
from backtester import backtest
from strategy_manager import StrategyManager


logger = logging.getLogger(__name__)





def run_multi_timeframe_backtests(df: pd.DataFrame, timeframes: List[str]) -> None:
    """Run backtests for each strategy and timeframe, logging Sharpe ratios."""
    strategy_names = list(strategies.strategies.keys())[:300]

    df = get_all_indicators(df)
    run_multi_timeframe_backtests(df, ["M1", "H1", "D1"])
    env = TradingEnv(df)
    logger.info("RL environment with %d steps ready", len(df))



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:

        sys.exit(1)
    main(sys.argv[1])
