"""Utility script to run multi-timeframe backtests and prepare an RL environment.

The script loads up to five years of historical data from a CSV file, computes a
large set of indicators, applies up to 300 available strategies, and reports the
Sharpe ratio for each strategy across multiple timeframes. It also instantiates
the reinforcement learning environment defined in ``ai_module`` so further
training can be layered on top. The heavy lifting of computing hundreds of
indicators and strategy variations is handled by ``indicators.py`` and
``strategies.py``.

Usage:
    python start_ai.py data/historical.csv

The CSV must contain columns: Date, Open, High, Low, Close, Volume.
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from typing import List

import pandas as pd

import strategies
from indicators import get_all_indicators
from backtester import backtest
from strategy_manager import StrategyManager
from ai_module import TradingEnv

logger = logging.getLogger(__name__)


def load_recent_data(path: str, years: int = 5) -> pd.DataFrame:
    """Load CSV data and keep only the most recent ``years`` of records."""
    df = pd.read_csv(path, parse_dates=["Date"])
    end = df["Date"].max()
    start = end - timedelta(days=years * 365)
    return df[df["Date"].between(start, end)].set_index("Date")


def run_multi_timeframe_backtests(df: pd.DataFrame, timeframes: List[str]) -> None:
    """Run backtests for each strategy and timeframe, logging Sharpe ratios."""
    strategy_names = list(strategies.strategies.keys())[:300]
    sm = StrategyManager(strategy_names)
    for tf in timeframes:
        logger.info("Running %s backtests with %d strategies", tf, len(strategy_names))
        for strat in strategy_names:
            result = backtest(df.copy(), strat, timeframe=tf)
            logger.info("%s @ %s Sharpe: %.4f", strat, tf, result["sharpe"])


def main(csv_path: str) -> None:
    df = load_recent_data(csv_path)
    df = get_all_indicators(df)
    run_multi_timeframe_backtests(df, ["M1", "H1", "D1"])
    env = TradingEnv(df)
    logger.info("RL environment with %d steps ready", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python start_ai.py path/to/historical.csv")
        sys.exit(1)
    main(sys.argv[1])
