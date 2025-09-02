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
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import strategies
from indicators import get_all_indicators
from backtester import backtest
from strategy_manager import StrategyManager
from ai_module import TradingEnv, AIModule
from config import RL_ENV_PARAMS

try:  # Optional torch dependency for saving trained models
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None

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
    StrategyManager(strategy_names)  # Ensure strategies are validated

    def _run(strat: str, tf: str) -> tuple[str, str, float]:
        result = backtest(df.copy(), strat, timeframe=tf)
        return strat, tf, result["sharpe"]

    for tf in timeframes:
        logger.info("Running %s backtests with %d strategies", tf, len(strategy_names))
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_run, strat, tf): strat for strat in strategy_names}
            for future in as_completed(futures):
                strat, timeframe, sharpe = future.result()
                logger.info("%s @ %s Sharpe: %.4f", strat, timeframe, sharpe)


def train_rl_agent(df: pd.DataFrame) -> None:
    """Train the DQN agent on the provided data and persist the model if possible."""
    env = TradingEnv(df)
    ai = AIModule()
    ai.init_dqn(RL_ENV_PARAMS["state_size"], RL_ENV_PARAMS["action_size"])
    ai.train_dqn(env, episodes=10, batch_size=32)
    if ai.dqn and ai.dqn.model and torch is not None:
        torch.save(ai.dqn.model.state_dict(), "trained_dqn.pth")
        logger.info("Saved trained DQN model to trained_dqn.pth")


def main(csv_path: str) -> None:
    df = load_recent_data(csv_path)
    df = get_all_indicators(df)
    run_multi_timeframe_backtests(df, ["M1", "H1", "D1"])
    env = TradingEnv(df)
    logger.info("RL environment with %d steps ready", len(df))
    train_rl_agent(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python start_ai.py path/to/historical.csv")
        sys.exit(1)
    main(sys.argv[1])
