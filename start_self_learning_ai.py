"""Autonomous training script for the trading AI.

This script loads up to five years of price data, computes a large set of
indicators, evaluates hundreds of strategies over multiple timeframes and
performs a simple evolutionary search to derive new weighted meta-strategies.
The best strategy weights are saved to ``best_weights.json`` and a DQN agent is
trained (or further trained) on the enriched dataset.

Usage:
    python start_self_learning_ai.py <csv path or symbol>

The script reuses utility functions from ``start_ai.py`` for data loading and
backtesting.  It is intentionally lightweight and serves as a foundation for
more advanced research such as adding genetic operators or extended indicator
sets.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import strategies
from backtester import backtest
from strategy_manager import StrategyManager
from indicators import get_all_indicators
from start_ai import load_recent_data, run_multi_timeframe_backtests
from ai_module import TradingEnv, AIModule
from config import RL_ENV_PARAMS

try:  # Optional torch dependency
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None

logger = logging.getLogger(__name__)


def _temp_strategy_factory(weights: Dict[str, float]):
    """Return a callable that aggregates signals using provided weights."""

    def _temp(df):
        sm = StrategyManager(list(weights.keys()), weights)
        return sm.apply_strategies(df)["aggregate"]

    return _temp


def evolve_strategy_weights(df, generations: int = 5, population: int = 10) -> Dict[str, float]:
    """Randomly search strategy weightings to maximise Sharpe ratio.

    Parameters
    ----------
    df:
        Price data enriched with indicators.
    generations:
        Number of generations to iterate.
    population:
        Candidate solutions per generation.
    """
    strategy_names = list(strategies.strategies.keys())[:300]
    best_weights: Dict[str, float] | None = None
    best_sharpe = float("-inf")

    for g in range(generations):
        for _ in range(population):
            weights = {s: float(np.random.rand()) for s in strategy_names}
            result = backtest(df, "temp", temp_strategy=_temp_strategy_factory(weights))
            sharpe = result["sharpe"]
            if sharpe > best_sharpe:
                best_sharpe, best_weights = sharpe, weights
                logger.info("Gen %d new best Sharpe %.4f", g, sharpe)

    return best_weights if best_weights is not None else {}


def train_continuous_dqn(df) -> None:
    """Train the DQN agent and persist the model, resuming if possible."""
    env = TradingEnv(df)
    ai = AIModule()
    ai.init_dqn(RL_ENV_PARAMS["state_size"], RL_ENV_PARAMS["action_size"])
    model_path = Path("trained_dqn.pth")
    if ai.dqn and torch is not None and model_path.is_file():
        ai.dqn.model.load_state_dict(torch.load(model_path))
        logger.info("Loaded existing DQN model for continued training")
    ai.train_dqn(env, episodes=50, batch_size=64)
    if ai.dqn and ai.dqn.model and torch is not None:
        torch.save(ai.dqn.model.state_dict(), model_path)
        logger.info("Saved trained DQN model to %s", model_path)


def main(source: str) -> None:
    df = load_recent_data(source, years=5)
    df = get_all_indicators(df)
    run_multi_timeframe_backtests(df, ["M1", "H1", "D1"])
    weights = evolve_strategy_weights(df)
    with open("best_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    logger.info("Persisted best strategy weights to best_weights.json")
    train_continuous_dqn(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python start_self_learning_ai.py <csv path or symbol>")
        sys.exit(1)
    main(sys.argv[1])
