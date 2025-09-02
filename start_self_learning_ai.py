"""Autonomous training script for the trading AI.

This script loads a large window of recent price data, computes a wide array of
indicators, evaluates hundreds of strategies over multiple timeframes and
performs an evolutionary search to derive weighted meta-strategies.  The best
weights are saved for later use and a DQN agent is trained (or further
trained) on the enriched dataset.

The script is intentionally configurable so heavy experiments can be controlled
from the command line.  It serves as a starting point for more sophisticated
research such as adding genetic operators, extended indicator sets or
distributed search.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import strategies
from backtester import backtest
from strategy_manager import StrategyManager
from indicators import get_all_indicators
from start_ai import load_recent_data, run_multi_timeframe_backtests
from ai_module import TradingEnv, AIModule
from config import RL_ENV_PARAMS
from concurrent.futures import ThreadPoolExecutor

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


def _evaluate_candidate(df, weights: Dict[str, float]) -> tuple[Dict[str, float], float]:
    """Helper used for parallel evaluation of candidate weightings."""

    result = backtest(df, "temp", temp_strategy=_temp_strategy_factory(weights))
    return weights, result["sharpe"]


def evolve_strategy_weights(
    df, generations: int = 5, population: int = 10, executor_workers: int | None = None
) -> Dict[str, float]:
    """Randomly search strategy weightings to maximise Sharpe ratio using threads."""

    strategy_names = list(strategies.strategies.keys())[:300]
    best_weights: Dict[str, float] | None = None
    best_sharpe = float("-inf")

    for g in range(generations):
        candidates = [
            {s: float(np.random.rand()) for s in strategy_names} for _ in range(population)
        ]
        with ThreadPoolExecutor(max_workers=executor_workers) as ex:
            for weights, sharpe in ex.map(lambda w: _evaluate_candidate(df, w), candidates):
                if sharpe > best_sharpe:
                    best_sharpe, best_weights = sharpe, weights
                    logger.info("Gen %d new best Sharpe %.4f", g, sharpe)

    return best_weights if best_weights is not None else {}


def train_continuous_dqn(df, model_path: Path) -> None:
    """Train the DQN agent and persist the model, resuming if possible."""

    env = TradingEnv(df)
    ai = AIModule()
    ai.init_dqn(RL_ENV_PARAMS["state_size"], RL_ENV_PARAMS["action_size"])
    if ai.dqn and torch is not None and model_path.is_file():
        ai.dqn.model.load_state_dict(torch.load(model_path))
        logger.info("Loaded existing DQN model for continued training")
    ai.train_dqn(env, episodes=50, batch_size=64)
    if ai.dqn and ai.dqn.model and torch is not None:
        torch.save(ai.dqn.model.state_dict(), model_path)
        logger.info("Saved trained DQN model to %s", model_path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self learning AI trainer")
    parser.add_argument("source", help="CSV path or MT5 symbol")
    parser.add_argument("--years", type=int, default=5, help="Years of data to load")
    parser.add_argument(
        "--generations", type=int, default=5, help="Generations for weight evolution"
    )
    parser.add_argument(
        "--population", type=int, default=10, help="Population size per generation"
    )
    parser.add_argument(
        "--model-path", default="trained_dqn.pth", help="Where to store the DQN model"
    )
    parser.add_argument(
        "--weights-path", default="best_weights.json", help="Where to store weights"
    )
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    df = load_recent_data(args.source, years=args.years)
    df = get_all_indicators(df)
    run_multi_timeframe_backtests(df, ["M1", "H1", "D1"])
    weights = evolve_strategy_weights(df, args.generations, args.population)
    with open(args.weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    logger.info("Persisted best strategy weights to %s", args.weights_path)
    train_continuous_dqn(df, Path(args.model_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_args())
