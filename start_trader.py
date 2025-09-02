"""Run a trading simulation using a previously trained DQN model.

The script loads recent market data, computes indicators, restores a saved
DQN model and executes actions within the ``TradingEnv`` to simulate trading.

Optionally the DQN can continue training on the same dataset after the
simulation, allowing online adaptation.
=======

Usage:
    # From CSV data
    python start_trader.py data/historical.csv trained_dqn.pth

    # Or fetch data from MetaTrader 5 using a symbol
    python start_trader.py EURUSD trained_dqn.pth

"""

from __future__ import annotations


import argparse
import json
import logging
=======
import logging
import sys

from pathlib import Path

from start_ai import load_recent_data
from indicators import get_all_indicators

from ai_module import TradingEnv, TradingDQN, AIModule
from config import RL_ENV_PARAMS
from strategy_manager import StrategyManager
=======
from ai_module import TradingEnv, TradingDQN
from config import RL_ENV_PARAMS


try:
    import torch
except Exception:  # pragma: no cover
    torch = None

logger = logging.getLogger(__name__)


def load_model(path: Path) -> TradingDQN:
    """Restore a DQN model from disk if available."""\

=======\
    agent = TradingDQN(RL_ENV_PARAMS["state_size"], RL_ENV_PARAMS["action_size"])
    if agent.model and torch is not None and path.is_file():
        agent.model.load_state_dict(torch.load(path))
        agent.epsilon = 0  # Use greedy policy for execution
        logger.info("Loaded model from %s", path)
    else:
        logger.warning("Using untrained DQN agent; actions will be random")
    return agent


def load_strategy_signals(df, path: Path) -> list | None:
    """Load optimised strategy weights if available and return signals."""

    if not path.is_file():
        return None
    with path.open() as f:
        weights = json.load(f)
    sm = StrategyManager(list(weights.keys()), weights)
    return sm.apply_strategies(df)["aggregate"].values


def continue_training(df, model_path: Path, episodes: int = 10) -> None:
    """Optionally continue DQN training on the provided data."""

    env = TradingEnv(df)
    ai = AIModule()
    ai.init_dqn(RL_ENV_PARAMS["state_size"], RL_ENV_PARAMS["action_size"])
    if ai.dqn and torch is not None and model_path.is_file():
        ai.dqn.model.load_state_dict(torch.load(model_path))
    ai.train_dqn(env, episodes=episodes, batch_size=32)
    if ai.dqn and ai.dqn.model and torch is not None:
        torch.save(ai.dqn.model.state_dict(), model_path)
        logger.info("Saved updated model to %s", model_path)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trading simulation")
    parser.add_argument("source", help="CSV path or MT5 symbol")
    parser.add_argument("model_path", help="Path to trained DQN model")
    parser.add_argument(
        "--weights-path", default="best_weights.json", help="Path to strategy weights"
    )
    parser.add_argument(
        "--continue-training", action="store_true", help="Continue training after simulation"
    )
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    df = load_recent_data(args.source, years=1)
    df = get_all_indicators(df)
    signals = load_strategy_signals(df, Path(args.weights_path))
    env = TradingEnv(df)
    agent = load_model(Path(args.model_path))
    state, _ = env.reset()
    total_reward = 0.0
    for i in range(len(df) - 1):
        if signals is not None and signals[i] != 0:
            action = 0 if signals[i] > 0 else 2
        else:
            action = agent.act(state)
=======
def main(source: str, model_path: str) -> None:
    df = load_recent_data(source, years=1)
    df = get_all_indicators(df)
    env = TradingEnv(df)
    agent = load_model(Path(model_path))
    state, _ = env.reset()
    total_reward = 0.0
    for _ in range(len(df) - 1):
        action = agent.act(state)

        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Total simulated reward: {total_reward:.2f}")
    if args.continue_training:
        continue_training(df, Path(args.model_path))
=======



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_args())
=======
    if len(sys.argv) < 3:
        print("Usage: python start_trader.py <csv path or symbol> trained_dqn.pth")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

