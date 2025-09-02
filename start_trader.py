"""Run a trading simulation using a previously trained DQN model.

The script loads recent market data, computes indicators, restores a saved
DQN model and executes actions within the ``TradingEnv`` to simulate trading.

Usage:
    # From CSV data
    python start_trader.py data/historical.csv trained_dqn.pth

    # Or fetch data from MetaTrader 5 using a symbol
    python start_trader.py EURUSD trained_dqn.pth
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from start_ai import load_recent_data
from indicators import get_all_indicators
from ai_module import TradingEnv, TradingDQN
from config import RL_ENV_PARAMS
from strategy_manager import StrategyManager

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

logger = logging.getLogger(__name__)


def load_model(path: Path) -> TradingDQN:
    """Restore a DQN model from disk if available."""
    agent = TradingDQN(RL_ENV_PARAMS["state_size"], RL_ENV_PARAMS["action_size"])
    if agent.model and torch is not None and path.is_file():
        agent.model.load_state_dict(torch.load(path))
        agent.epsilon = 0  # Use greedy policy for execution
        logger.info("Loaded model from %s", path)
    else:
        logger.warning("Using untrained DQN agent; actions will be random")
    return agent


def load_strategy_signals(df):
    """Load optimised strategy weights if available and return signals."""
    path = Path("best_weights.json")
    if not path.is_file():
        return None
    with path.open() as f:
        weights = json.load(f)
    sm = StrategyManager(list(weights.keys()), weights)
    return sm.apply_strategies(df)["aggregate"].values


def main(source: str, model_path: str) -> None:
    df = load_recent_data(source, years=1)
    df = get_all_indicators(df)
    signals = load_strategy_signals(df)
    env = TradingEnv(df)
    agent = load_model(Path(model_path))
    state, _ = env.reset()
    total_reward = 0.0
    for i in range(len(df) - 1):
        if signals is not None and signals[i] != 0:
            action = 0 if signals[i] > 0 else 2
        else:
            action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Total simulated reward: {total_reward:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print("Usage: python start_trader.py <csv path or symbol> trained_dqn.pth")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
