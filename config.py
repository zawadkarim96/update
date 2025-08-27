import os
from pathlib import Path

# In the real project MT5 credentials are required for live trading.  The
# unit tests run in an isolated environment so we provide harmless defaults
# instead of failing when credentials are missing.
MT5_ACCOUNT = int(os.getenv("MT5_ACCOUNT", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "password")
MT5_SERVER = os.getenv("MT5_SERVER", "ICMarketsSC-Demo")
MT5_PATH = os.getenv("MT5_PATH", "")  # Full path to terminal64.exe if not in PATH

# Validate MT5 path only if explicitly provided
if MT5_PATH and not os.path.exists(MT5_PATH):
    raise FileNotFoundError(f"MT5 path not found: {MT5_PATH}")

# Trading Parameters
INITIAL_CAPITAL = 10000  # For backtesting/demo
RISK_PER_TRADE = 0.01   # 1% risk per trade
SYMBOLS = ["XAUUSD", "QQQ.nas", "BTCUSD", "EURUSD", "GBPJPY"]  # Updated for IC Markets NASDAQ symbol
DEFAULT_TIMEFRAME = "TIMEFRAME_M1"
DEFAULT_SYMBOL = SYMBOLS[0] if SYMBOLS else "EURUSD"
POLLING_INTERVAL = 60   # Seconds between real-time data fetches (M1 = 1 minute)

# AI Settings
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "")
if not LLM_MODEL_PATH:
    repo_dir = Path(__file__).resolve().parent
    candidates = list(repo_dir.glob("*.gguf")) + list(repo_dir.glob("*.bin"))
    if not candidates:
        qwen_dir = repo_dir / "qwen-3b"
        if qwen_dir.is_dir():
            candidates = list(qwen_dir.glob("*.gguf")) + list(qwen_dir.glob("*.bin"))
    if candidates:
        LLM_MODEL_PATH = str(candidates[0])


RL_ENV_PARAMS = {
    "state_size": 10,    # Number of indicators in RL observation
    "action_size": 3     # Buy/hold/sell
}

# Strategy Weights (for StrategyManager)
DEFAULT_STRATEGY_WEIGHTS = {
    "moving_average_crossover": 1.0,
    "rsi_overbought_oversold": 0.8
}

# Optimization Settings
OPTIMIZATION_INTERVAL = "daily"  # Options: 'daily', 'weekly', 'monthly'
OPTIMIZATION_MIN_ROWS = 2000    # Minimum data rows for learning_engine

