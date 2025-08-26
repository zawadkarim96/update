import os
import configparser

# MT5 credentials must be supplied via environment variables or a secure
# configuration file. Set MT5_ACCOUNT and MT5_PASSWORD environment variables, or
# point MT5_CREDENTIALS_FILE to an INI file containing:
# [mt5]
# account=123456
# password=yourpassword
MT5_ACCOUNT = os.getenv("MT5_ACCOUNT")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")

if MT5_ACCOUNT is None or MT5_PASSWORD is None:
    creds_file = os.getenv("MT5_CREDENTIALS_FILE")
    if creds_file and os.path.exists(creds_file):
        parser = configparser.ConfigParser()
        parser.read(creds_file)
        if MT5_ACCOUNT is None:
            MT5_ACCOUNT = parser.get("mt5", "account", fallback=None)
        if MT5_PASSWORD is None:
            MT5_PASSWORD = parser.get("mt5", "password", fallback=None)

if MT5_ACCOUNT is None or MT5_PASSWORD is None:
    raise EnvironmentError(
        "MT5_ACCOUNT and MT5_PASSWORD must be provided via environment variables "
        "or a secure config file referenced by MT5_CREDENTIALS_FILE."
    )

MT5_ACCOUNT = int(MT5_ACCOUNT)
MT5_SERVER = os.getenv("MT5_SERVER", "ICMarketsSC-Demo")
MT5_PATH = os.getenv("MT5_PATH", "")  # Full path to terminal64.exe if not in PATH

# Validate MT5 path
if MT5_PATH and not os.path.exists(MT5_PATH):
    raise FileNotFoundError(f"MT5 path not found: {MT5_PATH}")

# Trading Parameters
INITIAL_CAPITAL = 10000  # For backtesting/demo
RISK_PER_TRADE = 0.01   # 1% risk per trade
SYMBOLS = ["XAUUSD", "QQQ.nas", "BTCUSD", "EURUSD", "GBPJPY"]  # Updated for IC Markets NASDAQ symbol
DEFAULT_TIMEFRAME = "TIMEFRAME_M1"
POLLING_INTERVAL = 60   # Seconds between real-time data fetches (M1 = 1 minute)

# AI Settings
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/qwen-3b.gguf")
if LLM_MODEL_PATH and not os.path.exists(LLM_MODEL_PATH):
    raise FileNotFoundError(f"LLM model not found: {LLM_MODEL_PATH}")

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
