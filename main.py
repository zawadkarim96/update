import logging
from broker_interface import MT5Broker, MockBroker
from indicators import get_all_indicators
from strategy_manager import StrategyManager
from risk_manager import RiskManager
from ai_module import AIModule
from learning_engine import LearningEngine
from config import (
    SYMBOLS,
    DEFAULT_TIMEFRAME,
    DEFAULT_STRATEGY_WEIGHTS,
    OPTIMIZATION_INTERVAL,
    POLLING_INTERVAL,
)
import time
import datetime
try:
    import pandas as pd
except ImportError as e:  # pragma: no cover - executed when pandas isn't available
    raise SystemExit(
        "pandas is required to run this application. Please install it via 'pip install pandas'."
    ) from e

try:  # MetaTrader5 may not be installed in some environments
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # pragma: no cover - executed when MT5 isn't available
    import metatrader5_stub as mt5  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(live_mode=False, active_strategies=None):
    if active_strategies is None:
        active_strategies = [
            "moving_average_crossover",
            "rsi_overbought_oversold",
        ]
    try:
        broker = MT5Broker()
    except Exception as e:
        logger.warning(f"Falling back to MockBroker: {e}")
        broker = MockBroker()
    ai = AIModule()
    le = LearningEngine()
    rm = RiskManager()
    sm = StrategyManager(active_strategies, DEFAULT_STRATEGY_WEIGHTS)
    
    if not live_mode:
        # Demo/backtest mode for multiple symbols
        for symbol in SYMBOLS:
            try:
                df = broker.get_historical_data(symbol, DEFAULT_TIMEFRAME, 5000)
                logger.info(f"Fetched {len(df)} rows for {symbol}")
                if len(df) == 0:
                    logger.warning(f"No data fetched for {symbol}. Skipping.")
                    continue
                df = get_all_indicators(df, include_price=True)  # Ensure price columns are included
                logger.info(f"DataFrame columns after indicators for {symbol}: {df.columns.tolist()}")
                signals = sm.apply_strategies(df)
                latest_sig = signals['aggregate'].iloc[-1]
                if latest_sig != 0:
                    entry = df['Close'].iloc[-1]
                    sl, tp = rm.set_sl_tp(df, entry, latest_sig)
                    size = rm.calculate_position_size(entry, sl)
                    action = 'buy' if latest_sig > 0 else 'sell'
                    broker.send_order(symbol, action, size, sl, tp)
                le.optimize_strategies(df)
                le.self_reflect(df)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    else:
        # Live real-time mode for multiple symbols
        data_dict = {}  # Store last data per symbol
        for symbol in SYMBOLS:
            try:
                df = broker.get_historical_data(symbol, DEFAULT_TIMEFRAME, 5000)
                if len(df) == 0:
                    logger.warning(f"No initial data for {symbol}. Skipping.")
                    continue
                data_dict[symbol] = {'df': get_all_indicators(df, include_price=True), 'last_time': df.index[-1]}  # Ensure price columns
                logger.info(f"Initial DataFrame columns after indicators for {symbol}: {data_dict[symbol]['df'].columns.tolist()}")
            except Exception as e:
                logger.error(f"Error initializing {symbol}: {e}")
                continue
        
        while True:
            now = datetime.datetime.now()
            for symbol, info in data_dict.items():
                try:
                    new_df = broker.get_real_time_data(symbol, DEFAULT_TIMEFRAME, info['last_time'], now)
                    if not new_df.empty:
                        info['df'] = pd.concat([info['df'], new_df]).drop_duplicates()
                        info['df'] = get_all_indicators(info['df'], include_price=True)  # Ensure price columns
                        logger.info(f"Updated DataFrame columns after indicators for {symbol}: {info['df'].columns.tolist()}")
                        signals = sm.apply_strategies(info['df'])
                        latest_sig = signals['aggregate'].iloc[-1]
                        if latest_sig != 0:
                            entry = info['df']['Close'].iloc[-1]
                            sl, tp = rm.set_sl_tp(info['df'], entry, latest_sig)
                            size = rm.calculate_position_size(entry, sl)
                            action = 'buy' if latest_sig > 0 else 'sell'
                            broker.send_order(symbol, action, size, sl, tp)
                        info['last_time'] = info['df'].index[-1]
                    
                    # Manage open positions with trailing stops
                    positions = broker.get_open_positions(symbol)
                    for pos in positions:
                        direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                        trail = rm.trailing_stop(info['df']['Close'].iloc[-1], pos.price_open, direction, df=info['df'])
                        if trail is not None and pos.sl != trail:
                            rm.modify_order(pos.ticket, trail, symbol)
                except Exception as e:
                    logger.error(f"Error in live loop for {symbol}: {e}")
            
            # Nightly optimization check (3:24 AM +06 daily)
            if now.hour == 3 and now.minute == 24 and OPTIMIZATION_INTERVAL == "daily":
                for symbol, info in data_dict.items():
                    le.optimize_strategies(info['df'])
                    le.self_reflect(info['df'])
            
            time.sleep(POLLING_INTERVAL)

    broker.close()

if __name__ == "__main__":
    main(live_mode=False)  # Toggle to True for live mode
