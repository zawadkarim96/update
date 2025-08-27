"""Broker interfaces used by the trading system.

This module exposes two brokers:

``MT5Broker``
    Real broker that connects to MetaTrader5.

``MockBroker``
    Lightweight stand‑in used when the MetaTrader5 package is not
    available.  It produces synthetic historical data and implements the
    required methods as no‑ops so the rest of the system can operate in a
    testing environment.
"""

from __future__ import annotations

import logging
import time

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover - executed when pandas isn't available
    raise SystemExit(
        "pandas is required for broker interfaces. Please install it via 'pip install pandas'."
    ) from e

from config import MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER, MT5_PATH

try:  # MetaTrader5 is optional – tests may run without it
    import MetaTrader5 as mt5  # type: ignore
    MT5_STUB = getattr(mt5, "META_TRADER5_STUB", False)
except Exception:  # pragma: no cover - executed when MT5 isn't installed
    import metatrader5_stub as mt5  # type: ignore
    MT5_STUB = True

logger = logging.getLogger(__name__)

class MT5Broker:
    """Broker implementation backed by MetaTrader5."""

    def __init__(self) -> None:
        if MT5_STUB:
            raise RuntimeError("MetaTrader5 package is not available")
        if not mt5.initialize(path=MT5_PATH):
            raise Exception("MT5 initialization failed")
        for _ in range(3):  # Retry login
            if mt5.login(MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER):
                break
            time.sleep(5)
        else:
            raise Exception("MT5 login failed after retries")

    def get_historical_data(self, symbol, timeframe, count=1000):
        if not mt5.symbol_select(symbol, True):
            raise Exception(f"Failed to select symbol {symbol}")
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, timeframe), 0, count)
        if rates is None or len(rates) == 0:
            raise Exception(f"Failed to fetch data for {symbol}")
        df = pd.DataFrame(rates)
        # Handle if columns are numerical (list of tuples case in older versions)
        if len(df.columns) > 0 and isinstance(df.columns[0], int):
            df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        logger.info(f"DataFrame columns after fetch for {symbol}: {df.columns.tolist()}")
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def get_real_time_data(self, symbol, timeframe, from_date, to_date):
        if not mt5.symbol_select(symbol, True):
            raise Exception(f"Failed to select symbol {symbol}")
        rates = mt5.copy_rates_range(symbol, getattr(mt5, timeframe), from_date, to_date)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        # Handle numerical columns if needed
        if len(df.columns) > 0 and isinstance(df.columns[0], int):
            df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        logger.info(f"Real-time DataFrame columns for {symbol}: {df.columns.tolist()}")
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def send_order(self, symbol, action, volume, sl=0, tp=0):
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            raise Exception(f"Failed to get tick for {symbol}")
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if action == 'buy' else tick.bid,
            "sl": sl,
            "tp": tp,
            "magic": 123456,
            "comment": "Bot trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result.comment}")
        return result

    def close_position(self, ticket):
        request = {
            "action": mt5.TRADE_ACTION_CLOSE_BY,
            "position": ticket,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return mt5.order_send(request)

    def get_open_positions(self, symbol=None):
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return positions

    def close(self):
        mt5.shutdown()


class MockBroker:
    """Minimal broker used for environments without MetaTrader5.

    The goal of this class is simply to provide enough behaviour for the
    rest of the system to run.  Historical data is synthesised as a simple
    price series and trading related methods perform no actions.
    """

    def __init__(self) -> None:
        logger.info("Using MockBroker – MetaTrader5 is unavailable")

    # ------------------------------------------------------------------
    def _timeframe_to_freq(self, timeframe: str) -> str:
        mapping = {
            "TIMEFRAME_M1": "1min",
            "TIMEFRAME_M5": "5min",
            "TIMEFRAME_M15": "15min",
            "TIMEFRAME_M30": "30min",
            "TIMEFRAME_H1": "1H",
            "TIMEFRAME_D1": "1D",
        }
        return mapping.get(timeframe, "1min")

    def get_historical_data(self, symbol: str, timeframe: str, count: int = 1000) -> pd.DataFrame:
        freq = self._timeframe_to_freq(timeframe)
        index = pd.date_range(end=pd.Timestamp.utcnow(), periods=count, freq=freq)
        base = pd.Series(range(count), dtype=float)
        df = pd.DataFrame(
            {
                "Open": 100 + base,
                "High": 100.1 + base,
                "Low": 99.9 + base,
                "Close": 100 + base,
                "Volume": 0,
            },
            index=index,
        )
        return df

    def get_real_time_data(
        self, symbol: str, timeframe: str, from_date: pd.Timestamp, to_date: pd.Timestamp
    ) -> pd.DataFrame:
        freq = self._timeframe_to_freq(timeframe)
        index = pd.date_range(start=from_date + pd.to_timedelta(freq), end=to_date, freq=freq)
        if len(index) == 0:
            return pd.DataFrame()
        base = pd.Series(range(len(index)), dtype=float)
        df = pd.DataFrame(
            {
                "Open": 100 + base,
                "High": 100.1 + base,
                "Low": 99.9 + base,
                "Close": 100 + base,
                "Volume": 0,
            },
            index=index,
        )
        return df

    def send_order(self, symbol: str, action: str, volume: float, sl: float = 0, tp: float = 0):
        logger.info(
            f"Mock order: {action} {symbol} volume={volume} sl={sl} tp={tp}"
        )

    def close_position(self, ticket):  # pragma: no cover - mock method
        logger.info(f"Mock close position: {ticket}")

    def get_open_positions(self, symbol=None):
        return []

    def close(self):
        logger.info("MockBroker shutdown")
