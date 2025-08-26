import talib
import MetaTrader5 as mt5
import pandas as pd
from config import RISK_PER_TRADE, INITIAL_CAPITAL

class RiskManager:
    def __init__(self, account_balance=INITIAL_CAPITAL):
        self.balance = account_balance

    def update_balance(self, new_balance):
        """Update account balance for dynamic risk sizing."""
        self.balance = max(0, new_balance)

    def calculate_position_size(self, entry_price, sl_price, symbol='EURUSD'):
        """Calculate lot size based on risk percentage and stop loss distance."""
        if not mt5.initialize():
            raise Exception("MT5 not initialized for position sizing")
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            raise ValueError(f"Invalid symbol: {symbol}")
        
        risk_amount = self.balance * RISK_PER_TRADE
        point = symbol_info.point
        contract_size = symbol_info.trade_contract_size
        pip_value = contract_size * point  # Per lot
        risk_pips = abs(entry_price - sl_price) / point
        if risk_pips == 0:
            return 0.01  # Avoid division by zero
        size = risk_amount / (risk_pips * pip_value)
        return max(0.01, round(size, 2))  # Min lot 0.01

    def set_sl_tp(self, df, entry_price, direction, atr_mult_sl=1.5, atr_mult_tp=3.0):
        """Set dynamic stop-loss and take-profit based on ATR and volatility."""
        if len(df) < 14:
            raise ValueError("Insufficient data for ATR calculation")
        atr = talib.ATR(df['High'], df['Low'], df['Close'], 14).iloc[-1]
        if pd.isna(atr) or atr == 0:
            atr = 0.0001  # Fallback to avoid zero ATR
        volatility = atr * atr_mult_sl  # Dynamic based on volatility
        if direction == 1:  # Buy
            sl = entry_price - volatility
            tp = entry_price + atr * atr_mult_tp
        else:  # Sell
            sl = entry_price + volatility
            tp = entry_price - atr * atr_mult_tp
        return sl, tp

    def trailing_stop(self, current_price, entry_price, direction, atr_mult=2.0, df=None):
        """Calculate dynamic trailing stop based on ATR."""
        if df is None or len(df) < 14:
            return None
        atr = talib.ATR(df['High'], df['Low'], df['Close'], 14).iloc[-1]
        if pd.isna(atr) or atr == 0:
            return None
        if direction == 1:
            new_stop = current_price - atr * atr_mult
            return max(new_stop, entry_price - atr * atr_mult)  # Trail up
        elif direction == -1:
            new_stop = current_price + atr * atr_mult
            return min(new_stop, entry_price + atr * atr_mult)  # Trail down
        return None

    def modify_order(self, ticket, sl, symbol):
        """Modify stop-loss for an open position."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            print(f"Position {ticket} not found")
            return False
        position = position[0]
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "position": ticket,
            "sl": sl,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Modify order failed: {result.comment}")
            return False
        return True