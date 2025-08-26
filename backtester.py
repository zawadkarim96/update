import pandas as pd
from strategy_manager import StrategyManager
from risk_manager import RiskManager

def backtest(df: pd.DataFrame, strategy_name: str, initial_capital=10000, transaction_cost=0.0001, timeframe='M1', temp_strategy=None):
    if timeframe == 'H1':
        df = df.resample('H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    elif timeframe == 'D1':
        df = df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    
    rm = RiskManager(initial_capital)

    if temp_strategy is None:
        sm = StrategyManager([strategy_name])
        signals = sm.apply_strategies(df)
    else:
        sm = None
        signals = pd.DataFrame({'aggregate': temp_strategy(df)}, index=df.index)
    positions = pd.DataFrame(index=df.index, data={'position': 0.0, 'entry_price': 0.0})
    equity = pd.Series(index=df.index, data=initial_capital, dtype=float)  # Set dtype=float to avoid warning
    current_position = 0.0
    entry_price = 0.0

    for i in range(1, len(df)):
        equity.iloc[i] = equity.iloc[i-1]
        sig = signals['aggregate'].iloc[i]
        current_price = df['Close'].iloc[i]

        if current_position != 0:
            direction = 1 if current_position > 0 else -1
            sl, tp = rm.set_sl_tp(df.iloc[:i], entry_price, direction)
            trail_sl = rm.trailing_stop(current_price, entry_price, direction, df=df.iloc[:i])
            if trail_sl is not None:
                sl = trail_sl
            if (direction == 1 and (current_price <= sl or current_price >= tp)) or \
               (direction == -1 and (current_price >= sl or current_price <= tp)):
                pl = (current_price - entry_price) * abs(current_position) * direction
                equity.iloc[i] += pl - abs(current_position) * current_price * transaction_cost
                current_position = 0.0
                entry_price = 0.0

        if sig != 0 and current_position == 0:
            direction = sig
            entry = current_price
            sl, tp = rm.set_sl_tp(df.iloc[:i], entry, direction)
            size = rm.calculate_position_size(entry, sl)
            positions.at[df.index[i], 'position'] = size * direction
            positions.at[df.index[i], 'entry_price'] = entry
            current_position = size * direction
            entry_price = entry
            equity.iloc[i] -= abs(current_position) * entry * transaction_cost

    returns = equity.pct_change().fillna(0)
    positive_trades = (returns > 0).sum()
    total_trades = len(returns[returns != 0])
    win_rate = positive_trades / total_trades if total_trades > 0 else 0
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    cum_returns = (1 + returns).cumprod() - 1
    sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() != 0 else 0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    return {"equity_curve": equity, "sharpe": sharpe, "cum_returns": cum_returns, "win_rate": win_rate, "profit_factor": profit_factor, "max_drawdown": max_drawdown}

def simulate_trades(df, strategy_name):
    return backtest(df, strategy_name)
