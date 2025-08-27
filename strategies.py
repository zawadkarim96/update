import numpy as np
import pandas as pd

try:  # TA‑Lib is optional; provide fallbacks when unavailable
    import talib  # type: ignore
except Exception:  # pragma: no cover - executed when TA‑Lib isn't installed
    talib = None


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return RSI using TA‑Lib when available, otherwise a pandas fallback."""
    if talib is not None:  # pragma: no cover - exercised when TA‑Lib is installed
        return talib.RSI(series, timeperiod=period)

    # pandas implementation based on Wilder's smoothing
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 1
def moving_average_crossover(df, multi_data=None, sym=None, news_bias=0):
    """Simple moving‑average crossover strategy.

    When TA‑Lib is unavailable the moving averages are computed using
    ``pandas`` to keep the strategy functional for tests.
    """
    if talib is not None:
        short_ma = talib.SMA(df['Close'], timeperiod=50)
        long_ma = talib.SMA(df['Close'], timeperiod=200)
    else:  # pragma: no cover - used when TA‑Lib isn't installed
        short_ma = df['Close'].rolling(50).mean()
        long_ma = df['Close'].rolling(200).mean()
    return np.where(short_ma > long_ma, 1, np.where(short_ma < long_ma, -1, 0))

# 2
def golden_cross(df, multi_data=None, sym=None, news_bias=0):
    short_ma = talib.SMA(df['Close'], timeperiod=50)
    long_ma = talib.SMA(df['Close'], timeperiod=200)
    cross = (short_ma.shift(1) < long_ma.shift(1)) & (short_ma > long_ma)
    return np.where(cross, 1, 0)

# 3
def death_cross(df, multi_data=None, sym=None, news_bias=0):
    short_ma = talib.SMA(df['Close'], timeperiod=50)
    long_ma = talib.SMA(df['Close'], timeperiod=200)
    cross = (short_ma.shift(1) > long_ma.shift(1)) & (short_ma < long_ma)
    return np.where(cross, -1, 0)

# 4
def macd_crossover(df, multi_data=None, sym=None, news_bias=0):
    macd, signal, hist = talib.MACD(df['Close'])
    return np.where(macd > signal, 1, np.where(macd < signal, -1, 0))

# 5
def macd_divergence(df, multi_data=None, sym=None, news_bias=0):
    macd, signal, hist = talib.MACD(df['Close'])
    price_diff = df['Close'] - df['Close'].shift(5)
    macd_diff = macd - macd.shift(5)
    return np.where((price_diff > 0) & (macd_diff < 0), -1,
                   np.where((price_diff < 0) & (macd_diff > 0), 1, 0))
# 6
def rsi_overbought_oversold(df, multi_data=None, sym=None, news_bias=0):
    rsi = _rsi(df['Close'])
    return np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))

# 7
def rsi_divergence(df, multi_data=None, sym=None, news_bias=0):
    rsi = _rsi(df['Close'])
    price_diff = df['Close'] - df['Close'].shift(5)
    rsi_diff = rsi - rsi.shift(5)
    return np.where((price_diff > 0) & (rsi_diff < 0), -1,
                    np.where((price_diff < 0) & (rsi_diff > 0), 1, 0))

# 8
def stochastic_oscillator_signals(df, multi_data=None, sym=None, news_bias=0):
    k, d = talib.STOCH(df['High'], df['Low'], df['Close'])
    return np.where(k > d, 1, np.where(k < d, -1, 0))

# 9
def bollinger_band_mean_reversion(df, multi_data=None, sym=None, news_bias=0):
    upper, middle, lower = talib.BBANDS(df['Close'])
    return np.where(df['Close'] < lower, 1, np.where(df['Close'] > upper, -1, 0))

# 10
def bollinger_band_squeeze(df, multi_data=None, sym=None, news_bias=0):
    upper, middle, lower = talib.BBANDS(df['Close'])
    band_width = (upper - lower) / middle
    squeeze = band_width < band_width.rolling(20).mean() * 0.5
    return np.where(squeeze, 1, 0)

# 11
def parabolic_sar(df, multi_data=None, sym=None, news_bias=0):
    sar = talib.SAR(df['High'], df['Low'])
    return np.where(df['Close'] > sar, 1, np.where(df['Close'] < sar, -1, 0))

# 12
def adx_trend_trading(df, multi_data=None, sym=None, news_bias=0):
    adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    plus_di = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    minus_di = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    return np.where((adx > 25) & (plus_di > minus_di), 1,
                   np.where((adx > 25) & (plus_di < minus_di), -1, 0))
# 13
def keltner_channel_breakout(df, multi_data=None, sym=None, news_bias=0):
    ema = talib.EMA(df['Close'], timeperiod=20)
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=10)
    upper = ema + 2 * atr
    lower = ema - 2 * atr
    return np.where(df['Close'] > upper, 1, np.where(df['Close'] < lower, -1, 0))

# 14
def donchian_channel_breakout(df, multi_data=None, sym=None, news_bias=0):
    upper = df['High'].rolling(20).max()
    lower = df['Low'].rolling(20).min()
    return np.where(df['Close'] > upper, 1, np.where(df['Close'] < lower, -1, 0))

# 15
def ichimoku_cloud(df, multi_data=None, sym=None, news_bias=0):
    nine_high = df['High'].rolling(9).max()
    nine_low = df['Low'].rolling(9).min()
    tenkan = (nine_high + nine_low) / 2
    period26_high = df['High'].rolling(26).max()
    period26_low = df['Low'].rolling(26).min()
    kijun = (period26_high + period26_low) / 2
    return np.where(tenkan > kijun, 1, np.where(tenkan < kijun, -1, 0))

# 16
def pivot_point_reversal(df, multi_data=None, sym=None, news_bias=0):
    pivot = (df['High'] + df['Low'] + df['Close']) / 3
    return np.where(df['Close'] > pivot, 1, np.where(df['Close'] < pivot, -1, 0))

# 17
def support_bounce(df, multi_data=None, sym=None, news_bias=0):
    support = df['Low'].rolling(50).min()
    return np.where((df['Close'] > support) & (df['Close'].shift(1) <= support), 1, 0)

# 18
def support_breakout(df, multi_data=None, sym=None, news_bias=0):
    support = df['Low'].rolling(50).min()
    return np.where(df['Close'] < support, -1, 0)

# 19
def resistance_bounce(df, multi_data=None, sym=None, news_bias=0):
    resistance = df['High'].rolling(50).max()
    return np.where((df['Close'] < resistance) & (df['Close'].shift(1) >= resistance), -1, 0)

# 20
def resistance_breakout(df, multi_data=None, sym=None, news_bias=0):
    resistance = df['High'].rolling(50).max()
    return np.where(df['Close'] > resistance, 1, 0)

# 21
def trendline_break(df, multi_data=None, sym=None, news_bias=0):
    rolling_max = df['High'].rolling(20).max()
    rolling_min = df['Low'].rolling(20).min()
    break_up = df['Close'] > rolling_max.shift(1)
    break_down = df['Close'] < rolling_min.shift(1)
    return np.where(break_up, 1, np.where(break_down, -1, 0))

# 22
def head_and_shoulders(df, multi_data=None, sym=None, news_bias=0):
    highs = df['High']
    signals = np.zeros(len(df))
    for i in range(2, len(df)-2):
        if highs[i-2] < highs[i-1] and highs[i-1] > highs[i] and highs[i] < highs[i+1] and highs[i+1] > highs[i+2]:
            signals[i] = -1
    return signals

# 23
def inverse_head_and_shoulders(df, multi_data=None, sym=None, news_bias=0):
    lows = df['Low']
    signals = np.zeros(len(df))
    for i in range(2, len(df)-2):
        if lows[i-2] > lows[i-1] and lows[i-1] < lows[i] and lows[i] > lows[i+1] and lows[i+1] < lows[i+2]:
            signals[i] = 1
    return signals

# 24
def double_top(df, multi_data=None, sym=None, news_bias=0):
    highs = df['High']
    signals = np.zeros(len(df))
    for i in range(2, len(df)):
        if abs(highs[i] - highs[i-2]) / highs[i-2] < 0.001 and highs[i] < highs[i-1]:
            signals[i] = -1
    return signals

# 25
def double_bottom(df, multi_data=None, sym=None, news_bias=0):
    lows = df['Low']
    signals = np.zeros(len(df))
    for i in range(2, len(df)):
        if abs(lows[i] - lows[i-2]) / lows[i-2] < 0.001 and lows[i] > lows[i-1]:
            signals[i] = 1
    return signals

# 26
def cup_and_handle(df, multi_data=None, sym=None, news_bias=0):
    rolling_max = df['High'].rolling(30).max()
    handle_break = df['Close'] > rolling_max.shift(5)
    return np.where(handle_break, 1, 0)

# 27
def ascending_triangle_breakout(df, multi_data=None, sym=None, news_bias=0):
    resistance = df['High'].rolling(20).max()
    higher_lows = df['Low'].rolling(3).min().diff() > 0
    breakout = df['Close'] > resistance
    return np.where(breakout & higher_lows, 1, 0)

# 28
def descending_triangle_breakout(df, multi_data=None, sym=None, news_bias=0):
    support = df['Low'].rolling(20).min()
    lower_highs = df['High'].rolling(3).max().diff() < 0
    breakout = df['Close'] < support
    return np.where(breakout & lower_highs, -1, 0)

# 29
def symmetrical_triangle_breakout(df, multi_data=None, sym=None, news_bias=0):
    lower_highs = df['High'].rolling(3).max().diff() < 0
    higher_lows = df['Low'].rolling(3).min().diff() > 0
    breakout_up = (lower_highs & higher_lows) & (df['Close'] > df['High'].shift(1))
    breakout_down = (lower_highs & higher_lows) & (df['Close'] < df['Low'].shift(1))
    return np.where(breakout_up, 1, np.where(breakout_down, -1, 0))

# 30
def bullish_flag(df, multi_data=None, sym=None, news_bias=0):
    run_up = df['Close'].diff(5) > 0
    small_pullback = df['Close'].diff() < 0
    breakout = df['Close'] > df['High'].shift(1)
    return np.where(run_up & small_pullback & breakout, 1, 0)

# 31
def bearish_flag(df, multi_data=None, sym=None, news_bias=0):
    run_down = df['Close'].diff(5) < 0
    small_pullback = df['Close'].diff() > 0
    breakout = df['Close'] < df['Low'].shift(1)
    return np.where(run_down & small_pullback & breakout, -1, 0)

# 32
def bullish_pennant(df, multi_data=None, sym=None, news_bias=0):
    pole = df['Close'].diff(5) > 0
    contraction = (df['High'] - df['Low']).rolling(5).mean().diff() < 0
    breakout = df['Close'] > df['High'].shift(1)
    return np.where(pole & contraction & breakout, 1, 0)

# 33
def bearish_pennant(df, multi_data=None, sym=None, news_bias=0):
    pole = df['Close'].diff(5) < 0
    contraction = (df['High'] - df['Low']).rolling(5).mean().diff() < 0
    breakout = df['Close'] < df['Low'].shift(1)
    return np.where(pole & contraction & breakout, -1, 0)

# 34
def rising_wedge(df, multi_data=None, sym=None, news_bias=0):
    rising = (df['High'].diff() > 0) & (df['Low'].diff() > 0)
    narrowing = (df['High'] - df['Low']).diff() < 0
    return np.where(rising & narrowing & (df['Close'] < df['Low'].shift(1)), -1, 0)

# 35
def falling_wedge(df, multi_data=None, sym=None, news_bias=0):
    falling = (df['High'].diff() < 0) & (df['Low'].diff() < 0)
    narrowing = (df['High'] - df['Low']).diff() < 0
    return np.where(falling & narrowing & (df['Close'] > df['High'].shift(1)), 1, 0)

# 36
def price_channel_buy_sell(df, multi_data=None, sym=None, news_bias=0):
    upper = df['High'].rolling(20).max()
    lower = df['Low'].rolling(20).min()
    return np.where(df['Close'] <= lower, 1, np.where(df['Close'] >= upper, -1, 0))

# 37
def hammer_signal(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, 1, 0)

# 38
def hanging_man_signal(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, -1, 0)

# 39
def doji_reversal(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, 1, 0)

# 40
def morning_star(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, 1, 0)

# 41
def evening_star(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, -1, 0)

# 42
def bullish_engulfing(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern > 0, 1, 0)

# 43
def bearish_engulfing(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern < 0, -1, 0)

# 44
def piercing_pattern(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, 1, 0)

# 45
def dark_cloud_cover(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, -1, 0)

# 46
def abandoned_baby(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDLABANDONEDBABY(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, 1, 0)

# 47
def three_white_soldiers(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, 1, 0)

# 48
def three_black_crows(df, multi_data=None, sym=None, news_bias=0):
    pattern = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
    return np.where(pattern != 0, -1, 0)

# 49
def scalping(df, multi_data=None, sym=None, news_bias=0):
    rsi = talib.RSI(df['Close'], 5)
    return np.where(rsi < 20, 1, np.where(rsi > 80, -1, 0))

# 50
def day_trading(df, multi_data=None, sym=None, news_bias=0):
    intraday_vol = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    return np.where(df['Close'] > df['Open'] + intraday_vol, 1,
                    np.where(df['Close'] < df['Open'] - intraday_vol, -1, 0))

# 51
def swing_trading(df, multi_data=None, sym=None, news_bias=0):
    sma20 = talib.SMA(df['Close'], 20)
    sma50 = talib.SMA(df['Close'], 50)
    return np.where(sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0))

# 52
def position_trading(df, multi_data=None, sym=None, news_bias=0):
    sma50 = talib.SMA(df['Close'], 50)
    sma200 = talib.SMA(df['Close'], 200)
    return np.where(sma50 > sma200, 1, np.where(sma50 < sma200, -1, 0))

# 53
def trend_following(df, multi_data=None, sym=None, news_bias=0):
    return np.where(df['Close'] > talib.EMA(df['Close'], 50), 1, -1)

# 54
def mean_reversion(df, multi_data=None, sym=None, news_bias=0):
    mean = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    return np.where(df['Close'] < mean - std, 1, np.where(df['Close'] > mean + std, -1, 0))

# 55
def momentum_trading(df, multi_data=None, sym=None, news_bias=0):
    roc = talib.ROC(df['Close'], 10)
    return np.where(roc > 0, 1, np.where(roc < 0, -1, 0))

# 56
def range_trading(df, multi_data=None, sym=None, news_bias=0):
    upper = df['High'].rolling(20).max()
    lower = df['Low'].rolling(20).min()
    return np.where(df['Close'] <= lower, 1, np.where(df['Close'] >= upper, -1, 0))

# 57
def breakout_trading(df, multi_data=None, sym=None, news_bias=0):
    resistance = df['High'].rolling(20).max()
    support = df['Low'].rolling(20).min()
    return np.where(df['Close'] > resistance, 1, np.where(df['Close'] < support, -1, 0))

# 58
def reversal_trading(df, multi_data=None, sym=None, news_bias=0):
    rsi = talib.RSI(df['Close'], 14)
    return np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))

# 59
def gap_fade(df, multi_data=None, sym=None, news_bias=0):
    gap = df['Open'] - df['Close'].shift(1)
    return np.where(gap > 0, -1, np.where(gap < 0, 1, 0))

# 60
def gap_and_go(df, multi_data=None, sym=None, news_bias=0):
    gap = df['Open'] - df['Close'].shift(1)
    return np.where((gap > 0) & (df['Close'] > df['Open']), 1,
                    np.where((gap < 0) & (df['Close'] < df['Open']), -1, 0))

# 61
def news_trading(df, multi_data=None, sym=None, news_bias=0):
    return np.where(news_bias > 0, 1, np.where(news_bias < 0, -1, 0))

# 62
def event_driven_trading(df, multi_data=None, sym=None, news_bias=0):
    return news_trading(df, multi_data, sym, news_bias)

# 63
def sentiment_contrarian(df, multi_data=None, sym=None, news_bias=0):
    return np.where(news_bias > 0, -1, np.where(news_bias < 0, 1, 0))

# 64
def cot_fade(df, multi_data=None, sym=None, news_bias=0):
    rsi = talib.RSI(df['Close'], timeperiod=14)
    return np.where(rsi > 60, -1, np.where(rsi < 40, 1, 0))

# 65
def carry_trade(df, multi_data=None, sym=None, news_bias=0):
    if multi_data and len(multi_data) > 0:
        base = df['Close'].pct_change()
        other = multi_data[0]['Close'].pct_change()
        spread = base - other
        return np.where(spread > 0, 1, np.where(spread < 0, -1, 0))
    return trend_following(df, multi_data, sym, news_bias)

# 66
def triangular_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return pairs_trading(df, multi_data, sym, news_bias)

# 67
def pure_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return pairs_trading(df, multi_data, sym, news_bias)

# 68
def exchange_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return pairs_trading(df, multi_data, sym, news_bias)

# 69
def statistical_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return correlation_fade(df, multi_data, sym, news_bias)

# 70
def pairs_trading(df, multi_data=None, sym=None, news_bias=0):
    if multi_data and len(multi_data) >= 2:
        s1 = multi_data[0]['Close']
        s2 = multi_data[1]['Close']
        spread = s1 - s2
        zscore = (spread - spread.mean()) / spread.std()
        return np.where(zscore > 1, -1, np.where(zscore < -1, 1, 0))
    return np.zeros(len(df))

# 71
def index_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return pairs_trading(df, multi_data, sym, news_bias)

# 72
def merger_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return news_trading(df, multi_data, sym, news_bias)

# 73
def convertible_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return pairs_trading(df, multi_data, sym, news_bias)

# 74
def long_short_equity(df, multi_data=None, sym=None, news_bias=0):
    return np.where(df['Close'] > df['Open'], 1, -1)

# 75
def market_neutral(df, multi_data=None, sym=None, news_bias=0):
    return rsi_overbought_oversold(df, multi_data, sym, news_bias)

# 76
def volatility_trading(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    return np.where(atr > atr.mean(), 1, -1)

# 77
def dispersion_trading(df, multi_data=None, sym=None, news_bias=0):
    return bollinger_band_squeeze(df, multi_data, sym, news_bias)

# 78
def calendar_spread(df, multi_data=None, sym=None, news_bias=0):
    return pairs_trading(df, multi_data, sym, news_bias)

# 79
def yield_curve_steepener(df, multi_data=None, sym=None, news_bias=0):
    return trend_following(df, multi_data, sym, news_bias)

# 80
def yield_curve_flattener(df, multi_data=None, sym=None, news_bias=0):
    return -trend_following(df, multi_data, sym, news_bias)

# 81
def high_frequency_market_making(df, multi_data=None, sym=None, news_bias=0):
    return scalping(df, multi_data, sym, news_bias)

# 82
def latency_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return scalping(df, multi_data, sym, news_bias)

# 83
def iceberg_tape_reading(df, multi_data=None, sym=None, news_bias=0):
    high_vol = df['Volume'] > df['Volume'].rolling(20).mean() * 2
    return np.where(high_vol & (df['Close'] > df['Open']), 1,
                    np.where(high_vol & (df['Close'] < df['Open']), -1, 0))

# 84
def vwap_execution(df, multi_data=None, sym=None, news_bias=0):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return np.where(df['Close'] > vwap, 1, -1)

# 85
def twap_execution(df, multi_data=None, sym=None, news_bias=0):
    twap = df['Close'].expanding().mean()
    return np.where(df['Close'] > twap, 1, -1)

# 86
def grid_trading(df, multi_data=None, sym=None, news_bias=0):
    mean_price = df['Close'].mean()
    return np.where(df['Close'] < mean_price * 0.98, 1, np.where(df['Close'] > mean_price * 1.02, -1, 0))

# 87
def martingale(df, multi_data=None, sym=None, news_bias=0):
    return np.where(df['Close'] > df['Open'], 1, -1)

# 88
def anti_martingale(df, multi_data=None, sym=None, news_bias=0):
    return np.where(df['Close'] > df['Open'], 1, -1)

# 89
def kelly_criterion_sizing(df, multi_data=None, sym=None, news_bias=0):
    return trend_following(df, multi_data, sym, news_bias)

# 90
def fixed_fractional_sizing(df, multi_data=None, sym=None, news_bias=0):
    return trend_following(df, multi_data, sym, news_bias)

# 91
def dollar_cost_averaging(df, multi_data=None, sym=None, news_bias=0):
    return np.where(df.index % 10 == 0, 1, 0)

# 92
def seasonality(df, multi_data=None, sym=None, news_bias=0):
    months = df.index.month
    return np.where((months == 11) | (months == 12), 1, 0)

# 93
def cycle_trading(df, multi_data=None, sym=None, news_bias=0):
    sine, leadsine = talib.HT_SINE(df['Close'])
    return np.where(sine > leadsine, 1, np.where(sine < leadsine, -1, 0))

# 94
def gann_angle_strategies(df, multi_data=None, sym=None, news_bias=0):
    return trendline_break(df, multi_data, sym, news_bias)

# 95
def fibonacci_retracement_extension(df, multi_data=None, sym=None, news_bias=0):
    high = df['High'].rolling(50).max()
    low = df['Low'].rolling(50).min()
    level_618 = high - (high - low) * 0.618
    return np.where(df['Close'] < level_618, 1, np.where(df['Close'] > level_618, -1, 0))

# 96
def elliott_wave_counting(df, multi_data=None, sym=None, news_bias=0):
    return momentum_trading(df, multi_data, sym, news_bias)

# 97
def ichimoku_tenkan_kijun_cross(df, multi_data=None, sym=None, news_bias=0):
    nine_high = df['High'].rolling(9).max()
    nine_low = df['Low'].rolling(9).min()
    tenkan = (nine_high + nine_low) / 2
    period26_high = df['High'].rolling(26).max()
    period26_low = df['Low'].rolling(26).min()
    kijun = (period26_high + period26_low) / 2
    return np.where(tenkan > kijun, 1, np.where(tenkan < kijun, -1, 0))

# 98
def volume_profile_value_area_trades(df, multi_data=None, sym=None, news_bias=0):
    return supply_demand_zone_trading(df, multi_data, sym, news_bias)

# 99
def order_flow_imbalance(df, multi_data=None, sym=None, news_bias=0):
    delta = np.where(df['Close'] > df['Open'], df['Volume'], -df['Volume'])
    imbalance = talib.EMA(delta, timeperiod=5)
    return np.where(imbalance > 0, 1, np.where(imbalance < 0, -1, 0))

# 100
def footprint_chart_trading(df, multi_data=None, sym=None, news_bias=0):
    delta = np.where(df['Close'] > df['Open'], df['Volume'], -df['Volume'])
    cum_delta = delta.cumsum()
    divergence_bull = (df['Close'].diff() < 0) & (cum_delta.diff() > 0)
    divergence_bear = (df['Close'].diff() > 0) & (cum_delta.diff() < 0)
    return np.where(divergence_bull, 1, np.where(divergence_bear, -1, 0))

# 101
def supply_demand_zone_trading(df, multi_data=None, sym=None, news_bias=0):
    support = df['Low'].rolling(20).min()
    resistance = df['High'].rolling(20).max()
    high_vol = df['Volume'] > df['Volume'].rolling(20).mean() * 1.5
    demand_bounce = (df['Close'] > support) & (df['Close'].shift(1) <= support) & high_vol
    supply_bounce = (df['Close'] < resistance) & (df['Close'].shift(1) >= resistance) & high_vol
    return np.where(demand_bounce, 1, np.where(supply_bounce, -1, 0))

# 102
def wyckoff_accumulation_distribution(df, multi_data=None, sym=None, news_bias=0):
    vol_trend = talib.LINEARREG_SLOPE(df['Volume'], 20)
    price_trend = talib.LINEARREG_SLOPE(df['Close'], 20)
    accum = (vol_trend < 0) & (price_trend > 0)
    distrib = (vol_trend < 0) & (price_trend < 0)
    return np.where(accum, 1, np.where(distrib, -1, 0))

# 103
def institutional_footprint_scalp(df, multi_data=None, sym=None, news_bias=0):
    vol_spike = df['Volume'] > df['Volume'].rolling(50).mean() * 3
    return np.where(vol_spike & (df['Close'] > df['Open']), 1, np.where(vol_spike & (df['Close'] < df['Open']), -1, 0))

# 104
def market_profile_reversion(df, multi_data=None, sym=None, news_bias=0):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return np.where(df['Close'] < vwap, 1, np.where(df['Close'] > vwap, -1, 0))

# 105
def delta_divergence(df, multi_data=None, sym=None, news_bias=0):
    delta = np.where(df['Close'] > df['Open'], df['Volume'], -df['Volume'])
    cum_delta = delta.cumsum()
    price_diff = df['Close'].diff(5)
    delta_diff = cum_delta.diff(5)
    return np.where((price_diff > 0) & (delta_diff < 0), -1,
                    np.where((price_diff < 0) & (delta_diff > 0), 1, 0))

# 106
def vix_mean_reversion(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    vix = multi_data[0]['Close']
    mean = vix.rolling(200).mean()
    std = vix.rolling(200).std()
    return np.where(vix > mean + 2*std, 1, np.where(vix < mean - 2*std, -1, 0))

# 107
def vix_breakout_plays(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    vix = multi_data[0]['Close']
    upper = vix.rolling(20).max().shift(1)
    return np.where(vix > upper, -1, 0)

# 108
def bitcoin_halving_season_play(df, multi_data=None, sym=None, news_bias=0):
    if 'BTC' not in (sym or ''):
        return np.zeros(len(df))
    halvings = pd.to_datetime(['2024-04-20', '2028-04-20'])
    signals = np.zeros(len(df))
    for h in halvings:
        pre = (df.index > h - pd.Timedelta(180, 'D')) & (df.index < h)
        signals[pre] = 1
    return signals

# 109
def momentum_rotation(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    rocs = pd.DataFrame({'0': talib.ROC(df['Close'], 90)})
    for i, d in enumerate(multi_data, 1):
        rocs[str(i)] = talib.ROC(d['Close'], 90)
    ranks = rocs.rank(axis=1, ascending=False)
    return np.where(ranks['0'] == 1, 1, np.where(ranks['0'] == rocs.shape[1], -1, 0))

# 110
def risk_parity_rebalancing(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    rets = pd.DataFrame({'0': df['Close'].pct_change()})
    for i, d in enumerate(multi_data, 1):
        rets[str(i)] = d['Close'].pct_change()
    vols = rets.rolling(20).std()
    inv_vol = 1 / vols.replace(0, np.inf)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
    avg_w = 1 / vols.shape[1]
    return np.where(weights['0'] > avg_w, 1, np.where(weights['0'] < avg_w, -1, 0))

# 111
def equal_weight_rebalancing(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    return np.ones(len(df))

# 112
def sector_rotation(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    rocs = pd.DataFrame({'0': talib.ROC(df['Close'], 30)})
    for i, d in enumerate(multi_data, 1):
        rocs[str(i)] = talib.ROC(d['Close'], 30)
    ranks = rocs.rank(axis=1, ascending=False)
    n = rocs.shape[1]
    return np.where(ranks['0'] <= n*0.3, 1, np.where(ranks['0'] > n*0.7, -1, 0))

# 113
def currency_strength_index_pairing(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    strength = talib.ROC(df['Close'], 14)
    other_strength = np.mean([talib.ROC(d['Close'], 14) for d in multi_data], axis=0)
    rel_strength = strength - other_strength
    return np.where(rel_strength > 0, 1, np.where(rel_strength < 0, -1, 0))

# 114
def correlation_fade(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    asset2 = multi_data[0]['Close']
    corr = df['Close'].rolling(20).corr(asset2)
    high_corr = corr > 0.8
    spread = df['Close'] - asset2
    zscore = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
    return np.where(high_corr & (zscore > 1), -1, np.where(high_corr & (zscore < -1), 1, 0))

# 115
def high_yield_carry_em_fx(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    low_yield = multi_data[0]['Close']
    carry = df['Close'].pct_change() - low_yield.pct_change()
    return np.where(carry > 0, 1, np.where(carry < 0, -1, 0))

# 116
def soft_commodity_spread_trade(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    other = multi_data[0]['Close']
    ratio = df['Close'].mean() / other.mean()
    spread = df['Close'] - other * ratio
    zscore = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
    return np.where(zscore > 1, -1, np.where(zscore < -1, 1, 0))

# 117
def metal_spread_trade(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    other = multi_data[0]['Close']
    ratio = df['Close'] / other
    zscore = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std()
    return np.where(zscore > 2, -1, np.where(zscore < -2, 1, 0))

# 118
def seasonal_agricultural_trading(df, multi_data=None, sym=None, news_bias=0):
    months = df.index.month
    return np.where(months.isin([12,1,2,3]), 1, np.where(months.isin([6,7,8]), -1, 0))

# 119
def overbought_oversold_commodity_channels(df, multi_data=None, sym=None, news_bias=0):
    cci = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    return np.where(cci > 100, -1, np.where(cci < -100, 1, 0))

# 120
def macd_histogram_divergence(df, multi_data=None, sym=None, news_bias=0):
    macd, signal, hist = talib.MACD(df['Close'])
    price_diff = df['Close'].diff(5)
    hist_diff = hist - hist.shift(5)
    return np.where((price_diff > 0) & (hist_diff < 0), -1, np.where((price_diff < 0) & (hist_diff > 0), 1, 0))

# 121
def vwap_reversion(df, multi_data=None, sym=None, news_bias=0):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return np.where(df['Close'] < vwap, 1, np.where(df['Close'] > vwap, -1, 0))

# 122
def vwap_trend_ride(df, multi_data=None, sym=None, news_bias=0):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    trend_up = df['Close'] > df['Close'].shift(1)
    return np.where((df['Close'] > vwap) & trend_up, 1, np.where((df['Close'] < vwap) & ~trend_up, -1, 0))

# 123
def round_number_psych_levels(df, multi_data=None, sym=None, news_bias=0):
    round_level = np.round(df['Close'].shift(1), 0)
    cross_up = (df['Close'] > round_level) & (df['Close'].shift(1) <= round_level)
    cross_down = (df['Close'] < round_level) & (df['Close'].shift(1) >= round_level)
    return np.where(cross_up, 1, np.where(cross_down, -1, 0))

# 124
def fibonacci_time_extensions(df, multi_data=None, sym=None, news_bias=0):
    fib_ratios = [0, 1, 2, 3, 5, 8, 13, 21, 34]
    signals = np.zeros(len(df))
    for r in fib_ratios[2:]:
        idx = min(r, len(df)-1)
        signals[idx] = 1
    return signals

# 125
def calendar_anomaly_weekend_effect(df, multi_data=None, sym=None, news_bias=0):
    dow = df.index.dayofweek
    return np.where(dow == 4, -1, np.where(dow == 0, 1, 0))

# 126
def insider_sentiment_edge(df, multi_data=None, sym=None, news_bias=0):
    return np.where(news_bias > 0, 1, np.where(news_bias < 0, -1, 0))

# 127
def analyst_revision_filter_strategy(df, multi_data=None, sym=None, news_bias=0):
    return np.where(news_bias > 1, 1, np.where(news_bias < -1, -1, 0))

# 128
def short_interest_momentum(df, multi_data=None, sym=None, news_bias=0):
    return momentum_trading(df, multi_data, sym, news_bias)

# 129
def share_buyback_momentum(df, multi_data=None, sym=None, news_bias=0):
    return momentum_trading(df, multi_data, sym, news_bias)

# 130
def option_gamma_scalping(df, multi_data=None, sym=None, news_bias=0):
    return scalping(df, multi_data, sym, news_bias)

# 131
def volatility_skew_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return volatility_trading(df, multi_data, sym, news_bias)

# 132
def covered_call_income_strategy(df, multi_data=None, sym=None, news_bias=0):
    return range_trading(df, multi_data, sym, news_bias)

# 133
def put_write_strategy(df, multi_data=None, sym=None, news_bias=0):
    return mean_reversion(df, multi_data, sym, news_bias)

# 134
def straddle_strangle_breakout(df, multi_data=None, sym=None, news_bias=0):
    return breakout_trading(df, multi_data, sym, news_bias)

# 135
def delta_neutral_rebalance(df, multi_data=None, sym=None, news_bias=0):
    return rsi_overbought_oversold(df, multi_data, sym, news_bias)

# 136
def mean_reversion_pairs_cross_asset(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    asset2 = multi_data[0]['Close']
    spread = df['Close'] - asset2
    zscore = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
    return np.where(zscore > 1, -1, np.where(zscore < -1, 1, 0))

# 137
def crypto_whale_alert_reaction(df, multi_data=None, sym=None, news_bias=0):
    whale = df['Volume'] > df['Volume'].rolling(50).mean() * 5
    return np.where(whale & (df['Close'] > df['Open']), 1, np.where(whale & (df['Close'] < df['Open']), -1, 0))

# 138
def social_media_sentiment_spike_play(df, multi_data=None, sym=None, news_bias=0):
    return np.where(abs(news_bias) > 1, np.sign(news_bias), 0)

# 139
def etf_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    if multi_data is None or len(multi_data) == 0:
        return np.zeros(len(df))
    underlying = multi_data[0]['Close']
    spread = df['Close'] - underlying
    zscore = (spread - spread.mean()) / spread.std()
    return np.where(zscore > 1, -1, np.where(zscore < -1, 1, 0))

# 140
def leveraged_etf_decay_reversion(df, multi_data=None, sym=None, news_bias=0):
    mean = df['Close'].rolling(20).mean()
    return np.where(df['Close'] < mean, 1, np.where(df['Close'] > mean, -1, 0))

# 141
def seasonality_in_fx(df, multi_data=None, sym=None, news_bias=0):
    months = df.index.month
    return np.where(months == 12, -1, 0)

# 142
def central_bank_speculative_position_reactions(df, multi_data=None, sym=None, news_bias=0):
    return news_trading(df, multi_data, sym, news_bias)

# 143
def macro_flow_fed_hike_surprise_arbitrage(df, multi_data=None, sym=None, news_bias=0):
    return news_trading(df, multi_data, sym, news_bias)

# 144
def quant_factor_rotation(df, multi_data=None, sym=None, news_bias=0):
    momentum = talib.ROC(df['Close'], 12)
    value = df['Close'] / df['Close'].rolling(200).mean()
    carry = df['Close'].pct_change()
    factors = pd.DataFrame({'mom': momentum, 'val': 1/value, 'carry': carry})
    ranks = factors.rank(axis=1, ascending=False)
    top_factor = ranks.idxmin(axis=1)
    return np.where(top_factor == 'mom', 1 if momentum.iloc[-1] > 0 else -1, 0)

# 145
def reinforcement_learning_agent(df, multi_data=None, sym=None, news_bias=0):
    return adaptive_ensemble_multi_strategy_blend(df, multi_data, sym, news_bias)

# 146
def neural_net_prediction_model(df, multi_data=None, sym=None, news_bias=0):
    return adaptive_ensemble_multi_strategy_blend(df, multi_data, sym, news_bias)

# 147
def genetic_algorithm_strategy_evolution(df, multi_data=None, sym=None, news_bias=0):
    return adaptive_ensemble_multi_strategy_blend(df, multi_data, sym, news_bias)

# 148
def cluster_analysis_pattern_recognition(df, multi_data=None, sym=None, news_bias=0):
    from sklearn.cluster import KMeans
    features = pd.DataFrame({'ret': df['Close'].pct_change(), 'vol': df['Volume']}).dropna()
    kmeans = KMeans(n_clusters=3).fit(features)
    labels = kmeans.labels_
    signals = np.zeros(len(df))
    signals[1:len(labels)+1] = np.where(labels == 0, 1, np.where(labels == 1, -1, 0))
    return signals

# 149
def anomaly_detection_re_entry(df, multi_data=None, sym=None, news_bias=0):
    from sklearn.ensemble import IsolationForest
    features = pd.DataFrame({'ret': df['Close'].pct_change().fillna(0)})
    iso = IsolationForest().fit(features)
    anomalies = iso.predict(features)
    return np.where(anomalies == -1, 1, 0)

# 150
def adaptive_ensemble_multi_strategy_blend(df, multi_data=None, sym=None, news_bias=0):
    s1 = moving_average_crossover(df, multi_data, sym, news_bias)
    s2 = rsi_overbought_oversold(df, multi_data, sym, news_bias)
    s3 = bollinger_band_mean_reversion(df, multi_data, sym, news_bias)
    avg = (s1 + s2 + s3) / 3
    return np.where(avg > 0.3, 1, np.where(avg < -0.3, -1, 0))

# 151
def ict_fvg(df, multi_data=None, sym=None, news_bias=0):
    h2 = df['High'].shift(2); l2 = df['Low'].shift(2)
    h1 = df['High'].shift(1); l1 = df['Low'].shift(1)
    hb = df['High']; lb = df['Low']
    bull_fvg = l1 > h2
    bear_fvg = h1 < l2
    mid_bull = (h2 + l1) / 2
    mid_bear = (l2 + h1) / 2
    long_sig = bull_fvg & (df['Low'] <= mid_bull) & (df['Close'] > mid_bull)
    short_sig = bear_fvg & (df['High'] >= mid_bear) & (df['Close'] < mid_bear)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 152
def ict_order_block(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    impulse_up = (df['Close'] - df['Open']) > 1.5 * atr
    impulse_dn = (df['Open'] - df['Close']) > 1.5 * atr
    last_down = (df['Close'].shift(1) < df['Open'].shift(1)) & impulse_up
    last_up = (df['Close'].shift(1) > df['Open'].shift(1)) & impulse_dn
    bull_ob = last_down & (df['Low'] <= df['Close'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    bear_ob = last_up & (df['High'] >= df['Close'].shift(1)) & (df['Close'] < df['Close'].shift(1))
    return np.where(bull_ob, 1, np.where(bear_ob, -1, 0))

# 153
def ict_breaker_block(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    up_imp = (df['Close'].shift(2) - df['Open'].shift(2)) > 1.2 * atr.shift(2)
    broke_below = df['Close'] < df['Low'].shift(2)
    retest = df['High'] >= df['Low'].shift(2)
    short_sig = up_imp & broke_below & retest
    dn_imp = (df['Open'].shift(2) - df['Close'].shift(2)) > 1.2 * atr.shift(2)
    broke_above = df['Close'] > df['High'].shift(2)
    retest2 = df['Low'] <= df['High'].shift(2)
    long_sig = dn_imp & broke_above & retest2
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 154
def ict_liquidity_sweep(df, multi_data=None, sym=None, news_bias=0):
    prev_high = df['High'].shift(1).rolling(20).max()
    prev_low = df['Low'].shift(1).rolling(20).min()
    sweep_high = (df['High'] > prev_high) & (df['Close'] < df['Open'])
    sweep_low = (df['Low'] < prev_low) & (df['Close'] > df['Open'])
    return np.where(sweep_low, 1, np.where(sweep_high, -1, 0))

# 155
def ict_pd_array(df, multi_data=None, sym=None, news_bias=0):
    sig_ob = ict_order_block(df, multi_data, sym, news_bias)
    sig_fvg = ict_fvg(df, multi_data, sym, news_bias)
    prev_ext = (df['High'].rolling(30).max() - df['Low'].rolling(30).min()) > 0
    liq_target = prev_ext.astype(int)
    score = sig_ob + sig_fvg + liq_target
    return np.where(score >= 2, np.sign(score), 0)

# 156
def ict_premium_discount_zones(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High'].rolling(50).max()
    lo = df['Low'].rolling(50).min()
    mid = (hi + lo) / 2.0
    long = (df['Close'] < mid) & (df['Close'].shift(1) <= mid) & (df['Close'] > df['Open'])
    short = (df['Close'] > mid) & (df['Close'].shift(1) >= mid) & (df['Close'] < df['Open'])
    return np.where(long, 1, np.where(short, -1, 0))

# 157
def ict_judas_swing(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    early = hour.isin([0,1])
    fake_up = early & (df['High'] == df['High'].rolling(6).max()) & (df['Close'] < df['Open'])
    fake_dn = early & (df['Low'] == df['Low'].rolling(6).min()) & (df['Close'] > df['Open'])
    return np.where(fake_dn, 1, np.where(fake_up, -1, 0))

# 158
def ict_kill_zones_london(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    in_kz = hour.isin([7,8,9,10])
    rng_hi = df['High'].rolling(12).max(); rng_lo = df['Low'].rolling(12).min()
    breakout = (df['Close'] > rng_hi) | (df['Close'] < rng_lo)
    return np.where(in_kz & breakout & (df['Close'] > rng_hi), 1,
                    np.where(in_kz & breakout & (df['Close'] < rng_lo), -1, 0))

# 159
def ict_kill_zones_newyork(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    in_kz = hour.isin([13,14,15,16])
    rng_hi = df['High'].rolling(12).max(); rng_lo = df['Low'].rolling(12).min()
    breakout = (df['Close'] > rng_hi) | (df['Close'] < rng_lo)
    return np.where(in_kz & breakout & (df['Close'] > rng_hi), 1,
                    np.where(in_kz & breakout & (df['Close'] < rng_lo), -1, 0))

# 160
def ict_kill_zones_asian(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    in_kz = hour.isin([0,1,2,3,4,5,6])
    range_tight = (df['High'] - df['Low']).rolling(24).mean()
    squeeze = (df['High'] - df['Low']) < 0.7 * range_tight
    return np.where(in_kz & squeeze, 0, 0)

# 161
def equal_highs_lows_trap(df, multi_data=None, sym=None, news_bias=0):
    eq_high = (df['High'].round(5) == df['High'].shift(1).round(5)) & (df['Close'] < df['Open'])
    eq_low = (df['Low'].round(5) == df['Low'].shift(1).round(5)) & (df['Close'] > df['Open'])
    return np.where(eq_low, 1, np.where(eq_high, -1, 0))

# 162
def ict_turtle_soup(df, multi_data=None, sym=None, news_bias=0):
    hi20 = df['High'].shift(1).rolling(20).max()
    lo20 = df['Low'].shift(1).rolling(20).min()
    long_sig = (df['Low'] < lo20) & (df['Close'] > lo20)
    short_sig = (df['High'] > hi20) & (df['Close'] < hi20)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 163
def breaker_retest(df, multi_data=None, sym=None, news_bias=0):
    base = ict_breaker_block(df, multi_data, sym, news_bias)
    retest = (df['Close'] == df['Close'].rolling(5).max()) | (df['Close'] == df['Close'].rolling(5).min())
    return np.where((base > 0) & retest, 1, np.where((base < 0) & retest, -1, 0))

# 164
def liquidity_void_fill(df, multi_data=None, sym=None, news_bias=0):
    r = df['High'] - df['Low']
    big = r > 1.8 * r.rolling(20).mean()
    mid = (df['High'] + df['Low']) / 2
    long_sig = big.shift(1) & (df['Low'] <= mid.shift(1)) & (df['Close'] > mid.shift(1))
    short_sig = big.shift(1) & (df['High'] >= mid.shift(1)) & (df['Close'] < mid.shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 165
def ict_daily_bias_model(df, multi_data=None, sym=None, news_bias=0):
    ema5 = talib.EMA(df['Close'], 5); ema20 = talib.EMA(df['Close'], 20)
    bias_up = ema5 > ema20
    bias_dn = ema5 < ema20
    return np.where(bias_up, 1, np.where(bias_dn, -1, 0))

# 166
def ict_one_shot_one_kill(df, multi_data=None, sym=None, news_bias=0):
    day = getattr(df.index, 'date', pd.Series(index=df.index))
    confluence = (ict_fvg(df) != 0) & (ict_order_block(df) != 0)
    first = confluence & (day != pd.Series(day).shift(1).values)
    return np.where(first & (ict_fvg(df) > 0), 1,
                    np.where(first & (ict_fvg(df) < 0), -1, 0))

# 167
def asian_range_setup(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    asia = (hour >= 0) & (hour <= 6)
    asia_high = df['High'].where(asia).rolling(12).max().fillna(method='ffill')
    asia_low = df['Low'].where(asia).rolling(12).min().fillna(method='ffill')
    long_sig = (df['Close'] > asia_high)
    short_sig = (df['Close'] < asia_low)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 168
def asian_liquidity_raid(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    asia = hour.isin([0,1,2,3,4,5,6])
    raid_up = asia & (df['High'] >= df['High'].rolling(24).max()) & (df['Close'] < df['Open'])
    raid_dn = asia & (df['Low'] <= df['Low'].rolling(24).min()) & (df['Close'] > df['Open'])
    return np.where(raid_dn, 1, np.where(raid_up, -1, 0))

# 169
def midnight_open_price_setup(df, multi_data=None, sym=None, news_bias=0):
    day = getattr(df.index, 'date', pd.Series(index=df.index))
    first_price = df['Open'].where(day != pd.Series(day).shift(1).values).fillna(method='ffill')
    long_sig = df['Close'] > first_price
    short_sig = df['Close'] < first_price
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 170
def daily_high_low_liquidity_grab(df, multi_data=None, sym=None, news_bias=0):
    day = getattr(df.index, 'date', pd.Series(index=df.index))
    day_high = df['High'].groupby(day).transform('cummax')
    day_low = df['Low'].groupby(day).transform('cummin')
    grab_high = (df['High'] > day_high.shift(1)) & (df['Close'] < df['Open'])
    grab_low = (df['Low'] < day_low.shift(1)) & (df['Close'] > df['Open'])
    return np.where(grab_low, 1, np.where(grab_high, -1, 0))

# 171
def session_overlap_trap(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    overlap = hour.isin([12,13,14,15])
    false_break = ((df['Close'] > df['High'].shift(1)) & (df['Close'] < df['Open'])) | \
                  ((df['Close'] < df['Low'].shift(1)) & (df['Close'] > df['Open']))
    return np.where(overlap & false_break, -1, 0)

# 172
def ict_ote_retracement(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High'].rolling(30).max(); lo = df['Low'].rolling(30).min()
    up_trend = talib.EMA(df['Close'], 20) > talib.EMA(df['Close'], 50)
    ote_buy = up_trend & (df['Low'] <= (hi - 0.62*(hi-lo))) & (df['Low'] >= (hi - 0.79*(hi-lo)))
    dn_trend = ~up_trend
    ote_sell = dn_trend & (df['High'] >= (lo + 0.62*(hi-lo))) & (df['High'] <= (lo + 0.79*(hi-lo)))
    return np.where(ote_buy, 1, np.where(ote_sell, -1, 0))

# 173
def power_of_three_po3(df, multi_data=None, sym=None, news_bias=0):
    rng = (df['High'] - df['Low'])
    accumulate = rng.rolling(10).mean() < rng.rolling(50).mean() * 0.7
    manipulate = ict_liquidity_sweep(df) != 0
    distribute = talib.ADX(df['High'], df['Low'], df['Close'], 14) > 25
    score = accumulate.astype(int) + manipulate.astype(int) + distribute.astype(int)
    return np.where(score >= 2, np.sign(talib.LINEARREG_SLOPE(df['Close'], 10)), 0)

# 174
def amd_model(df, multi_data=None, sym=None, news_bias=0):
    day = getattr(df.index, 'date', pd.Series(index=df.index))
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    accum = hour.isin([0,1,2,3,4,5,6])
    manip = hour.isin([7,8,9,10])
    distrib = hour.isin([13,14,15,16])
    return np.where(manip, (ict_liquidity_sweep(df)), np.where(distrib, np.sign(df['Close'].diff()), 0))

# 175
def weekly_profile_setup(df, multi_data=None, sym=None, news_bias=0):
    week = getattr(df.index, 'week', pd.Series(index=df.index, dtype=int))
    wk_hi = df['High'].groupby(week).transform('cummax')
    wk_lo = df['Low'].groupby(week).transform('cummin')
    long_sig = (df['Close'] > wk_hi.shift(1))
    short_sig = (df['Close'] < wk_lo.shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 176
def liquidity_pool_targeting(df, multi_data=None, sym=None, news_bias=0):
    eqh = (df['High'].round(4) == df['High'].shift(1).round(4)) | (df['High'].round(4) == df['High'].shift(2).round(4))
    eql = (df['Low'].round(4) == df['Low'].shift(1).round(4)) | (df['Low'].round(4) == df['Low'].shift(2).round(4))
    aim_up = eql & (df['Close'] > df['Open'])
    aim_dn = eqh & (df['Close'] < df['Open'])
    return np.where(aim_up, 1, np.where(aim_dn, -1, 0))

# 177
def asia_range_break_retest(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    asia_mask = hour.isin([0,1,2,3,4,5,6])
    asia_hi = df['High'].where(asia_mask).rolling(12).max().fillna(method='ffill')
    asia_lo = df['Low'].where(asia_mask).rolling(12).min().fillna(method='ffill')
    break_up = df['Close'] > asia_hi
    break_dn = df['Close'] < asia_lo
    retest_up = (df['Low'] <= asia_hi) & break_up
    retest_dn = (df['High'] >= asia_lo) & break_dn
    return np.where(retest_up, 1, np.where(retest_dn, -1, 0))

# 178
def triple_top(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High']
    cond = (abs(hi - hi.shift(5)) / hi.shift(5) < 0.002) & (abs(hi - hi.shift(10)) / hi.shift(10) < 0.002)
    return np.where(cond & (df['Close'] < df['Open']), -1, 0)

# 179
def triple_bottom(df, multi_data=None, sym=None, news_bias=0):
    lo = df['Low']
    cond = (abs(lo - lo.shift(5)) / lo.shift(5) < 0.002) & (abs(lo - lo.shift(10)) / lo.shift(10) < 0.002)
    return np.where(cond & (df['Close'] > df['Open']), 1, 0)

# 180
def rounding_bottom_saucer(df, multi_data=None, sym=None, news_bias=0):
    lr = talib.LINEARREG_SLOPE(df['Close'], 20)
    curv_up = (lr.rolling(10).mean() > 0) & (df['Close'] > talib.SMA(df['Close'], 50))
    return np.where(curv_up, 1, 0)

# 181
def rounding_top(df, multi_data=None, sym=None, news_bias=0):
    lr = talib.LINEARREG_SLOPE(df['Close'], 20)
    curv_dn = (lr.rolling(10).mean() < 0) & (df['Close'] < talib.SMA(df['Close'], 50))
    return np.where(curv_dn, -1, 0)

# 182
def megaphone_broadening(df, multi_data=None, sym=None, news_bias=0):
    rng = (df['High'] - df['Low'])
    widening = rng > rng.rolling(20).mean() * 1.5
    fade = widening & (df['Close'].diff() * (df['Close'] - df['Close'].shift(1)) < 0)
    return np.where(fade, -np.sign(df['Close'].diff()), 0)

# 183
def rectangle_range_breakout(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High'].rolling(30).max(); lo = df['Low'].rolling(30).min()
    return np.where(df['Close'] > hi, 1, np.where(df['Close'] < lo, -1, 0))

# 184
def diamond_top(df, multi_data=None, sym=None, news_bias=0):
    rng = (df['High'] - df['Low'])
    expand = rng.rolling(10).mean() > rng.rolling(30).mean()
    contract = rng.rolling(5).mean() < rng.rolling(15).mean()
    brk = df['Close'] < df['Low'].rolling(10).min().shift(1)
    return np.where(expand & contract & brk, -1, 0)

# 185
def diamond_bottom(df, multi_data=None, sym=None, news_bias=0):
    rng = (df['High'] - df['Low'])
    expand = rng.rolling(10).mean() > rng.rolling(30).mean()
    contract = rng.rolling(5).mean() < rng.rolling(15).mean()
    brk = df['Close'] > df['High'].rolling(10).max().shift(1)
    return np.where(expand & contract & brk, 1, 0)

# 186
def quasimodo_over_under(df, multi_data=None, sym=None, news_bias=0):
    hh = df['High'] > df['High'].shift(1)
    ll = df['Low'] < df['Low'].shift(1)
    long_sig = ll & (df['Close'] > df['Open']) & (df['Close'] > df['High'].shift(2))
    short_sig = hh & (df['Close'] < df['Open']) & (df['Close'] < df['Low'].shift(2))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 187
def slingshot_entry(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High'].rolling(20).max(); lo = df['Low'].rolling(20).min()
    short_sig = (df['High'] > hi) & (df['Close'] < hi)
    long_sig = (df['Low'] < lo) & (df['Close'] > lo)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 188
def channel_fakeout(df, multi_data=None, sym=None, news_bias=0):
    upper = df['High'].rolling(30).max(); lower = df['Low'].rolling(30).min()
    fake = ((df['High'] > upper) & (df['Close'] < upper)) | ((df['Low'] < lower) & (df['Close'] > lower))
    return np.where(fake & (df['Close'] > df['Open']), 1, np.where(fake & (df['Close'] < df['Open']), -1, 0))

# 189
def overlapping_consolidation_break(df, multi_data=None, sym=None, news_bias=0):
    rng = (df['High'] - df['Low'])
    tight = rng < 0.7 * rng.rolling(20).mean()
    stack = tight & tight.shift(1) & tight.shift(2)
    brk_up = stack & (df['Close'] > df['High'].rolling(5).max())
    brk_dn = stack & (df['Close'] < df['Low'].rolling(5).min())
    return np.where(brk_up, 1, np.where(brk_dn, -1, 0))

# 190
def impulse_correction_continuation(df, multi_data=None, sym=None, news_bias=0):
    ema20 = talib.EMA(df['Close'], 20)
    impulse = abs(df['Close'] - df['Open']) > (df['High'] - df['Low']).rolling(20).mean()
    pullback = (df['Close'] < ema20) & (df['Close'] > df['Low'].rolling(5).min())
    cont_up = impulse.shift(3) & pullback & (df['Close'] > ema20)
    cont_dn = impulse.shift(3) & ~pullback & (df['Close'] < ema20)
    return np.where(cont_up, 1, np.where(cont_dn, -1, 0))

# 191
def bar_by_bar_momentum_trap(df, multi_data=None, sym=None, news_bias=0):
    strong = (df['Close'] - df['Open']).abs() > (df['High'] - df['Low']).rolling(10).mean()
    trap = strong & (df['Close'].diff() * (df['Close'] - df['Open']) < 0)
    return np.where(trap, -np.sign(df['Close'] - df['Open']), 0)

# 192
def gap_window_fill(df, multi_data=None, sym=None, news_bias=0):
    gap = df['Open'] - df['Close'].shift(1)
    long_sig = (gap < 0) & (df['Close'] > df['Open'])
    short_sig = (gap > 0) & (df['Close'] < df['Open'])
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 193
def failed_breakout_fb(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High'].rolling(20).max(); lo = df['Low'].rolling(20).min()
    f_up = (df['High'] > hi) & (df['Close'] < hi)
    f_dn = (df['Low'] < lo) & (df['Close'] > lo)
    return np.where(f_dn, 1, np.where(f_up, -1, 0))

# 194
def pre_asia_session_fade(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    mask = hour.isin([22,23])
    fade = mask & (df['Close'] < df['Open']) & (df['Close'] > df['Close'].shift(1))
    return np.where(fade, 1, 0)

# 195
def london_close_reversal(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    mask = hour.isin([16,17])
    rev = mask & (df['Close'].diff().rolling(3).sum() * df['Close'].diff() < 0)
    return np.where(rev, -np.sign(df['Close'].diff()), 0)

# 196
def friday_close_profit_taking(df, multi_data=None, sym=None, news_bias=0):
    dow = df.index.dayofweek if hasattr(df.index, 'dayofweek') else pd.Series(index=df.index)
    near_close = (getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int)) >= 19)
    sig = (dow == 4) & near_close & (df['Close'] < df['Open'])
    return np.where(sig, -1, 0)

# 197
def monday_gap_fill(df, multi_data=None, sym=None, news_bias=0):
    dow = df.index.dayofweek if hasattr(df.index, 'dayofweek') else pd.Series(index=df.index)
    gap = df['Open'] - df['Close'].shift(1)
    long_sig = (dow == 0) & (gap > 0)
    short_sig = (dow == 0) & (gap < 0)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 198
def end_of_month_rebalancing(df, multi_data=None, sym=None, news_bias=0):
    day = df.index.day if hasattr(df.index, 'day') else pd.Series(index=df.index)
    eom = (day >= 28)
    return np.where(eom & (df['Close'] < df['Open']), -1, np.where(eom & (df['Close'] > df['Open']), 1, 0))

# 199
def first_15min_candle_trap(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    minute = getattr(df.index, 'minute', pd.Series(index=df.index, dtype=int))
    open_block = (minute < 15)
    spike = open_block & ((df['High'] - df['Low']) > (df['High'] - df['Low']).rolling(20).mean() * 1.5)
    fade = spike & (df['Close'] < df['Open'])
    inv_fade = spike & (df['Close'] > df['Open'])
    return np.where(inv_fade, 1, np.where(fade, -1, 0))

# 200
def futures_rollover_liquidity_spike(df, multi_data=None, sym=None, news_bias=0):
    spike = (df.index.to_series().diff().dt.days.fillna(0) > 3)
    big_vol = df['Volume'] > df['Volume'].rolling(30).mean() * 1.5
    return np.where(spike & big_vol & (df['Close'] > df['Open']), 1,
                    np.where(spike & big_vol & (df['Close'] < df['Open']), -1, 0))

# 201
def supertrend_strategy(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 10)
    basis = (df['High'] + df['Low']) / 2
    up = basis - 3 * atr
    dn = basis + 3 * atr
    trend_up = df['Close'] > dn
    trend_dn = df['Close'] < up
    return np.where(trend_up, 1, np.where(trend_dn, -1, 0))

# 202
def hull_moving_average_trend(df, multi_data=None, sym=None, news_bias=0):
    wma9 = talib.WMA(df['Close'], 9); wma18 = talib.WMA(df['Close'], 18)
    hma = talib.WMA(2*wma9 - wma18, 9)
    return np.where(df['Close'] > hma, 1, np.where(df['Close'] < hma, -1, 0))

# 203
def ema_ribbon_entry_exit(df, multi_data=None, sym=None, news_bias=0):
    emas = [talib.EMA(df['Close'], p) for p in [8,13,21,34,55]]
    bull = all(emas[i] > emas[i+1] for i in range(len(emas)-1))
    bear = all(emas[i] < emas[i+1] for i in range(len(emas)-1))
    return np.where(bull, 1, np.where(bear, -1, 0))

# 204
def traders_dynamic_index_tdi(df, multi_data=None, sym=None, news_bias=0):
    rsi = talib.RSI(df['Close'], 13)
    sig = talib.SMA(rsi, 7)
    band_hi = sig + talib.STDDEV(rsi, 34)
    band_lo = sig - talib.STDDEV(rsi, 34)
    return np.where((rsi > band_hi), -1, np.where((rsi < band_lo), 1, 0))

# 205
def qqe_mod(df, multi_data=None, sym=None, news_bias=0):
    rsi = talib.RSI(df['Close'], 14)
    smooth = talib.EMA(rsi, 5)
    sig = talib.EMA(smooth, 5)
    return np.where(smooth > sig, 1, np.where(smooth < sig, -1, 0))

# 206
def squeeze_momentum_indicator(df, multi_data=None, sym=None, news_bias=0):
    bb_up, bb_mid, bb_lo = talib.BBANDS(df['Close'], timeperiod=20)
    tr = talib.ATR(df['High'], df['Low'], df['Close'], 20)
    kc_up = bb_mid + 1.5 * tr
    kc_lo = bb_mid - 1.5 * tr
    squeeze_on = (bb_up - bb_lo) < (kc_up - kc_lo)
    breakout = df['Close'] > df['High'].rolling(5).max()
    breakdn = df['Close'] < df['Low'].rolling(5).min()
    return np.where(squeeze_on & breakout, 1, np.where(squeeze_on & breakdn, -1, 0))

# 207
def chandelier_exit(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 22)
    long_stop = df['High'].rolling(22).max() - 3*atr
    short_stop = df['Low'].rolling(22).min() + 3*atr
    return np.where(df['Close'] > long_stop, 1, np.where(df['Close'] < short_stop, -1, 0))

# 208
def awesome_oscillator_zero_line(df, multi_data=None, sym=None, news_bias=0):
    ao = talib.SMA((df['High']+df['Low'])/2, 5) - talib.SMA((df['High']+df['Low'])/2, 34)
    return np.where(ao > 0, 1, np.where(ao < 0, -1, 0))

# 209
def williams_r_overbought_oversold(df, multi_data=None, sym=None, news_bias=0):
    w = talib.WILLR(df['High'], df['Low'], df['Close'], 14)
    return np.where(w < -80, 1, np.where(w > -20, -1, 0))

# 210
def chaikin_money_flow_breakout(df, multi_data=None, sym=None, news_bias=0):
    cmf = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
    return np.where(cmf > cmf.rolling(20).mean(), 1, -1)

# 211
def nfp_breakout(df, multi_data=None, sym=None, news_bias=0):
    return np.where(news_bias > 0.5, 1, np.where(news_bias < -0.5, -1, 0))

# 212
def cpi_inflation_spike(df, multi_data=None, sym=None, news_bias=0):
    return np.where(abs(news_bias) > 0.7, np.sign(news_bias), 0)

# 213
def fomc_rate_decision_whipsaw(df, multi_data=None, sym=None, news_bias=0):
    vol = (df['High'] - df['Low'])
    spike = vol > 1.8 * vol.rolling(20).mean()
    return np.where(spike & (news_bias > 0), 1, np.where(spike & (news_bias < 0), -1, 0))

# 214
def earnings_gap_play(df, multi_data=None, sym=None, news_bias=0):
    gap = df['Open'] - df['Close'].shift(1)
    follow = np.where(gap > 0, (df['Close'] > df['Open']), (df['Close'] < df['Open']))
    return np.where(follow, np.sign(gap), 0)

# 215
def dividend_announcement_momentum(df, multi_data=None, sym=None, news_bias=0):
    return np.where(news_bias > 0.3, 1, np.where(news_bias < -0.3, -1, 0))

# 216
def scalping_1min_ema_cross(df, multi_data=None, sym=None, news_bias=0):
    ema8 = talib.EMA(df['Close'], 8); ema21 = talib.EMA(df['Close'], 21)
    cross_up = (ema8 > ema21) & (ema8.shift(1) <= ema21.shift(1))
    cross_dn = (ema8 < ema21) & (ema8.shift(1) >= ema21.shift(1))
    return np.where(cross_up, 1, np.where(cross_dn, -1, 0))

# 217
def vwap_fade_extremes(df, multi_data=None, sym=None, news_bias=0):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    dist = (df['Close'] - vwap) / vwap
    return np.where(dist < -0.01, 1, np.where(dist > 0.01, -1, 0))

# 218
def pre_market_gap_fill(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    premkt = hour.isin([4,5,6,7,8])
    gap = df['Open'] - df['Close'].shift(1)
    fade = (gap > 0) & premkt & (df['Close'] < df['Open'])
    inv = (gap < 0) & premkt & (df['Close'] > df['Open'])
    return np.where(inv, 1, np.where(fade, -1, 0))

# 219
def london_breakout(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    asia_mask = hour.isin([0,1,2,3,4,5,6])
    asia_hi = df['High'].where(asia_mask).rolling(12).max().fillna(method='ffill')
    asia_lo = df['Low'].where(asia_mask).rolling(12).min().fillna(method='ffill')
    long_sig = (df['Close'] > asia_hi) & hour.isin([7,8,9,10])
    short_sig = (df['Close'] < asia_lo) & hour.isin([7,8,9,10])
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 220
def newyork_open_fade(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    open_block = hour.isin([13,14])
    fade = open_block & (df['Close'] < df['Open']) & (df['Close'] > df['Close'].shift(1))
    inv = open_block & (df['Close'] > df['Open']) & (df['Close'] < df['Close'].shift(1))
    return np.where(inv, -1, np.where(fade, 1, 0))

# 221
def opening_range_breakout_orb(df, multi_data=None, sym=None, news_bias=0):
    minute = getattr(df.index, 'minute', pd.Series(index=df.index, dtype=int))
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    first_window = (minute < 30) & (hour == hour.iloc[0] if len(hour) else False)
    or_hi = df['High'].where(first_window).cummax().fillna(method='ffill')
    or_lo = df['Low'].where(first_window).cummin().fillna(method='ffill')
    return np.where(df['Close'] > or_hi, 1, np.where(df['Close'] < or_lo, -1, 0))

# 222
def spoof_order_fade(df, multi_data=None, sym=None, news_bias=0):
    vol_spike = df['Volume'] > 2.5 * df['Volume'].rolling(20).mean()
    wick_up = (df['High'] - np.maximum(df['Open'], df['Close'])) > (df['High'] - df['Low']) * 0.6
    wick_dn = (np.minimum(df['Open'], df['Close']) - df['Low']) > (df['High'] - df['Low']) * 0.6
    short_sig = vol_spike & wick_up & (df['Close'] < df['Open'])
    long_sig = vol_spike & wick_dn & (df['Close'] > df['Open'])
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 223
def ema8_21_rsi_divergence_combo(df, multi_data=None, sym=None, news_bias=0):
    ema8 = talib.EMA(df['Close'], 8); ema21 = talib.EMA(df['Close'], 21)
    rsi = talib.RSI(df['Close'], 14)
    bull = (ema8 > ema21) & (df['Close'] < df['Close'].shift(5)) & (rsi > rsi.shift(5))
    bear = (ema8 < ema21) & (df['Close'] > df['Close'].shift(5)) & (rsi < rsi.shift(5))
    return np.where(bull, 1, np.where(bear, -1, 0))

# 224
def bbands_stochastic_cross(df, multi_data=None, sym=None, news_bias=0):
    up, mid, lo = talib.BBANDS(df['Close'], 20)
    k, d = talib.STOCH(df['High'], df['Low'], df['Close'])
    long_sig = (df['Close'] <= lo) & (k > d)
    short_sig = (df['Close'] >= up) & (k < d)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 225
def macd_ema_trend_filter(df, multi_data=None, sym=None, news_bias=0):
    macd, signal, _ = talib.MACD(df['Close'])
    ema50 = talib.EMA(df['Close'], 50)
    long_sig = (macd > signal) & (df['Close'] > ema50)
    short_sig = (macd < signal) & (df['Close'] < ema50)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 226
def atr_stop_hunt_filter(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    stop_run_up = (df['High'] - df['Close']) > 1.2 * atr
    stop_run_dn = (df['Close'] - df['Low']) > 1.2 * atr
    return np.where(stop_run_dn, 1, np.where(stop_run_up, -1, 0))

# 227
def pivot_points_vwap_confluence(df, multi_data=None, sym=None, news_bias=0):
    pivot = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    long_sig = (df['Close'] > pivot) & (df['Close'] > vwap)
    short_sig = (df['Close'] < pivot) & (df['Close'] < vwap)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 228
def elder_ray_index(df, multi_data=None, sym=None, news_bias=0):
    ema13 = talib.EMA(df['Close'], 13)
    bull_power = df['High'] - ema13
    bear_power = df['Low'] - ema13
    long_sig = (bull_power > 0) & (bull_power.shift(1) <= 0)
    short_sig = (bear_power < 0) & (bear_power.shift(1) >= 0)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 229
def coppock_curve_turn(df, multi_data=None, sym=None, news_bias=0):
    roc14 = talib.ROC(df['Close'], 14)
    roc11 = talib.ROC(df['Close'], 11)
    cc_raw = roc14 + roc11
    coppock = talib.WMA(cc_raw, 10)
    turn_up = (coppock > 0) & (coppock.diff() > 0) & (coppock.shift(1) <= 0)
    turn_dn = (coppock < 0) & (coppock.diff() < 0) & (coppock.shift(1) >= 0)
    return np.where(turn_up, 1, np.where(turn_dn, -1, 0))

# 230
def vortex_cross(df, multi_data=None, sym=None, news_bias=0):
    tr = talib.TRANGE(df['High'], df['Low'], df['Close'])
    period = 14
    vm_plus = (df['High'] - df['Low'].shift(1)).abs()
    vm_minus = (df['Low'] - df['High'].shift(1)).abs()
    vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
    cross_up = (vi_plus > vi_minus) & (vi_plus.shift(1) <= vi_minus.shift(1))
    cross_dn = (vi_plus < vi_minus) & (vi_plus.shift(1) >= vi_minus.shift(1))
    return np.where(cross_up, 1, np.where(cross_dn, -1, 0))

# 231
def aroon_trend(df, multi_data=None, sym=None, news_bias=0):
    aroon_down, aroon_up = talib.AROON(df['High'], df['Low'], timeperiod=25)
    return np.where(aroon_up > aroon_down, 1, np.where(aroon_up < aroon_down, -1, 0))

# 232
def trix_signal_cross(df, multi_data=None, sym=None, news_bias=0):
    trix = talib.TRIX(df['Close'], timeperiod=15)
    sig = talib.SMA(trix, 9)
    long_sig = (trix > sig) & (trix.shift(1) <= sig.shift(1))
    short_sig = (trix < sig) & (trix.shift(1) >= sig.shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 233
def tema_crossover(df, multi_data=None, sym=None, news_bias=0):
    fast = talib.TEMA(df['Close'], timeperiod=10)
    slow = talib.TEMA(df['Close'], timeperiod=30)
    return np.where(fast > slow, 1, np.where(fast < slow, -1, 0))

# 234
def dema_crossover(df, multi_data=None, sym=None, news_bias=0):
    fast = talib.DEMA(df['Close'], timeperiod=10)
    slow = talib.DEMA(df['Close'], timeperiod=30)
    return np.where(fast > slow, 1, np.where(fast < slow, -1, 0))

# 235
def kama_trend_follow(df, multi_data=None, sym=None, news_bias=0):
    kama = talib.KAMA(df['Close'], timeperiod=30)
    return np.where(df['Close'] > kama, 1, np.where(df['Close'] < kama, -1, 0))

# 236
def dpo_mean_reversion(df, multi_data=None, sym=None, news_bias=0):
    n = 20
    sma = talib.SMA(df['Close'], n)
    dpo = df['Close'] - sma.shift(int(n/2)+1)
    thresh = dpo.rolling(50).std()
    long_sig = dpo < -thresh
    short_sig = dpo > thresh
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 237
def obv_breakout_reversal(df, multi_data=None, sym=None, news_bias=0):
    obv = talib.OBV(df['Close'], df['Volume'])
    obv_max = obv.rolling(20).max()
    obv_min = obv.rolling(20).min()
    long_sig = (obv > obv_max.shift(1)) & (df['Close'] > df['Close'].rolling(5).mean())
    short_sig = (obv < obv_min.shift(1)) & (df['Close'] < df['Close'].rolling(5).mean())
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 238
def ad_line_slope(df, multi_data=None, sym=None, news_bias=0):
    ad = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    slope = talib.LINEARREG_SLOPE(ad, 14)
    return np.where(slope > 0, 1, np.where(slope < 0, -1, 0))

# 239
def pvi_nvi_trend_combo(df, multi_data=None, sym=None, news_bias=0):
    close = df['Close']
    vol = df['Volume']
    ret = close.pct_change().fillna(0)
    pvi = pd.Series(index=close.index, dtype=float).fillna(0); pvi.iloc[0] = 1000
    nvi = pd.Series(index=close.index, dtype=float).fillna(0); nvi.iloc[0] = 1000
    for i in range(1, len(df)):
        pvi.iloc[i] = pvi.iloc[i-1]*(1+ret.iloc[i]) if vol.iloc[i] > vol.iloc[i-1] else pvi.iloc[i-1]
        nvi.iloc[i] = nvi.iloc[i-1]*(1+ret.iloc[i]) if vol.iloc[i] < vol.iloc[i-1] else nvi.iloc[i-1]
    pvi_ma = pvi.rolling(50).mean(); nvi_ma = nvi.rolling(50).mean()
    long_sig = (pvi > pvi_ma) & (nvi > nvi_ma)
    short_sig = (pvi < pvi_ma) & (nvi < nvi_ma)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 240
def price_volume_trend_cross(df, multi_data=None, sym=None, news_bias=0):
    pvt = (df['Close'].pct_change().fillna(0) * df['Volume']).cumsum()
    pvt_ma = talib.SMA(pvt, 20)
    return np.where(pvt > pvt_ma, 1, np.where(pvt < pvt_ma, -1, 0))

# 241
def mfi_extremes(df, multi_data=None, sym=None, news_bias=0):
    mfi = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    return np.where(mfi < 20, 1, np.where(mfi > 80, -1, 0))

# 242
def vortex_squeeze_break(df, multi_data=None, sym=None, news_bias=0):
    tr = talib.TRANGE(df['High'], df['Low'], df['Close'])
    p = 14
    vm_plus = (df['High'] - df['Low'].shift(1)).abs()
    vm_minus = (df['Low'] - df['High'].shift(1)).abs()
    vi_plus = vm_plus.rolling(p).sum() / tr.rolling(p).sum()
    vi_minus = vm_minus.rolling(p).sum() / tr.rolling(p).sum()
    spread = (vi_plus - vi_minus).abs()
    squeeze = spread < spread.rolling(20).mean() * 0.7
    brk_up = (~squeeze) & (df['Close'] > df['High'].rolling(5).max())
    brk_dn = (~squeeze) & (df['Close'] < df['Low'].rolling(5).min())
    return np.where(brk_up, 1, np.where(brk_dn, -1, 0))

# 243
def heikin_ashi_trend_flip(df, multi_data=None, sym=None, news_bias=0):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    long_sig = (ha_close > ha_open) & (ha_close.shift(1) <= ha_open.shift(1))
    short_sig = (ha_close < ha_open) & (ha_close.shift(1) >= ha_open.shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 244
def fractal_breakout(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High']; lo = df['Low']
    fractal_up = (hi.shift(2) < hi.shift(1)) & (hi.shift(1) > hi) & (hi.shift(1) > hi.shift(3)) & (hi.shift(1) > hi.shift(4))
    fractal_dn = (lo.shift(2) > lo.shift(1)) & (lo.shift(1) < lo) & (lo.shift(1) < lo.shift(3)) & (lo.shift(1) < lo.shift(4))
    up_lvl = hi.shift(1).where(fractal_up).ffill()
    dn_lvl = lo.shift(1).where(fractal_dn).ffill()
    long_sig = df['Close'] > up_lvl
    short_sig = df['Close'] < dn_lvl
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 245
def renko_proxy_trend(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    brick = atr
    up_move = (df['Close'] - df['Close'].shift(1)) > brick
    dn_move = (df['Close'].shift(1) - df['Close']) > brick
    trend_up = up_move.rolling(3).sum() >= 2
    trend_dn = dn_move.rolling(3).sum() >= 2
    return np.where(trend_up, 1, np.where(trend_dn, -1, 0))

# 246
def alligator_gator_proxy(df, multi_data=None, sym=None, news_bias=0):
    med = (df['High'] + df['Low']) / 2
    jaw = talib.EMA(med, 13).shift(8)
    teeth = talib.EMA(med, 8).shift(5)
    lips = talib.EMA(med, 5).shift(3)
    bull = (lips > teeth) & (teeth > jaw)
    bear = (lips < teeth) & (teeth < jaw)
    return np.where(bull, 1, np.where(bear, -1, 0))

# 247
def ad_divergence_filter(df, multi_data=None, sym=None, news_bias=0):
    ad = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    price_up = df['Close'] > df['Close'].shift(5)
    ad_dn = ad < ad.shift(5)
    price_dn = df['Close'] < df['Close'].shift(5)
    ad_up = ad > ad.shift(5)
    long_sig = price_dn & ad_up
    short_sig = price_up & ad_dn
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 248
def cci_zero_cross_trend(df, multi_data=None, sym=None, news_bias=0):
    cci = talib.CCI(df['High'], df['Low'], df['Close'], 20)
    long_sig = (cci > 0) & (cci.shift(1) <= 0)
    short_sig = (cci < 0) & (cci.shift(1) >= 0)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 249
def donchian_pullback_entry(df, multi_data=None, sym=None, news_bias=0):
    upper = df['High'].rolling(20).max()
    lower = df['Low'].rolling(20).min()
    mid = (upper + lower) / 2
    breakout_up = df['Close'] > upper.shift(1)
    breakout_dn = df['Close'] < lower.shift(1)
    long_sig = breakout_up & (df['Close'] <= mid)
    short_sig = breakout_dn & (df['Close'] >= mid)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 250
def bollinger_band_walk(df, multi_data=None, sym=None, news_bias=0):
    up, mid, lo = talib.BBANDS(df['Close'], 20, 2, 2)
    near_up = df['Close'] > mid
    streak = near_up.rolling(5).sum() >= 4
    exit_cond = df['Close'] < mid
    sig = np.where(streak, 1, 0)
    sig = np.where(exit_cond, -1, sig)
    return sig

# 251
def bb_keltner_confluence_break(df, multi_data=None, sym=None, news_bias=0):
    bb_up, bb_mid, bb_lo = talib.BBANDS(df['Close'], 20, 2, 2)
    tr = talib.ATR(df['High'], df['Low'], df['Close'], 20)
    kc_up = bb_mid + 1.5 * tr
    kc_lo = bb_mid - 1.5 * tr
    squeeze_on = (bb_up - bb_lo) < (kc_up - kc_lo)
    ema = talib.EMA(df['Close'], 50)
    long_sig = squeeze_on & (df['Close'] > bb_up) & (df['Close'] > ema)
    short_sig = squeeze_on & (df['Close'] < bb_lo) & (df['Close'] < ema)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 252
def supertrend_fast(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 7)
    basis = (df['High'] + df['Low']) / 2
    up = basis - 2 * atr
    dn = basis + 2 * atr
    trend_up = df['Close'] > dn
    trend_dn = df['Close'] < up
    return np.where(trend_up, 1, np.where(trend_dn, -1, 0))

# 253
def chandelier_exit_contra(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 22)
    long_stop = df['High'].rolling(22).max() - 3*atr
    short_stop = df['Low'].rolling(22).min() + 3*atr
    long_sig = df['Close'] < long_stop
    short_sig = df['Close'] > short_stop
    return np.where(short_sig, -1, np.where(long_sig, 1, 0))

# 254
def ema_9_21_55_stack(df, multi_data=None, sym=None, news_bias=0):
    e9 = talib.EMA(df['Close'], 9)
    e21 = talib.EMA(df['Close'], 21)
    e55 = talib.EMA(df['Close'], 55)
    bull = (e9 > e21) & (e21 > e55)
    bear = (e9 < e21) & (e21 < e55)
    return np.where(bull, 1, np.where(bear, -1, 0))

# 255
def rsi_signal_ma_cross(df, multi_data=None, sym=None, news_bias=0):
    rsi = talib.RSI(df['Close'], 14)
    rma = talib.SMA(rsi, 9)
    long_sig = (rsi > rma) & (rsi.shift(1) <= rma.shift(1))
    short_sig = (rsi < rma) & (rsi.shift(1) >= rma.shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 256
def stochastic_rsi_cross(df, multi_data=None, sym=None, news_bias=0):
    try:
        fastk, fastd = talib.STOCHRSI(df['Close'], timeperiod=14, fastk_period=5, fastd_period=5, fastd_matype=0)
        long_sig = (fastk > fastd) & (fastk.shift(1) <= fastd.shift(1)) & (fastd < 20)
        short_sig = (fastk < fastd) & (fastk.shift(1) >= fastd.shift(1)) & (fastd > 80)
        return np.where(long_sig, 1, np.where(short_sig, -1, 0))
    except Exception:
        return np.zeros(len(df))

# 257
def macd_hist_zero_cross(df, multi_data=None, sym=None, news_bias=0):
    _, _, hist = talib.MACD(df['Close'])
    long_sig = (hist > 0) & (hist.shift(1) <= 0)
    short_sig = (hist < 0) & (hist.shift(1) >= 0)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 258
def adx_di_cross_filter(df, multi_data=None, sym=None, news_bias=0):
    adx = talib.ADX(df['High'], df['Low'], df['Close'], 14)
    plus_di = talib.PLUS_DI(df['High'], df['Low'], df['Close'], 14)
    minus_di = talib.MINUS_DI(df['High'], df['Low'], df['Close'], 14)
    long_sig = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)) & (adx > 20)
    short_sig = (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1)) & (adx > 20)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 259
def ultimate_osc_extremes(df, multi_data=None, sym=None, news_bias=0):
    ult = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    return np.where(ult < 30, 1, np.where(ult > 70, -1, 0))

# 260
def rsi_bb_confluence(df, multi_data=None, sym=None, news_bias=0):
    rsi = talib.RSI(df['Close'], 14)
    up, mid, lo = talib.BBANDS(df['Close'], 20, 2, 2)
    long_sig = (df['Close'] <= lo) & (rsi < 30)
    short_sig = (df['Close'] >= up) & (rsi > 70)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 261
def kama_pullback_entry(df, multi_data=None, sym=None, news_bias=0):
    kama = talib.KAMA(df['Close'], 20)
    pullback_long = (df['Close'] > kama) & (df['Low'] <= kama) & (df['Close'] > df['Open'])
    pullback_short = (df['Close'] < kama) & (df['High'] >= kama) & (df['Close'] < df['Open'])
    return np.where(pullback_long, 1, np.where(pullback_short, -1, 0))

# 262
def ema_slope_break(df, multi_data=None, sym=None, news_bias=0):
    e50 = talib.EMA(df['Close'], 50)
    slope = talib.LINEARREG_SLOPE(e50, 10)
    long_sig = (df['Close'] > e50) & (slope > 0)
    short_sig = (df['Close'] < e50) & (slope < 0)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 263
def ichimoku_cloud_break(df, multi_data=None, sym=None, news_bias=0):
    nine_high = df['High'].rolling(9).max()
    nine_low = df['Low'].rolling(9).min()
    tenkan = (nine_high + nine_low) / 2
    per26_high = df['High'].rolling(26).max()
    per26_low = df['Low'].rolling(26).min()
    kijun = (per26_high + per26_low) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    cloud_top = np.maximum(span_a, span_b)
    cloud_bot = np.minimum(span_a, span_b)
    long_sig = df['Close'] > cloud_top
    short_sig = df['Close'] < cloud_bot
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 264
def rsi_divergence_simple(df, multi_data=None, sym=None, news_bias=0):
    rsi = talib.RSI(df['Close'], 14)
    price_diff = df['Close'] - df['Close'].shift(10)
    rsi_diff = rsi - rsi.shift(10)
    long_sig = (price_diff < 0) & (rsi_diff > 0)
    short_sig = (price_diff > 0) & (rsi_diff < 0)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 265
def ema_ribbon_pullback(df, multi_data=None, sym=None, news_bias=0):
    emas = [talib.EMA(df['Close'], p) for p in [8, 13, 21, 34, 55]]
    bull_ribbon = (emas[0] > emas[1]) & (emas[1] > emas[2]) & (emas[2] > emas[3]) & (emas[3] > emas[4])
    bear_ribbon = (emas[0] < emas[1]) & (emas[1] < emas[2]) & (emas[2] < emas[3]) & (emas[3] < emas[4])
    pullback_long = bull_ribbon & (df['Low'] <= emas[2]) & (df['Close'] > df['Open'])
    pullback_short = bear_ribbon & (df['High'] >= emas[2]) & (df['Close'] < df['Open'])
    return np.where(pullback_long, 1, np.where(pullback_short, -1, 0))

# 266
def frankfurt_fakeout(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    frankfurt = hour.isin([6,7])
    spike = (df['High'] - df['Low']) > 1.5 * (df['High'] - df['Low']).rolling(20).mean()
    fade_dn = frankfurt & spike & (df['Close'] < df['Open'])
    fade_up = frankfurt & spike & (df['Close'] > df['Open'])
    return np.where(fade_up, 1, np.where(fade_dn, -1, 0))

# 267
def midweek_reversal(df, multi_data=None, sym=None, news_bias=0):
    if not hasattr(df.index, 'dayofweek'):
        return np.zeros(len(df))
    dow = df.index.dayofweek
    large_move = (df['Close'] - df['Open']).abs() > (df['High'] - df['Low']).rolling(10).mean()
    long_sig = (dow == 2) & large_move & (df['Close'] < df['Open'])
    short_sig = (dow == 2) & large_move & (df['Close'] > df['Open'])
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 268
def liquidity_void_entry(df, multi_data=None, sym=None, news_bias=0):
    rng = (df['High'] - df['Low'])
    big_impulse = rng > 1.8 * rng.rolling(20).mean()
    void_top = df['High'].shift(1).where(big_impulse.shift(1)).ffill()
    void_bot = df['Low'].shift(1).where(big_impulse.shift(1)).ffill()
    mid = (void_top + void_bot) / 2
    long_sig = (df['Low'] <= mid) & (df['Close'] > mid)
    short_sig = (df['High'] >= mid) & (df['Close'] < mid)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 269
def double_tap_liquidity_raid(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High']; lo = df['Low']
    near = 0.0005
    eq_hi = (abs(hi - hi.shift(5)) / hi.shift(5) < near)
    eq_lo = (abs(lo - lo.shift(5)) / lo.shift(5) < near)
    raid_up = eq_hi & (df['Close'] < df['Open'])
    raid_dn = eq_lo & (df['Close'] > df['Open'])
    return np.where(raid_dn, 1, np.where(raid_up, -1, 0))

# 270
def displacement_fvg_retest(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    disp_up = (df['Close'] - df['Open']) > 1.5 * atr
    disp_dn = (df['Open'] - df['Close']) > 1.5 * atr
    gap_up = df['Low'].shift(1) > df['High'].shift(2)
    gap_dn = df['High'].shift(1) < df['Low'].shift(2)
    mid_up = (df['High'].shift(2) + df['Low'].shift(1)) / 2
    mid_dn = (df['Low'].shift(2) + df['High'].shift(1)) / 2
    long_sig = disp_up.shift(1) & gap_up.shift(1) & (df['Low'] <= mid_up) & (df['Close'] > mid_up)
    short_sig = disp_dn.shift(1) & gap_dn.shift(1) & (df['High'] >= mid_dn) & (df['Close'] < mid_dn)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 271
def institutional_candle_trap(df, multi_data=None, sym=None, news_bias=0):
    rng = (df['High'] - df['Low'])
    huge = rng > 2.0 * rng.rolling(20).mean()
    trap_dn = huge & (df['Close'] < df['Open']) & (df['Close'] > df['Low'] + 0.6*rng)
    trap_up = huge & (df['Close'] > df['Open']) & (df['Close'] < df['High'] - 0.6*rng)
    return np.where(trap_dn, 1, np.where(trap_up, -1, 0))

# 272
def ipda_3day_dealing_range(df, multi_data=None, sym=None, news_bias=0):
    day = getattr(df.index, 'date', pd.Series(index=df.index))
    day_id = pd.Series(day).factorize()[0]
    hi3 = df['High'].rolling(3*24 if hasattr(df.index, 'hour') else 3).max()
    lo3 = df['Low'].rolling(3*24 if hasattr(df.index, 'hour') else 3).min()
    long_sig = (df['Low'] <= lo3) & (df['Close'] > df['Open'])
    short_sig = (df['High'] >= hi3) & (df['Close'] < df['Open'])
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 273
def daily_range_expansion_model(df, multi_data=None, sym=None, news_bias=0):
    day = getattr(df.index, 'date', pd.Series(index=df.index))
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    first_half = hour < 12
    dr = (df['High'].groupby(day).transform('max') - df['Low'].groupby(day).transform('min'))
    early_vol = (df['High'] - df['Low']).where(first_half).rolling(12).sum()
    exp = (df['High'] - df['Low']) > (dr.rolling(24).mean())
    long_sig = first_half & exp & (df['Close'] > df['Open']) & (early_vol > early_vol.rolling(5).mean())
    short_sig = first_half & exp & (df['Close'] < df['Open']) & (early_vol > early_vol.rolling(5).mean())
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 274
def quarter_theory_levels_reversals(df, multi_data=None, sym=None, news_bias=0):
    price = df['Close']
    q = np.floor(price) + 0.25
    h = np.floor(price) + 0.50
    tq = np.floor(price) + 0.75
    near = (abs(price - q) < (df['High'] - df['Low']).rolling(20).mean()*0.25) | \
           (abs(price - h) < (df['High'] - df['Low']).rolling(20).mean()*0.25) | \
           (abs(price - tq) < (df['High'] - df['Low']).rolling(20).mean()*0.25)
    fade_up = near & (df['Close'] < df['Open'])
    fade_dn = near & (df['Close'] > df['Open'])
    return np.where(fade_dn, 1, np.where(fade_up, -1, 0))

# 275
def institutional_swing_failure(df, multi_data=None, sym=None, news_bias=0):
    swing_hi = df['High'].rolling(20).max().shift(1)
    swing_lo = df['Low'].rolling(20).min().shift(1)
    fail_up = (df['High'] > swing_hi) & (df['Close'] < swing_hi)
    fail_dn = (df['Low'] < swing_lo) & (df['Close'] > swing_lo)
    return np.where(fail_dn, 1, np.where(fail_up, -1, 0))

# 276
def ny_pm_session_reversal(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    ny_pm = hour.isin([18,19,20])
    strong_move = (df['Close'] - df['Open']).abs() > (df['High'] - df['Low']).rolling(20).mean()
    rev = ny_pm & strong_move & (df['Close'].diff() * (df['Close'] - df['Open']) < 0)
    return np.where(rev, -np.sign(df['Close'] - df['Open']), 0)

# 277
def killzone_news_trap(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    kz = hour.isin([7,8,9,10,13,14,15,16])
    spike = (df['High'] - df['Low']) > 1.6 * (df['High'] - df['Low']).rolling(20).mean()
    fake_up = kz & spike & (news_bias < 0) & (df['Close'] < df['Open'])
    fake_dn = kz & spike & (news_bias > 0) & (df['Close'] > df['Open'])
    return np.where(fake_dn, 1, np.where(fake_up, -1, 0))

# 278
def monday_liquidity_hunt(df, multi_data=None, sym=None, news_bias=0):
    if not hasattr(df.index, 'dayofweek'):
        return np.zeros(len(df))
    dow = df.index.dayofweek
    raid_hi = (df['High'] >= df['High'].rolling(24).max()) & (df['Close'] < df['Open'])
    raid_lo = (df['Low'] <= df['Low'].rolling(24).min()) & (df['Close'] > df['Open'])
    return np.where((dow == 0) & raid_lo, 1, np.where((dow == 0) & raid_hi, -1, 0))

# 279
def swing_liquidity_sweep_bos(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High'].rolling(20).max().shift(1)
    lo = df['Low'].rolling(20).min().shift(1)
    sweep_hi = (df['High'] > hi)
    sweep_lo = (df['Low'] < lo)
    bos_up = df['Close'] > df['High'].shift(1)
    bos_dn = df['Close'] < df['Low'].shift(1)
    long_sig = sweep_lo & bos_up
    short_sig = sweep_hi & bos_dn
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 280
def relative_equal_highs_lows_run(df, multi_data=None, sym=None, news_bias=0):
    eqh = (df['High'].round(4) == df['High'].shift(2).round(4)) | (df['High'].round(4) == df['High'].shift(3).round(4))
    eql = (df['Low'].round(4) == df['Low'].shift(2).round(4)) | (df['Low'].round(4) == df['Low'].shift(3).round(4))
    run_hi = eqh & (df['High'] > df['High'].shift(1)) & (df['Close'] < df['Open'])
    run_lo = eql & (df['Low'] < df['Low'].shift(1)) & (df['Close'] > df['Open'])
    return np.where(run_lo, 1, np.where(run_hi, -1, 0))

# 281
def order_block_chain_reaction(df, multi_data=None, sym=None, news_bias=0):
    ema = talib.EMA(df['Close'], 20)
    ob_bull = (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > ema)
    ob_bear = (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'] < ema)
    chain_bull = ob_bull & (df['Low'] <= df['Close'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    chain_bear = ob_bear & (df['High'] >= df['Close'].shift(1)) & (df['Close'] < df['Close'].shift(1))
    return np.where(chain_bull, 1, np.where(chain_bear, -1, 0))

# 282
def fvg_inside_order_block(df, multi_data=None, sym=None, news_bias=0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    imp_up = (df['Close'] - df['Open']) > 1.2 * atr
    imp_dn = (df['Open'] - df['Close']) > 1.2 * atr
    last_dn = (df['Close'].shift(1) < df['Open'].shift(1)) & imp_up
    last_up = (df['Close'].shift(1) > df['Open'].shift(1)) & imp_dn
    bull_fvg = df['Low'].shift(1) > df['High'].shift(2)
    bear_fvg = df['High'].shift(1) < df['Low'].shift(2)
    long_sig = last_dn & bull_fvg & (df['Low'] <= df['Close'].shift(1))
    short_sig = last_up & bear_fvg & (df['High'] >= df['Close'].shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 283
def breaker_mitigation_combo(df, multi_data=None, sym=None, news_bias=0):
    base = ict_breaker_block(df, multi_data, sym, news_bias) if 'ict_breaker_block' in globals() else np.zeros(len(df))
    retest = (df['Close'] == df['Close'].rolling(4).max()) | (df['Close'] == df['Close'].rolling(4).min())
    return np.where((base > 0) & retest, 1, np.where((base < 0) & retest, -1, 0))

# 284
def engulfing_ob_structure_flip(df, multi_data=None, sym=None, news_bias=0):
    bull_eng = (df['Close'] > df['Open']) & (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))
    bear_eng = (df['Close'] < df['Open']) & (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
    bos_up = df['Close'] > df['High'].shift(1)
    bos_dn = df['Close'] < df['Low'].shift(1)
    long_sig = bull_eng & bos_up
    short_sig = bear_eng & bos_dn
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 285
def range_to_trend_conversion(df, multi_data=None, sym=None, news_bias=0):
    rng = (df['High'] - df['Low'])
    tight = rng < 0.8 * rng.rolling(20).mean()
    stacked = tight & tight.shift(1) & tight.shift(2) & tight.shift(3)
    breakout_up = stacked & (df['Close'] > df['High'].rolling(5).max())
    breakout_dn = stacked & (df['Close'] < df['Low'].rolling(5).min())
    return np.where(breakout_up, 1, np.where(breakout_dn, -1, 0))

# 286
def stop_hunt_into_reversal_block(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High'].rolling(20).max().shift(1)
    lo = df['Low'].rolling(20).min().shift(1)
    hunt_up = (df['High'] > hi) & (df['Close'] < df['Open'])
    hunt_dn = (df['Low'] < lo) & (df['Close'] > df['Open'])
    rev_block_long = hunt_dn & (df['Low'] <= df['Close'].shift(1))
    rev_block_short = hunt_up & (df['High'] >= df['Close'].shift(1))
    return np.where(rev_block_long, 1, np.where(rev_block_short, -1, 0))

# 287
def session_vwap_reversion_trap(df, multi_data=None, sym=None, news_bias=0):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    dist = (df['Close'] - vwap) / vwap
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    active = hour.isin([7,8,9,10,13,14,15,16])
    long_sig = active & (dist < -0.01) & (df['Close'] > df['Open'])
    short_sig = active & (dist > 0.01) & (df['Close'] < df['Open'])
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 288
def cme_gap_fill_play(df, multi_data=None, sym=None, news_bias=0):
    ts = df.index.to_series()
    gap_open = (ts.diff().dt.total_seconds().fillna(0) > 8*3600)
    sess_open_price = df['Open'].where(gap_open).ffill()
    long_sig = (df['Close'] > df['Open']) & (df['Low'] <= sess_open_price) & (df['Close'] > sess_open_price)
    short_sig = (df['Close'] < df['Open']) & (df['High'] >= sess_open_price) & (df['Close'] < sess_open_price)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 289
def previous_session_sweep_reversal(df, multi_data=None, sym=None, news_bias=0):
    day = getattr(df.index, 'date', pd.Series(index=df.index))
    prev_high = df['High'].groupby(day).transform('max').shift(1)
    prev_low  = df['Low'].groupby(day).transform('min').shift(1)
    sweep_up = (df['High'] > prev_high) & (df['Close'] < df['Open'])
    sweep_dn = (df['Low'] < prev_low) & (df['Close'] > df['Open'])
    return np.where(sweep_dn, 1, np.where(sweep_up, -1, 0))

# 290
def london_fix_reversal(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    near_fix = hour == 16
    push = (df['Close'] - df['Open']).abs() > (df['High'] - df['Low']).rolling(20).mean()
    fade = near_fix & push & (df['Close'].diff() * (df['Close'] - df['Open']) < 0)
    return np.where(fade, -np.sign(df['Close'] - df['Open']), 0)

# 291
def pre_asia_stop_raid_reversal(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    pre_asia = hour.isin([22,23,0])
    raid_up = pre_asia & (df['High'] >= df['High'].rolling(24).max()) & (df['Close'] < df['Open'])
    raid_dn = pre_asia & (df['Low'] <= df['Low'].rolling(24).min()) & (df['Close'] > df['Open'])
    return np.where(raid_dn, 1, np.where(raid_up, -1, 0))

# 292
def ipda_midpoint_premium_discount(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High'].rolling(100).max()
    lo = df['Low'].rolling(100).min()
    mid = (hi + lo) / 2
    long_sig = (df['Close'] < mid) & (df['Low'] <= mid) & (df['Close'] > df['Open'])
    short_sig = (df['Close'] > mid) & (df['High'] >= mid) & (df['Close'] < df['Open'])
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 293
def ote_refined_entry(df, multi_data=None, sym=None, news_bias=0):
    hi = df['High'].rolling(40).max(); lo = df['Low'].rolling(40).min()
    ema20 = talib.EMA(df['Close'], 20); ema50 = talib.EMA(df['Close'], 50)
    up = ema20 > ema50
    buy_zone = (df['Low'] <= (hi - 0.62*(hi-lo))) & (df['Low'] >= (hi - 0.79*(hi-lo)))
    sell_zone = (df['High'] >= (lo + 0.62*(hi-lo))) & (df['High'] <= (lo + 0.79*(hi-lo)))
    return np.where(up & buy_zone, 1, np.where((~up) & sell_zone, -1, 0))

# 294
def power_of_three_intraday_v2(df, multi_data=None, sym=None, news_bias=0):
    rng = (df['High'] - df['Low'])
    accumulate = rng.rolling(8).mean() < 0.75 * rng.rolling(40).mean()
    manipulate = ((df['High'] > df['High'].rolling(30).max().shift(1)) & (df['Close'] < df['Open'])) | \
                 ((df['Low'] < df['Low'].rolling(30).min().shift(1)) & (df['Close'] > df['Open']))
    trend = talib.ADX(df['High'], df['Low'], df['Close'], 14) > 25
    score = accumulate.astype(int) + manipulate.astype(int) + trend.astype(int)
    dirn = np.sign(talib.LINEARREG_SLOPE(df['Close'], 10))
    return np.where(score >= 2, dirn, 0)

# 295
def asia_range_box_break_v2(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    asia = hour.isin([0,1,2,3,4,5,6])
    asia_hi = df['High'].where(asia).rolling(12).max().ffill()
    asia_lo = df['Low'].where(asia).rolling(12).min().ffill()
    break_up = (df['Close'] > asia_hi) & hour.isin([7,8,9,10])
    break_dn = (df['Close'] < asia_lo) & hour.isin([7,8,9,10])
    return np.where(break_up, 1, np.where(break_dn, -1, 0))

# 296
def london_session_engineered_sweep(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    london = hour.isin([7,8,9,10])
    sweep_hi = london & (df['High'] > df['High'].rolling(24).max().shift(1)) & (df['Close'] < df['Open'])
    sweep_lo = london & (df['Low'] < df['Low'].rolling(24).min().shift(1)) & (df['Close'] > df['Open'])
    return np.where(sweep_lo, 1, np.where(sweep_hi, -1, 0))

# 297
def ny_session_engineered_sweep(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    ny = hour.isin([13,14,15,16])
    sweep_hi = ny & (df['High'] > df['High'].rolling(24).max().shift(1)) & (df['Close'] < df['Open'])
    sweep_lo = ny & (df['Low'] < df['Low'].rolling(24).min().shift(1)) & (df['Close'] > df['Open'])
    return np.where(sweep_lo, 1, np.where(sweep_hi, -1, 0))

# 298
def session_bos_confirmation(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    active = hour.isin([7,8,9,10,13,14,15,16])
    bos_up = active & (df['Close'] > df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    bos_dn = active & (df['Close'] < df['Low'].shift(1)) & (df['High'] < df['High'].shift(1))
    return np.where(bos_up, 1, np.where(bos_dn, -1, 0))

# 299
def mitigation_block_return(df, multi_data=None, sym=None, news_bias=0):
    ema = talib.EMA(df['Close'], 20)
    imp_up = (df['Close'] - df['Open']) > (df['High'] - df['Low']).rolling(20).mean()
    imp_dn = (df['Open'] - df['Close']) > (df['High'] - df['Low']).rolling(20).mean()
    long_sig = imp_up.shift(1) & (df['Low'] <= ema) & (df['Close'] > ema)
    short_sig = imp_dn.shift(1) & (df['High'] >= ema) & (df['Close'] < ema)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 300
def time_price_synchronicity_pop(df, multi_data=None, sym=None, news_bias=0):
    hour = getattr(df.index, 'hour', pd.Series(index=df.index, dtype=int))
    window = hour.isin([7,8,13,14])
    rng = (df['High'] - df['Low'])
    squeeze = rng < 0.8 * rng.rolling(20).mean()
    pop_up = window & (~squeeze) & (df['Close'] > df['High'].rolling(5).max())
    pop_dn = window & (~squeeze) & (df['Close'] < df['Low'].rolling(5).min())
    return np.where(pop_up, 1, np.where(pop_dn, -1, 0))

# 301
def ict_fvg_displacement_mid_retest(df, multi_data=None, sym=None, news_bias=0):
    # FVG (3-candle) gated by displacement; entry on mid-gap retest
    h2, l2 = df['High'].shift(2), df['Low'].shift(2)
    h1, l1 = df['High'].shift(1), df['Low'].shift(1)
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    disp_up = (df['Close'].shift(1) - df['Open'].shift(1)) > 1.0 * atr.shift(1)
    disp_dn = (df['Open'].shift(1) - df['Close'].shift(1)) > 1.0 * atr.shift(1)
    bull_fvg = (l1 > h2) & disp_up
    bear_fvg = (h1 < l2) & disp_dn
    mid_bull = (h2 + l1) / 2.0
    mid_bear = (l2 + h1) / 2.0
    long_sig  = bull_fvg & (df['Low'] <= mid_bull) & (df['Close'] > mid_bull)
    short_sig = bear_fvg & (df['High'] >= mid_bear) & (df['Close'] < mid_bear)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 302
def ict_bos_choch_structure(df, multi_data=None, sym=None, news_bias=0, swing=3):
    # BOS/CHOCH using swing highs/lows
    hh = df['High'].rolling(swing).max()
    ll = df['Low'].rolling(swing).min()
    prev_hh = hh.shift(1); prev_ll = ll.shift(1)
    bos_up = df['Close'] > prev_hh  # break of prior swing high
    bos_dn = df['Close'] < prev_ll  # break of prior swing low
    # CHOCH when direction flips vs prior 20 bars trend proxy
    trend = talib.EMA(df['Close'], 20)
    choch_up = bos_up & (df['Close'] > trend)
    choch_dn = bos_dn & (df['Close'] < trend)
    return np.where(choch_up, 1, np.where(choch_dn, -1, 0))

# 303
def ict_liquidity_raid_then_fvg_entry(df, multi_data=None, sym=None, news_bias=0, lookback=20):
    # Raid (sweep) of recent high/low, then FVG forms and mean-revert into it
    hi = df['High'].rolling(lookback).max().shift(1)
    lo = df['Low'].rolling(lookback).min().shift(1)
    raid_up = df['High'] > hi
    raid_dn = df['Low'] < lo
    # FVG the next bar (simplified body gap)
    h2, l2 = df['High'].shift(2), df['Low'].shift(2)
    h1, l1 = df['High'].shift(1), df['Low'].shift(1)
    bull_fvg_next = (l1 > h2)
    bear_fvg_next = (h1 < l2)
    long_sig  = raid_dn & bull_fvg_next & (df['Close'] > (h2 + l1)/2)
    short_sig = raid_up & bear_fvg_next & (df['Close'] < (l2 + h1)/2)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 304
def smc_mitigation_block_retest(df, multi_data=None, sym=None, news_bias=0):
    # Mitigation block: last opposing candle before impulse; entry on 50% retest
    atr = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    imp_up = (df['Close'] - df['Open']) > 1.3 * atr
    imp_dn = (df['Open'] - df['Close']) > 1.3 * atr
    last_down = (df['Close'].shift(1) < df['Open'].shift(1)) & imp_up
    last_up   = (df['Close'].shift(1) > df['Open'].shift(1)) & imp_dn
    mid_dn = (df['Open'].shift(1) + df['Close'].shift(1)) / 2.0
    mid_up = (df['Open'].shift(1) + df['Close'].shift(1)) / 2.0
    long_sig  = last_down & (df['Low'] <= mid_dn) & (df['Close'] > mid_dn)
    short_sig = last_up   & (df['High'] >= mid_up) & (df['Close'] < mid_up)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 305
def smc_premium_discount_pd_array(df, multi_data=None, sym=None, news_bias=0, lookback=50):
    # Trade in discount (<EQ) for longs and premium (>EQ) for shorts with structure filter
    swing_hi = df['High'].rolling(lookback).max()
    swing_lo = df['Low'].rolling(lookback).min()
    eq = (swing_hi + swing_lo) / 2.0  # equilibrium 50%
    bos_up = df['Close'] > swing_hi.shift(1)
    bos_dn = df['Close'] < swing_lo.shift(1)
    long_sig  = (df['Close'] < eq) & bos_up
    short_sig = (df['Close'] > eq) & bos_dn
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 306
def smc_supply_demand_break_retest(df, multi_data=None, sym=None, news_bias=0, w=20):
    # Break of structure through supply/demand, then retest the zone body
    demand = df['Low'].rolling(w).min()
    supply = df['High'].rolling(w).max()
    bos_up = df['Close'] > supply.shift(1)
    bos_dn = df['Close'] < demand.shift(1)
    long_sig  = bos_up & (df['Low'] <= supply.shift(1)) & (df['Close'] > supply.shift(1))
    short_sig = bos_dn & (df['High'] >= demand.shift(1)) & (df['Close'] < demand.shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 307
def ict_london_killzone_breakout(df, multi_data=None, sym=None, news_bias=0, asia_hours=range(0,6), london_hours=range(7,11)):
    # Asian range + London KZ breakout (uses df.index.hour; ensure timezone-consistent data)
    hours = pd.Series(df.index.hour, index=df.index)
    asia_mask = hours.isin(asia_hours)
    london_mask = hours.isin(london_hours)
    asia_hi = df['High'].where(asia_mask).rolling(12, min_periods=1).max().ffill()
    asia_lo = df['Low' ].where(asia_mask).rolling(12, min_periods=1).min().ffill()
    long_sig  = london_mask & (df['Close'] > asia_hi)
    short_sig = london_mask & (df['Close'] < asia_lo)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 308
def ict_judas_swing_reversal(df, multi_data=None, sym=None, news_bias=0, lookback=24, window=6):
    # Early-session fakeout beyond prior extremes, then close back inside within N bars
    hi = df['High'].rolling(lookback).max().shift(1)
    lo = df['Low'].rolling(lookback).min().shift(1)
    fake_high = (df['High'] > hi) & (df['Close'] < hi)
    fake_low  = (df['Low']  < lo) & (df['Close'] > lo)
    # confirm within window by looking back outcome
    confirm_up   = fake_low & (df['Close'].rolling(window).max() > hi)
    confirm_down = fake_high & (df['Close'].rolling(window).min() < lo)
    return np.where(confirm_up, 1, np.where(confirm_down, -1, 0))

# 309
def retail_ut_bot(df, multi_data=None, sym=None, news_bias=0, length=14, factor=1.5):
    # UT Bot (approx.) using ATR trailing stop logic
    atr = talib.ATR(df['High'], df['Low'], df['Close'], length)
    hl2 = (df['High'] + df['Low']) / 2.0
    buy_trail  = hl2 - factor * atr
    sell_trail = hl2 + factor * atr
    buy_line  = np.maximum.accumulate(buy_trail)
    sell_line = np.minimum.accumulate(sell_trail)
    long_sig  = (df['Close'] > sell_line) & (df['Close'].shift(1) <= sell_line.shift(1))
    short_sig = (df['Close'] < buy_line)  & (df['Close'].shift(1) >= buy_line.shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 310
def retail_supertrend_trend_follow(df, multi_data=None, sym=None, news_bias=0, period=10, mult=3.0):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], period)
    mprice = (df['High'] + df['Low']) / 2.0
    upper = mprice + mult * atr
    lower = mprice - mult * atr
    dir_up = np.where(df['Close'] > upper.shift(1), 1, np.where(df['Close'] < lower.shift(1), -1, np.nan))
    dir_up = pd.Series(dir_up, index=df.index).ffill().fillna(0)
    return dir_up.to_numpy().astype(int)

# 311
def retail_tdi_cross(df, multi_data=None, sym=None, news_bias=0, rsi_len=13, ma_len=2, bb_len=34, bb_mult=1.618):
    rsi = talib.RSI(df['Close'], rsi_len)
    rsi_ma = talib.SMA(rsi, ma_len)
    mid = talib.SMA(rsi, bb_len)
    dev = talib.STDDEV(rsi, bb_len)
    upper = mid + bb_mult * dev
    lower = mid - bb_mult * dev
    bull = (rsi_ma.shift(1) < rsi.shift(1)) & (rsi_ma > rsi) & (rsi < lower)
    bear = (rsi_ma.shift(1) > rsi.shift(1)) & (rsi_ma < rsi) & (rsi > upper)
    return np.where(bull, 1, np.where(bear, -1, 0))

# 312
def retail_qqe_cross(df, multi_data=None, sym=None, news_bias=0, rsi_len=14, smooth=5, qqe=4.236):
    rsi = talib.RSI(df['Close'], rsi_len)
    rsi_s = talib.EMA(rsi, smooth)
    tr = abs(rsi_s - rsi_s.shift(1))
    dar = talib.EMA(tr, smooth) * qqe
    long_sig  = (rsi_s > rsi_s.shift(1)) & (rsi_s.shift(1) <= (rsi_s.shift(2) + dar.shift(2)))
    short_sig = (rsi_s < rsi_s.shift(1)) & (rsi_s.shift(1) >= (rsi_s.shift(2) - dar.shift(2)))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 313
def retail_turtle_soup(df, multi_data=None, sym=None, news_bias=0, n=20):
    prior_high = df['High'].rolling(n).max().shift(1)
    prior_low  = df['Low'].rolling(n).min().shift(1)
    short_sig = (df['High'] > prior_high) & (df['Close'] < prior_high)
    long_sig  = (df['Low']  < prior_low)  & (df['Close'] > prior_low)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 314
def retail_inside_bar_breakout(df, multi_data=None, sym=None, news_bias=0):
    inside = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    long_sig  = inside.shift(1) & (df['Close'] > df['High'].shift(1))
    short_sig = inside.shift(1) & (df['Close'] < df['Low'].shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 315
def retail_two_b_reversal(df, multi_data=None, sym=None, news_bias=0, n=10):
    # 2B pattern near recent extremes
    recent_hi = df['High'].rolling(n).max().shift(1)
    recent_lo = df['Low'].rolling(n).min().shift(1)
    prob_short = (df['High'] > recent_hi) & (df['Close'] < recent_hi)
    prob_long  = (df['Low']  < recent_lo) & (df['Close'] > recent_lo)
    return np.where(prob_long, 1, np.where(prob_short, -1, 0))

# 316
def retail_hikkake_pattern(df, multi_data=None, sym=None, news_bias=0):
    inside = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    fake_break_up = inside.shift(1) & (df['High'] > df['High'].shift(1)) & (df['Close'] < df['High'].shift(1))
    fake_break_dn = inside.shift(1) & (df['Low']  < df['Low'].shift(1))  & (df['Close'] > df['Low'].shift(1))
    long_sig  = fake_break_dn
    short_sig = fake_break_up
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 317
def retail_heikin_ashi_trend_pullback(df, multi_data=None, sym=None, news_bias=0):
    # Heikin-Ashi candles
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4.0
    ha_open = (ha_close.shift(1) + ha_close.shift(2)) / 2.0
    ha_trend = talib.EMA(ha_close, 20)
    long_sig  = (ha_close > ha_open) & (ha_close > ha_trend) & (df['Close'] > talib.EMA(df['Close'], 50))
    short_sig = (ha_close < ha_open) & (ha_close < ha_trend) & (df['Close'] < talib.EMA(df['Close'], 50))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 318
def retail_keltner_breakout_pullback(df, multi_data=None, sym=None, news_bias=0, ema_len=20, mult=2.0):
    ema = talib.EMA(df['Close'], ema_len)
    tr = talib.ATR(df['High'], df['Low'], df['Close'], ema_len)
    upper = ema + mult * tr
    lower = ema - mult * tr
    long_sig  = (df['Close'] > upper) & (df['Low'] <= ema)
    short_sig = (df['Close'] < lower) & (df['High'] >= ema)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 319
def retail_vwap_band_reversion(df, multi_data=None, sym=None, news_bias=0, dev=2.0):
    # Requires Volume; session-agnostic VWAP
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    std = (df['Close'] - vwap).rolling(50).std()
    upper = vwap + dev * std
    lower = vwap - dev * std
    long_sig  = df['Close'] < lower
    short_sig = df['Close'] > upper
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 320
def retail_rsi_divergence(df, multi_data=None, sym=None, news_bias=0, rsi_len=14, look=5):
    rsi = talib.RSI(df['Close'], rsi_len)
    price_hh = df['Close'] > df['Close'].shift(look)
    rsi_ll   = rsi < rsi.shift(look)
    bear_div = price_hh & rsi_ll
    price_ll = df['Close'] < df['Close'].shift(look)
    rsi_hh   = rsi > rsi.shift(look)
    bull_div = price_ll & rsi_hh
    return np.where(bull_div, 1, np.where(bear_div, -1, 0))

# 321
def retail_bollinger_mtf_reversion(df, multi_data=None, sym=None, news_bias=0, n=20, k=2.0):
    mid = talib.SMA(df['Close'], n)
    std = talib.STDDEV(df['Close'], n)
    upper = mid + k * std
    lower = mid - k * std
    trend = talib.EMA(df['Close'], 100)
    long_sig  = (df['Close'] < lower) & (df['Close'] > trend * 0.9)
    short_sig = (df['Close'] > upper) & (df['Close'] < trend * 1.1)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 322
def retail_donchian_trend_atr_trail(df, multi_data=None, sym=None, news_bias=0, n=20, atr_n=14):
    upper = df['High'].rolling(n).max()
    lower = df['Low'].rolling(n).min()
    atr = talib.ATR(df['High'], df['Low'], df['Close'], atr_n)
    long_sig  = df['Close'] > upper.shift(1)
    short_sig = df['Close'] < lower.shift(1)
    # optional trail: flip if close crosses back by >0.5*ATR
    exit_long  = long_sig & (df['Close'] < upper.shift(1) - 0.5 * atr)
    exit_short = short_sig & (df['Close'] > lower.shift(1) + 0.5 * atr)
    sig = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    sig = np.where(exit_long, 0, np.where(exit_short, 0, sig))
    return sig

# 323
def retail_range_filter_breakout(df, multi_data=None, sym=None, news_bias=0, n=14, mult=1.5):
    rng = (df['High'] - df['Low'])
    filt = talib.EMA(rng, n) * mult
    basis = talib.EMA(df['Close'], n)
    long_sig  = df['Close'] > (basis + filt)
    short_sig = df['Close'] < (basis - filt)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 324
def ict_equilibrium_fvg_confluence(df, multi_data=None, sym=None, news_bias=0, look=50):
    # Confluence: price in discount/premium + FVG revisit
    swing_hi = df['High'].rolling(look).max()
    swing_lo = df['Low'].rolling(look).min()
    eq = (swing_hi + swing_lo) / 2.0
    h2, l2 = df['High'].shift(2), df['Low'].shift(2)
    h1, l1 = df['High'].shift(1), df['Low'].shift(1)
    bull_fvg = l1 > h2; bear_fvg = h1 < l2
    mid_bull = (h2 + l1)/2.0; mid_bear = (l2 + h1)/2.0
    long_sig  = (df['Close'] < eq) & bull_fvg & (df['Low'] <= mid_bull) & (df['Close'] > mid_bull)
    short_sig = (df['Close'] > eq) & bear_fvg & (df['High'] >= mid_bear) & (df['Close'] < mid_bear)
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# 325
def smc_mss_breaker_shift(df, multi_data=None, sym=None, news_bias=0, swing=3):
    # Market Structure Shift (MSS) then breaker-like retest
    hh = df['High'].rolling(swing).max()
    ll = df['Low'].rolling(swing).min()
    mss_down = (df['Close'] < ll.shift(1)) & (df['Close'].shift(1) > ll.shift(2))
    mss_up   = (df['Close'] > hh.shift(1)) & (df['Close'].shift(1) < hh.shift(2))
    # breaker retest: return to the broken swing level
    long_sig  = mss_up   & (df['Low']  <= hh.shift(1)) & (df['Close'] > hh.shift(1))
    short_sig = mss_down & (df['High'] >= ll.shift(1)) & (df['Close'] < ll.shift(1))
    return np.where(long_sig, 1, np.where(short_sig, -1, 0))

# Consolidate strategies and define signal generation function
strategies = {name: func for name, func in globals().items() if callable(func) and name not in ['np', 'pd', 'talib', 'generate_signals']}

def generate_signals(df, strategy_name, **params):
    """
    Generate trading signals for the given DataFrame using the specified strategy.
    Args:
        df: DataFrame with OHLCV columns and indicators.
        strategy_name: Name of the strategy function to use.
        **params: Additional parameters (e.g., multi_data, news_bias).
    Returns:
        pandas Series of signals indexed to df.
    """
    if strategy_name not in strategies:
        raise ValueError(f"Strategy '{strategy_name}' is not recognized.")
    if len(df) < 200:
        raise ValueError("DataFrame must have at least 200 rows for strategy calculations.")
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing required columns: {required_cols}")
    
    strat_func = strategies[strategy_name]
    result = strat_func(df, **params)
    
    if isinstance(result, np.ndarray) and np.isnan(result).mean() > 0.5:
        print(f"Warning: Strategy '{strategy_name}' produced >50% NaN signals.")
    
    if isinstance(result, np.ndarray):
        return pd.Series(result, index=df.index)
    elif isinstance(result, pd.Series):
        return result
    else:
        return pd.Series(np.asarray(result), index=df.index)

# New strategy: evolved_macd_897
def evolved_macd_897(df, multi_data=None, sym=None, news_bias=0):
    macd, signal, hist = talib.MACD(df['Close'], fastperiod=20, slowperiod=24, signalperiod=11)
    return np.where(macd > signal, 1, np.where(macd < signal, -1, 0))



# New strategy: evolved_macd_905
def evolved_macd_905(df, multi_data=None, sym=None, news_bias=0):
    macd, signal, hist = talib.MACD(df['Close'], fastperiod=16, slowperiod=27, signalperiod=14)
    return np.where(macd > signal, 1, np.where(macd < signal, -1, 0))



# New strategy: evolved_macd_477
def evolved_macd_477(df, multi_data=None, sym=None, news_bias=0):
    macd, signal, hist = talib.MACD(df['Close'], fastperiod=9, slowperiod=23, signalperiod=5)
    return np.where(macd > signal, 1, np.where(macd < signal, -1, 0))



# New strategy: evolved_macd_567
def evolved_macd_567(df, multi_data=None, sym=None, news_bias=0):
    macd, signal, hist = talib.MACD(df['Close'], fastperiod=18, slowperiod=34, signalperiod=6)
    return np.where(macd > signal, 1, np.where(macd < signal, -1, 0))

