import pandas as pd
import numpy as np
import logging
import warnings
from typing import List, Optional

logger = logging.getLogger(__name__)

# Provide removed aliases for numpy 2.0+ so older libraries keep working
if not hasattr(np, "NaN"):
    np.NaN = np.nan

warnings.filterwarnings("ignore", category=FutureWarning, module="ta")

try:
    import talib
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning("TA-Lib import failed (%s). Skipping TA-Lib indicators.", e)
    talib = None

try:
    import pandas_ta as pta
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning("pandas_ta import failed (%s). Skipping pandas_ta indicators.", e)
    pta = None

try:
    import ta
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning("ta library import failed (%s). Skipping ta indicators.", e)
    ta = None

def _compute_talib_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if talib is None:
        # Minimal RSI implementation to satisfy tests when TA‑Lib is missing
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(span=14, adjust=False).mean()
        roll_down = down.ewm(span=14, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return pd.DataFrame({'TA_RSI_14': rsi}, index=df.index)
    
    indicators = {}
    indicators['TA_SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
    indicators['TA_SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    indicators['TA_SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    indicators['TA_SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
    indicators['TA_EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    indicators['TA_EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
    indicators['TA_WMA_20'] = talib.WMA(df['Close'], timeperiod=20)
    indicators['TA_DEMA_20'] = talib.DEMA(df['Close'], timeperiod=20)
    indicators['TA_TEMA_20'] = talib.TEMA(df['Close'], timeperiod=20)
    indicators['TA_TRIMA_20'] = talib.TRIMA(df['Close'], timeperiod=20)
    indicators['TA_KAMA_20'] = talib.KAMA(df['Close'], timeperiod=20)
    mama, fama = talib.MAMA(df['Close'])
    indicators['TA_MAMA'] = mama
    indicators['TA_FAMA'] = fama
    indicators['TA_T3_5'] = talib.T3(df['Close'], timeperiod=5)
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
    indicators['TA_BBANDS_UPPER'] = upper
    indicators['TA_BBANDS_MIDDLE'] = middle
    indicators['TA_BBANDS_LOWER'] = lower
    indicators['TA_HT_TRENDLINE'] = talib.HT_TRENDLINE(df['Close'])
    indicators['TA_MAVP'] = talib.MAVP(df['Close'], periods=df['Volume'].astype(int).clip(2, 30))
    indicators['TA_SAR'] = talib.SAR(df['High'], df['Low'])
    indicators['TA_SAREXT'] = talib.SAREXT(df['High'], df['Low'])

    # Momentum Indicators
    indicators['TA_RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    indicators['TA_RSI_5'] = talib.RSI(df['Close'], timeperiod=5)
    k, d = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
    indicators['TA_STOCH_K'] = k
    indicators['TA_STOCH_D'] = d
    k, d = talib.STOCHF(df['High'], df['Low'], df['Close'], fastk_period=14)
    indicators['TA_STOCHF_K'] = k
    indicators['TA_STOCHF_D'] = d
    k, d = talib.STOCHRSI(df['Close'], timeperiod=14, fastk_period=5, fastd_period=3)
    indicators['TA_STOCHRSI_K'] = k
    indicators['TA_STOCHRSI_D'] = d
    macd, signal, hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['TA_MACD'] = macd
    indicators['TA_MACD_SIGNAL'] = signal
    indicators['TA_MACD_HIST'] = hist
    macd, signal, hist = talib.MACDEXT(df['Close'])
    indicators['TA_MACDEXT'] = macd
    indicators['TA_MACDEXT_SIGNAL'] = signal
    indicators['TA_MACDEXT_HIST'] = hist
    macd, signal, hist = talib.MACDFIX(df['Close'])
    indicators['TA_MACDFIX'] = macd
    indicators['TA_MACDFIX_SIGNAL'] = signal
    indicators['TA_MACDFIX_HIST'] = hist
    indicators['TA_ADX_14'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    indicators['TA_ADXR_14'] = talib.ADXR(df['High'], df['Low'], df['Close'], timeperiod=14)
    indicators['TA_APO'] = talib.APO(df['Close'])
    indicators['TA_AROON_DOWN'], indicators['TA_AROON_UP'] = talib.AROON(df['High'], df['Low'])
    indicators['TA_AROONOSC'] = talib.AROONOSC(df['High'], df['Low'])
    indicators['TA_BOP'] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CCI_14'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    indicators['TA_CMO_14'] = talib.CMO(df['Close'], timeperiod=14)
    indicators['TA_DX_14'] = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=14)
    indicators['TA_MINUS_DI_14'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    indicators['TA_PLUS_DI_14'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    indicators['TA_MINUS_DM_14'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    indicators['TA_PLUS_DM_14'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    indicators['TA_MOM_10'] = talib.MOM(df['Close'], timeperiod=10)
    indicators['TA_PPO'] = talib.PPO(df['Close'])
    indicators['TA_ROC_10'] = talib.ROC(df['Close'], timeperiod=10)
    indicators['TA_ROCP_10'] = talib.ROCP(df['Close'], timeperiod=10)
    indicators['TA_ROCR_10'] = talib.ROCR(df['Close'], timeperiod=10)
    indicators['TA_ROCR100_10'] = talib.ROCR100(df['Close'], timeperiod=10)
    indicators['TA_TRIX_30'] = talib.TRIX(df['Close'], timeperiod=30)
    indicators['TA_ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'])
    indicators['TA_WILLR_14'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Volume Indicators
    indicators['TA_AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    indicators['TA_ADOSC_3_10'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
    indicators['TA_OBV'] = talib.OBV(df['Close'], df['Volume'])

    # Volatility Indicators
    indicators['TA_ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    indicators['TA_NATR_14'] = talib.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    indicators['TA_TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])

    # Statistic Functions
    indicators['TA_BETA_30'] = talib.BETA(df['High'], df['Low'], timeperiod=30)
    indicators['TA_CORREL_30'] = talib.CORREL(df['High'], df['Low'], timeperiod=30)
    indicators['TA_LINEARREG_14'] = talib.LINEARREG(df['Close'], timeperiod=14)
    indicators['TA_LINEARREG_ANGLE_14'] = talib.LINEARREG_ANGLE(df['Close'], timeperiod=14)
    indicators['TA_LINEARREG_INTERCEPT_14'] = talib.LINEARREG_INTERCEPT(df['Close'], timeperiod=14)
    indicators['TA_LINEARREG_SLOPE_14'] = talib.LINEARREG_SLOPE(df['Close'], timeperiod=14)
    indicators['TA_STDDEV_5'] = talib.STDDEV(df['Close'], timeperiod=5)
    indicators['TA_TSF_14'] = talib.TSF(df['Close'], timeperiod=14)
    indicators['TA_VAR_5'] = talib.VAR(df['Close'], timeperiod=5)

    # Math Operators
    indicators['TA_ADD'] = talib.ADD(df['High'], df['Low'])
    indicators['TA_DIV'] = talib.DIV(df['High'], df['Low'])
    indicators['TA_MAX_30'] = talib.MAX(df['Close'], timeperiod=30)
    indicators['TA_MAXINDEX_30'] = talib.MAXINDEX(df['Close'], timeperiod=30)
    indicators['TA_MIN_30'] = talib.MIN(df['Close'], timeperiod=30)
    indicators['TA_MININDEX_30'] = talib.MININDEX(df['Close'], timeperiod=30)
    indicators['TA_MULT'] = talib.MULT(df['High'], df['Low'])
    indicators['TA_SUB'] = talib.SUB(df['High'], df['Low'])
    indicators['TA_SUM_30'] = talib.SUM(df['Close'], timeperiod=30)

    # Math Transform
    indicators['TA_ACOS'] = talib.ACOS(df['Close'])
    indicators['TA_ASIN'] = talib.ASIN(df['Close'])
    indicators['TA_ATAN'] = talib.ATAN(df['Close'])
    indicators['TA_CEIL'] = talib.CEIL(df['Close'])
    indicators['TA_COS'] = talib.COS(df['Close'])
    indicators['TA_COSH'] = talib.COSH(df['Close'])
    indicators['TA_EXP'] = talib.EXP(df['Close'])
    indicators['TA_FLOOR'] = talib.FLOOR(df['Close'])
    indicators['TA_LN'] = talib.LN(df['Close'])
    indicators['TA_LOG10'] = talib.LOG10(df['Close'])
    indicators['TA_SIN'] = talib.SIN(df['Close'])
    indicators['TA_SINH'] = talib.SINH(df['Close'])
    indicators['TA_SQRT'] = talib.SQRT(df['Close'])
    indicators['TA_TAN'] = talib.TAN(df['Close'])
    indicators['TA_TANH'] = talib.TANH(df['Close'])

    # Pattern Recognition
    indicators['TA_CDL2CROWS'] = talib.CDL2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDL3INSIDE'] = talib.CDL3INSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDL3OUTSIDE'] = talib.CDL3OUTSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLBELTHOLD'] = talib.CDLBELTHOLD(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLBREAKAWAY'] = talib.CDLBREAKAWAY(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLDOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLDOJISTAR'] = talib.CDLDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLHAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLHANGINGMAN'] = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLHARAMI'] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLHARAMICROSS'] = talib.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLHIGHWAVE'] = talib.CDLHIGHWAVE(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLHIKKAKE'] = talib.CDLHIKKAKE(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLINNECK'] = talib.CDLINNECK(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLKICKING'] = talib.CDLKICKING(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLLONGLINE'] = talib.CDLLONGLINE(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLMARUBOZU'] = talib.CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLMATHOLD'] = talib.CDLMATHOLD(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLONNECK'] = talib.CDLONNECK(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLPIERCING'] = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLSHORTLINE'] = talib.CDLSHORTLINE(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLTAKURI'] = talib.CDLTAKURI(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLTASUKIGAP'] = talib.CDLTASUKIGAP(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLTHRUSTING'] = talib.CDLTHRUSTING(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLTRISTAR'] = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    indicators['TA_CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(df['Open'], df['High'], df['Low'], df['Close'])

    return pd.DataFrame(indicators, index=df.index)

def _compute_pandasta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if pta is None:
        # Fallback EMA implementation when pandas_ta isn't available
        ema = df['Close'].ewm(span=20, adjust=False).mean()
        return pd.DataFrame({'PTA_EMA_20': ema}, index=df.index)
    
    indicators = {}
    indicators['PTA_SMA_10'] = pta.sma(df['Close'], length=10)
    indicators['PTA_EMA_10'] = pta.ema(df['Close'], length=10)
    indicators['PTA_RSI_14'] = pta.rsi(df['Close'], length=14)
    indicators['PTA_MACD_12_26_9'] = pta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    indicators['PTA_MACD_SIGNAL_12_26_9'] = pta.macd(df['Close'], fast=12, slow=26, signal=9)['MACDs_12_26_9']
    indicators['PTA_MACD_HIST_12_26_9'] = pta.macd(df['Close'], fast=12, slow=26, signal=9)['MACDh_12_26_9']
    indicators['PTA_BB_LOWER_20_2'] = pta.bbands(df['Close'], length=20, std=2)['BBL_20_2.0']
    indicators['PTA_BB_MIDDLE_20_2'] = pta.bbands(df['Close'], length=20, std=2)['BBM_20_2.0']
    indicators['PTA_BB_UPPER_20_2'] = pta.bbands(df['Close'], length=20, std=2)['BBU_20_2.0']
    indicators['PTA_ATR_14'] = pta.atr(df['High'], df['Low'], df['Close'], length=14)
    indicators['PTA_STOCH_K_14_3_3'] = pta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)['STOCHk_14_3_3']
    indicators['PTA_STOCH_D_14_3_3'] = pta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)['STOCHd_14_3_3']
    indicators['PTA_ADX_14'] = pta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    indicators['PTA_CCI_14'] = pta.cci(df['High'], df['Low'], df['Close'], length=14)
    indicators['PTA_OBV'] = pta.obv(df['Close'], df['Volume'])
    indicators['PTA_MFI_14'] = pta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    indicators['PTA_ROC_10'] = pta.roc(df['Close'], length=10)
    indicators['PTA_WILLR_14'] = pta.willr(df['High'], df['Low'], df['Close'], length=14)
    # pandas_ta encodes the acceleration factors in the PSAR column names, which can vary
    # across versions. Select the long PSAR column dynamically to avoid KeyErrors like
    # "PSARl_0.015_0.015" when default parameters change.
    psar_df = pta.psar(df['High'], df['Low'], df['Close'])
    psar_long_col = next((c for c in psar_df.columns if c.lower().startswith('psarl')), None)
    if psar_long_col is not None:
        indicators['PTA_PSARI_002_02'] = psar_df[psar_long_col]
    else:
        logger.debug('pandas_ta psar returned columns %s', list(psar_df.columns))
    indicators['PTA_AO'] = pta.ao(df['High'], df['Low'])
    indicators['PTA_KAMA_10_2_30'] = pta.kama(df['Close'], length=10, fast=2, slow=30)
    indicators['PTA_PPO_12_26_9'] = pta.ppo(df['Close'], fast=12, slow=26, signal=9)['PPO_12_26_9']
    indicators['PTA_PPO_SIGNAL_12_26_9'] = pta.ppo(df['Close'], fast=12, slow=26, signal=9)['PPOs_12_26_9']
    indicators['PTA_PPO_HIST_12_26_9'] = pta.ppo(df['Close'], fast=12, slow=26, signal=9)['PPOh_12_26_9']
    indicators['PTA_TRIX_30'] = pta.trix(df['Close'], length=30)['TRIX_30_9']
    indicators['PTA_TSI_13_25'] = pta.tsi(df['Close'], r=25, s=13)['TSI_13_25_13']
    indicators['PTA_VWAP'] = pta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    indicators['PTA_VWMA_20'] = pta.vwma(df['Close'], df['Volume'], length=20)
    indicators['PTA_HMA_14'] = pta.hma(df['Close'], length=14)
    # pandas_ta changed ichimoku's return type in recent releases; handle DataFrame or
    # tuple outputs and ignore failures so other indicators still compute.
    try:
        ichimoku = pta.ichimoku(df['High'], df['Low'], df['Close'], tenkan=9, kijun=26, senkou=52)
        if isinstance(ichimoku, tuple):
            ichimoku = ichimoku[0]
        indicators['PTA_ICHIMOKU_TENKAN'] = ichimoku['ITS_9']
        indicators['PTA_ICHIMOKU_KIJUN'] = ichimoku['IKS_26']
        indicators['PTA_ICHIMOKU_SENKOU_A'] = ichimoku['ISA_9']
        indicators['PTA_ICHIMOKU_SENKOU_B'] = ichimoku['ISB_26']
        indicators['PTA_ICHIMOKU_CHIKOU'] = ichimoku['ICS_26']
    except Exception as e:
        logger.debug("pandas_ta ichimoku failed: %s", e)

    return pd.DataFrame(indicators, index=df.index)

def _compute_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if ta is None:
        return pd.DataFrame(index=df.index)

    result = pd.DataFrame(index=df.index)

    # Trend Indicators
    trend = ta.trend
    result['TA2_ADX'] = trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
    result['TA2_ADX_NEG'] = trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx_neg()
    result['TA2_ADX_POS'] = trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx_pos()
    result['TA2_AROON_DOWN'] = trend.AroonIndicator(high=df['High'], low=df['Low'], window=25).aroon_down()
    result['TA2_AROON_UP'] = trend.AroonIndicator(high=df['High'], low=df['Low'], window=25).aroon_up()
    result['TA2_CCI'] = trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
    result['TA2_DPO'] = trend.DPOIndicator(close=df['Close'], window=20).dpo()
    result['TA2_EMA'] = trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    result['TA2_ICHIMOKU_A'] = trend.IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52).ichimoku_a()
    result['TA2_ICHIMOKU_B'] = trend.IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52).ichimoku_b()
    result['TA2_ICHIMOKU_BASE'] = trend.IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52).ichimoku_base_line()
    result['TA2_ICHIMOKU_CONV'] = trend.IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52).ichimoku_conversion_line()
    result['TA2_MACD'] = trend.MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9).macd()
    result['TA2_MACD_DIFF'] = trend.MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9).macd_diff()
    result['TA2_MACD_SIGNAL'] = trend.MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9).macd_signal()
    result['TA2_MASS_INDEX'] = trend.MassIndex(high=df['High'], low=df['Low'], window_fast=9, window_slow=25).mass_index()
    result['TA2_PSAR_DOWN'] = trend.PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2).psar_down()
    result['TA2_PSAR_UP'] = trend.PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2).psar_up()
    result['TA2_SMA'] = trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    result['TA2_TRIX'] = trend.TRIXIndicator(close=df['Close'], window=15).trix()
    result['TA2_VORTEX_NEG'] = trend.VortexIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).vortex_indicator_neg()
    result['TA2_VORTEX_POS'] = trend.VortexIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).vortex_indicator_pos()
    result['TA2_VORTEX_DIFF'] = trend.VortexIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).vortex_indicator_diff()
    result['TA2_WMA'] = trend.WMAIndicator(close=df['Close'], window=20).wma()

    # Momentum Indicators
    momentum = ta.momentum
    result['TA2_KAMA'] = momentum.KAMAIndicator(close=df['Close'], window=10, pow1=2, pow2=30).kama()
    result['TA2_AWESOME_OSC'] = momentum.AwesomeOscillatorIndicator(high=df['High'], low=df['Low'], window1=5, window2=34).awesome_oscillator()
    result['TA2_PPO'] = momentum.PercentagePriceOscillator(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).ppo()
    result['TA2_PPO_HIST'] = momentum.PercentagePriceOscillator(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).ppo_hist()
    result['TA2_PPO_SIGNAL'] = momentum.PercentagePriceOscillator(close=df['Close'], window_slow=26, window_fast=12, window_sign=9).ppo_signal()
    result['TA2_PVO'] = momentum.PercentageVolumeOscillator(volume=df['Volume'], window_slow=26, window_fast=12, window_sign=9).pvo()
    result['TA2_PVO_HIST'] = momentum.PercentageVolumeOscillator(volume=df['Volume'], window_slow=26, window_fast=12, window_sign=9).pvo_hist()
    result['TA2_PVO_SIGNAL'] = momentum.PercentageVolumeOscillator(volume=df['Volume'], window_slow=26, window_fast=12, window_sign=9).pvo_signal()
    result['TA2_ROC'] = momentum.ROCIndicator(close=df['Close'], window=12).roc()
    result['TA2_RSI'] = momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    result['TA2_STOCH_RSI'] = momentum.StochRSIIndicator(close=df['Close'], window=14, smooth1=3, smooth2=3).stochrsi()
    result['TA2_STOCH_RSI_D'] = momentum.StochRSIIndicator(close=df['Close'], window=14, smooth1=3, smooth2=3).stochrsi_d()
    result['TA2_STOCH_RSI_K'] = momentum.StochRSIIndicator(close=df['Close'], window=14, smooth1=3, smooth2=3).stochrsi_k()
    result['TA2_STOCH'] = momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3).stoch()
    result['TA2_STOCH_SIGNAL'] = momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3).stoch_signal()
    result['TA2_TSI'] = momentum.TSIIndicator(close=df['Close'], window_slow=25, window_fast=13).tsi()
    result['TA2_ULT_OSC'] = momentum.UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close'], window1=7, window2=14, window3=28, weight1=4.0, weight2=2.0, weight3=1.0).ultimate_oscillator()
    result['TA2_WILLIAMS_R'] = momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()

    # Volatility Indicators
    volatility = ta.volatility
    result['TA2_ATR'] = volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    result['TA2_BOLLINGER_H'] = volatility.BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_hband()
    result['TA2_BOLLINGER_H_IND'] = volatility.BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_hband_indicator()
    result['TA2_BOLLINGER_L'] = volatility.BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_lband()
    result['TA2_BOLLINGER_L_IND'] = volatility.BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_lband_indicator()
    result['TA2_BOLLINGER_M'] = volatility.BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_mavg()
    result['TA2_BOLLINGER_PBAND'] = volatility.BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_pband()
    result['TA2_BOLLINGER_WBAND'] = volatility.BollingerBands(close=df['Close'], window=20, window_dev=2).bollinger_wband()
    result['TA2_DONCHIAN_H'] = volatility.DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20).donchian_channel_hband()
    result['TA2_DONCHIAN_L'] = volatility.DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20).donchian_channel_lband()
    result['TA2_DONCHIAN_M'] = volatility.DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20).donchian_channel_mband()
    result['TA2_DONCHIAN_PBAND'] = volatility.DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20).donchian_channel_pband()
    result['TA2_DONCHIAN_WBAND'] = volatility.DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20).donchian_channel_wband()
    result['TA2_KELTNER_H'] = volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10).keltner_channel_hband()
    result['TA2_KELTNER_H_IND'] = volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10).keltner_channel_hband_indicator()
    result['TA2_KELTNER_L'] = volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10).keltner_channel_lband()
    result['TA2_KELTNER_L_IND'] = volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10).keltner_channel_lband_indicator()
    result['TA2_KELTNER_M'] = volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10).keltner_channel_mband()
    result['TA2_KELTNER_PBAND'] = volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10).keltner_channel_pband()
    result['TA2_KELTNER_WBAND'] = volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10).keltner_channel_wband()

    # Volume Indicators
    volume = ta.volume
    result['TA2_ACC_DIST'] = volume.AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).acc_dist_index()
    result['TA2_CHAIKIN_MF'] = volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20).chaikin_money_flow()
    result['TA2_EOM'] = volume.EaseOfMovementIndicator(high=df['High'], low=df['Low'], volume=df['Volume'], window=14).ease_of_movement()
    result['TA2_EOM_SMA'] = volume.EaseOfMovementIndicator(high=df['High'], low=df['Low'], volume=df['Volume'], window=14).sma_ease_of_movement()
    result['TA2_FI'] = volume.ForceIndexIndicator(close=df['Close'], volume=df['Volume'], window=13).force_index()
    result['TA2_MFI'] = volume.MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14).money_flow_index()
    result['TA2_NEG_VOL'] = volume.NegativeVolumeIndexIndicator(close=df['Close'], volume=df['Volume']).negative_volume_index()
    result['TA2_OBV'] = volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    result['TA2_VOL_PRICE_TREND'] = volume.VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume']).volume_price_trend()
    result['TA2_VOL_WEIGHTED_AVG'] = volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14).volume_weighted_average_price()

    # Others
    others = ta.others
    result['TA2_CUM_RETURN'] = others.CumulativeReturnIndicator(close=df['Close']).cumulative_return()
    result['TA2_DAILY_LOG_RETURN'] = others.DailyLogReturnIndicator(close=df['Close']).daily_log_return()
    result['TA2_DAILY_RETURN'] = others.DailyReturnIndicator(close=df['Close']).daily_return()

    return result

def _compute_custom_indicators(df: pd.DataFrame, multi_data: Optional[List[pd.DataFrame]] = None) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    
    # VWAP Manual
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    result['vwap_manual'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

    # Cumulative Delta
    result['delta'] = np.where(df['Close'] > df['Open'], df['Volume'], -df['Volume'])
    result['cum_delta'] = result['delta'].cumsum()

    # Swing Highs/Lows
    result['swing_high_20'] = df['High'].rolling(20).max()
    result['swing_low_20'] = df['Low'].rolling(20).min()
    result['swing_high_50'] = df['High'].rolling(50).max()
    result['swing_low_50'] = df['Low'].rolling(50).min()

    # Fibonacci Retracement (0.618 level)
    result['fib_618'] = result['swing_high_50'] - (result['swing_high_50'] - result['swing_low_50']) * 0.618

    # Heikin-Ashi
    result['ha_close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4.0
    ha_open = pd.Series(np.nan, index=df.index)
    ha_open.iloc[0] = df['Open'].iloc[0]
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + result['ha_close'].iloc[i-1]) / 2.0
    result['ha_open'] = ha_open
    result['ha_high'] = pd.concat([df['High'], result['ha_open'], result['ha_close']], axis=1).max(axis=1)
    result['ha_low'] = pd.concat([df['Low'], result['ha_open'], result['ha_close']], axis=1).min(axis=1)

    # Time-Based Columns
    result['hour'] = df.index.hour
    result['dayofweek'] = df.index.dayofweek
    result['month'] = df.index.month

    # Multi-Asset Indicators
    if multi_data and len(multi_data) > 0:
        for i, md in enumerate(multi_data):
            result[f'roc_14_asset_{i}'] = (md['Close'] / md['Close'].shift(14) - 1) * 100 if talib is None else talib.ROC(md['Close'], timeperiod=14)
            result[f'roc_30_asset_{i}'] = (md['Close'] / md['Close'].shift(30) - 1) * 100 if talib is None else talib.ROC(md['Close'], timeperiod=30)
            result[f'roc_90_asset_{i}'] = (md['Close'] / md['Close'].shift(90) - 1) * 100 if talib is None else talib.ROC(md['Close'], timeperiod=90)
            result[f'corr_20_asset_{i}'] = df['Close'].rolling(20).corr(md['Close'])
            spread = df['Close'] - md['Close']
            result[f'spread_asset_{i}'] = spread
            result[f'zscore_asset_{i}'] = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()

    return result

def get_all_indicators(df: pd.DataFrame, include_price=False, multi_data: Optional[List[pd.DataFrame]] = None) -> pd.DataFrame:
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    frames: List[pd.DataFrame] = [
        _compute_talib_indicators(df),
        _compute_pandasta_indicators(df),
        _compute_ta_indicators(df),
        _compute_custom_indicators(df, multi_data)
    ]
    result = pd.concat(frames, axis=1)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    if include_price:
        result = pd.concat([result, df[['Open', 'High', 'Low', 'Close', 'Volume']]], axis=1)
    return result.ffill().fillna(0)  # Forward fill then 0 for remaining NaNs