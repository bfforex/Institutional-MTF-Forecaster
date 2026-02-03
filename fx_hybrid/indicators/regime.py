"""
Market Regime Detection Module

Classifies market conditions for model conditioning:
- Trending Up / Trending Down (ADX > threshold, slope check)
- Ranging (ADX < threshold)
- Low Volatility / High Volatility (ATR percentile)
"""

import numpy as np
import pandas as pd


def detect_regime(
    df: pd.DataFrame,
    lookback: int = 100,
    adx_threshold: float = 25
) -> pd.DataFrame:
    """
    Classify market regime for conditioning.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    lookback : int
        Lookback period for regime classification
    adx_threshold : float
        ADX threshold for trending vs ranging
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regime features
    """
    result = df.copy()
    
    high = result['high'].values
    low = result['low'].values
    close = result['close'].values
    
    # Calculate ADX
    adx = calculate_adx(high, low, close, period=14)
    
    # Calculate ATR percentile
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
    
    atr_percentile = np.zeros(len(result))
    for i in range(lookback, len(result)):
        atr_percentile[i] = (atr[i] > np.percentile(atr[i-lookback:i], 75)) * 1.0
    
    # Determine trend direction using EMA slope
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    
    trend_direction = np.where(ema_20 > ema_50, 1, -1)
    
    # Classify regime
    regime = np.zeros(len(result), dtype='U20')
    regime_numeric = np.zeros(len(result))
    
    for i in range(len(result)):
        if adx[i] > adx_threshold:
            if trend_direction[i] > 0:
                regime[i] = 'trending_up'
                regime_numeric[i] = 2
            else:
                regime[i] = 'trending_down'
                regime_numeric[i] = -2
        else:
            regime[i] = 'ranging'
            regime_numeric[i] = 0
    
    # Volatility regime
    volatility_regime = np.where(atr_percentile > 0, 'high_vol', 'low_vol')
    
    result['regime'] = regime
    result['regime_numeric'] = regime_numeric
    result['volatility_regime'] = volatility_regime
    result['adx'] = adx
    result['atr_percentile'] = atr_percentile
    
    return result


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth using Wilder's smoothing
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Calculate ADX
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    
    return adx
