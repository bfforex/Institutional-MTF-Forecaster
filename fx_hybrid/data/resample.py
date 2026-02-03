"""
Data Resampling Utilities

Handles OHLCV resampling from 15m to higher timeframes.
"""

import pandas as pd
from typing import Dict, List


def resample_ohlcv(
    df: pd.DataFrame,
    target_freq: str,
    calendar_aware: bool = True
) -> pd.DataFrame:
    """
    Resample OHLCV data to target frequency.
    
    Parameters
    ----------
    df : pd.DataFrame
        Source OHLCV DataFrame with DatetimeIndex
    target_freq : str
        Target frequency (e.g., '1H', '4H', '1D')
    calendar_aware : bool
        Whether to use calendar-aware resampling
        
    Returns
    -------
    pd.DataFrame
        Resampled OHLCV data
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Resample OHLCV
    resampled = pd.DataFrame()
    
    resampled['open'] = df['open'].resample(target_freq).first()
    resampled['high'] = df['high'].resample(target_freq).max()
    resampled['low'] = df['low'].resample(target_freq).min()
    resampled['close'] = df['close'].resample(target_freq).last()
    
    if 'volume' in df.columns:
        resampled['volume'] = df['volume'].resample(target_freq).sum()
    
    # Drop NaN rows (incomplete bars)
    resampled = resampled.dropna()
    
    return resampled


def generate_from_raw(
    raw_15m: pd.DataFrame,
    timeframes: List[str] = ['1H', '4H', '1D']
) -> Dict[str, pd.DataFrame]:
    """
    Generate multiple timeframes from raw 15m data.
    
    Parameters
    ----------
    raw_15m : pd.DataFrame
        Raw 15-minute OHLCV data
    timeframes : List[str]
        List of target timeframes
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of resampled dataframes
    """
    result = {'15min': raw_15m.copy()}
    
    for tf in timeframes:
        result[tf] = resample_ohlcv(raw_15m, tf)
    
    return result
