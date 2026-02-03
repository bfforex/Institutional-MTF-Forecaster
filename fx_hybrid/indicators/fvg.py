"""
Fair Value Gap (FVG) Detection Module

Implements deterministic FVG detection with institutional logic including:
- Bullish/Bearish FVG identification
- Displacement filtering using ATR
- Gap size and age tracking
- Partial fill ratio and validity tracking
"""

import numpy as np
import pandas as pd
from typing import Optional


def detect_fvg(
    df: pd.DataFrame,
    k: float = 2.0,
    atr_len: int = 14,
    min_gap_pips: float = 2.0,
    max_age: int = 200,
    displacement: str = 'atr',
    pip_size: float = 0.0001
) -> pd.DataFrame:
    """
    Detect Fair Value Gaps with institutional logic.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume
    k : float
        Displacement multiplier for ATR filter (default: 2.0)
    atr_len : int
        ATR period for displacement calculation (default: 14)
    min_gap_pips : float
        Minimum gap size in pips (default: 2.0)
    max_age : int
        Maximum age of FVG in bars before expiry (default: 200)
    displacement : str
        Displacement method: 'atr' or 'zscore' (default: 'atr')
    pip_size : float
        Pip size for the instrument (default: 0.0001 for forex majors)
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added FVG columns:
        - fvg_has_bull, fvg_has_bear (bool)
        - fvg_gap_pips, fvg_gap_zscore
        - fvg_partial_fill_ratio, fvg_age_bars
        - dist_to_nearest_bull_fvg, dist_to_nearest_bear_fvg (pips & normalized)
        - fvg_context (categorical: fresh/partially_filled/mitigated)
    """
    result = df.copy()
    
    # Calculate ATR for displacement filter
    high = result['high'].values
    low = result['low'].values
    close = result['close'].values
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar uses high-low
    
    # ATR using simple moving average
    atr = pd.Series(tr).rolling(window=atr_len, min_periods=1).mean().values
    
    # Calculate candle range
    candle_range = high - low
    
    # Initialize FVG arrays
    n_bars = len(result)
    fvg_has_bull = np.zeros(n_bars, dtype=bool)
    fvg_has_bear = np.zeros(n_bars, dtype=bool)
    fvg_gap_pips = np.zeros(n_bars)
    fvg_gap_zscore = np.zeros(n_bars)
    fvg_partial_fill_ratio = np.zeros(n_bars)
    fvg_age_bars = np.zeros(n_bars)
    dist_to_nearest_bull_fvg = np.full(n_bars, np.inf)
    dist_to_nearest_bear_fvg = np.full(n_bars, np.inf)
    fvg_context = np.full(n_bars, '', dtype='U20')
    
    # Track active FVGs
    active_bull_fvgs = []  # List of (bar_idx, gap_top, gap_bottom, gap_size)
    active_bear_fvgs = []
    
    # Detect FVGs (start from bar 2 to have n-1, n, n+1)
    for i in range(1, n_bars - 1):
        # Check for displacement bar
        displacement_valid = False
        if displacement == 'atr':
            displacement_valid = candle_range[i] >= k * atr[i]
        elif displacement == 'zscore':
            # Z-score based displacement
            mean_range = np.mean(candle_range[max(0, i-20):i+1])
            std_range = np.std(candle_range[max(0, i-20):i+1])
            if std_range > 0:
                zscore = (candle_range[i] - mean_range) / std_range
                displacement_valid = zscore >= k
        
        if not displacement_valid:
            continue
        
        # Bullish FVG: low[i+1] > high[i-1]
        if low[i+1] > high[i-1]:
            gap_size = low[i+1] - high[i-1]
            gap_pips = gap_size / pip_size
            
            if gap_pips >= min_gap_pips:
                fvg_has_bull[i+1] = True
                fvg_gap_pips[i+1] = gap_pips
                fvg_age_bars[i+1] = 0
                fvg_context[i+1] = 'fresh'
                
                # Add to active FVGs
                active_bull_fvgs.append({
                    'created_bar': i+1,
                    'gap_top': low[i+1],
                    'gap_bottom': high[i-1],
                    'gap_size': gap_size,
                    'filled': False
                })
        
        # Bearish FVG: high[i+1] < low[i-1]
        if high[i+1] < low[i-1]:
            gap_size = low[i-1] - high[i+1]
            gap_pips = gap_size / pip_size
            
            if gap_pips >= min_gap_pips:
                fvg_has_bear[i+1] = True
                fvg_gap_pips[i+1] = gap_pips
                fvg_age_bars[i+1] = 0
                fvg_context[i+1] = 'fresh'
                
                # Add to active FVGs
                active_bear_fvgs.append({
                    'created_bar': i+1,
                    'gap_top': low[i-1],
                    'gap_bottom': high[i+1],
                    'gap_size': gap_size,
                    'filled': False
                })
    
    # Track FVG evolution and fill status
    for i in range(n_bars):
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        
        # Update bullish FVGs
        min_bull_dist = np.inf
        for fvg in active_bull_fvgs:
            if fvg['filled']:
                continue
            
            age = i - fvg['created_bar']
            if age > max_age:
                fvg['filled'] = True
                continue
            
            # Check if price entered the gap
            gap_bottom = fvg['gap_bottom']
            gap_top = fvg['gap_top']
            
            if current_close <= gap_top and current_close >= gap_bottom:
                # Calculate partial fill
                fill_amount = gap_top - current_close
                fvg['partial_fill'] = fill_amount / fvg['gap_size']
                
                if current_close <= gap_bottom:
                    fvg['filled'] = True
                    fvg_context[i] = 'mitigated'
                else:
                    fvg_context[i] = 'partially_filled'
            
            # Calculate distance to FVG
            if current_close < gap_bottom:
                dist = gap_bottom - current_close
                min_bull_dist = min(min_bull_dist, dist)
        
        dist_to_nearest_bull_fvg[i] = min_bull_dist / pip_size if min_bull_dist != np.inf else 0
        
        # Update bearish FVGs
        min_bear_dist = np.inf
        for fvg in active_bear_fvgs:
            if fvg['filled']:
                continue
            
            age = i - fvg['created_bar']
            if age > max_age:
                fvg['filled'] = True
                continue
            
            # Check if price entered the gap
            gap_bottom = fvg['gap_bottom']
            gap_top = fvg['gap_top']
            
            if current_close >= gap_bottom and current_close <= gap_top:
                # Calculate partial fill
                fill_amount = current_close - gap_bottom
                fvg['partial_fill'] = fill_amount / fvg['gap_size']
                
                if current_close >= gap_top:
                    fvg['filled'] = True
                    fvg_context[i] = 'mitigated'
                else:
                    fvg_context[i] = 'partially_filled'
            
            # Calculate distance to FVG
            if current_close > gap_top:
                dist = current_close - gap_top
                min_bear_dist = min(min_bear_dist, dist)
        
        dist_to_nearest_bear_fvg[i] = min_bear_dist / pip_size if min_bear_dist != np.inf else 0
    
    # Add columns to result
    result['fvg_has_bull'] = fvg_has_bull
    result['fvg_has_bear'] = fvg_has_bear
    result['fvg_gap_pips'] = fvg_gap_pips
    result['fvg_gap_zscore'] = fvg_gap_zscore
    result['fvg_partial_fill_ratio'] = fvg_partial_fill_ratio
    result['fvg_age_bars'] = fvg_age_bars
    result['dist_to_nearest_bull_fvg'] = dist_to_nearest_bull_fvg
    result['dist_to_nearest_bear_fvg'] = dist_to_nearest_bear_fvg
    result['fvg_context'] = fvg_context
    
    return result
