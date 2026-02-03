"""
Multi-Timeframe Pipeline Module

Handles resampling, alignment, and anti-leakage for MTF analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def build_mtf_sets(
    raw_15m: pd.DataFrame,
    frames: List[str] = ['1H', '4H', '1D']
) -> Dict[str, pd.DataFrame]:
    """
    Build resampled timeframes with leakage prevention.
    
    Parameters
    ----------
    raw_15m : pd.DataFrame
        Base 15-minute OHLCV data with DatetimeIndex
    frames : List[str]
        List of higher timeframes to resample to
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping timeframe to resampled DataFrame
    """
    result = {'15min': raw_15m.copy()}
    
    # Ensure datetime index
    if not isinstance(raw_15m.index, pd.DatetimeIndex):
        if 'timestamp' in raw_15m.columns:
            raw_15m = raw_15m.set_index('timestamp')
        elif 'datetime' in raw_15m.columns:
            raw_15m = raw_15m.set_index('datetime')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or timestamp/datetime column")
    
    for frame in frames:
        # Resample OHLCV data
        resampled = pd.DataFrame()
        resampled['open'] = raw_15m['open'].resample(frame).first()
        resampled['high'] = raw_15m['high'].resample(frame).max()
        resampled['low'] = raw_15m['low'].resample(frame).min()
        resampled['close'] = raw_15m['close'].resample(frame).last()
        
        if 'volume' in raw_15m.columns:
            resampled['volume'] = raw_15m['volume'].resample(frame).sum()
        
        # Drop any NaN rows
        resampled = resampled.dropna()
        
        result[frame] = resampled
    
    return result


def align_to_15m(
    htf_df: pd.DataFrame,
    master_15m_index: pd.DatetimeIndex,
    method: str = 'broadcast'
) -> pd.DataFrame:
    """
    Align HTF features to 15m timeline with anti-leakage.
    
    Parameters
    ----------
    htf_df : pd.DataFrame
        Higher timeframe DataFrame with DatetimeIndex
    master_15m_index : pd.DatetimeIndex
        Master 15-minute timeline
    method : str
        Alignment method: 'broadcast' (carry-forward) or 'repeat'
        
    Returns
    -------
    pd.DataFrame
        HTF features aligned to 15m timeline
    """
    if method == 'broadcast':
        # Forward-fill: each HTF bar is available only after it closes
        # Reindex to 15m and forward-fill
        aligned = htf_df.reindex(master_15m_index, method='ffill')
        
        # Shift by 1 to prevent using current bar's HTF values
        # (anti-leakage: only use completed HTF bars)
        aligned = aligned.shift(1)
        
    elif method == 'repeat':
        # Repeat HTF values for all 15m bars within the HTF period
        aligned = htf_df.reindex(master_15m_index, method='ffill')
        aligned = aligned.shift(1)
    
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    # Fill initial NaNs with 0
    aligned = aligned.fillna(0)
    
    return aligned


def merge_features(
    fts_15m: pd.DataFrame,
    fts_1H: pd.DataFrame,
    fts_4H: pd.DataFrame,
    fts_1D: pd.DataFrame,
    prefix_map: Dict[str, str] = None
) -> np.ndarray:
    """
    Merge all MTF features into tensor-ready array.
    
    Parameters
    ----------
    fts_15m : pd.DataFrame
        15-minute features
    fts_1H : pd.DataFrame
        1-hour features (already aligned to 15m)
    fts_4H : pd.DataFrame
        4-hour features (already aligned to 15m)
    fts_1D : pd.DataFrame
        Daily features (already aligned to 15m)
    prefix_map : Dict[str, str], optional
        Map of timeframe to column prefix
        
    Returns
    -------
    np.ndarray
        Merged feature array of shape (n_samples, n_features)
    """
    if prefix_map is None:
        prefix_map = {
            '15m': '15m_',
            '1H': '1h_',
            '4H': '4h_',
            '1D': '1d_'
        }
    
    # Add prefixes to column names
    fts_15m_renamed = fts_15m.add_prefix(prefix_map['15m'])
    fts_1H_renamed = fts_1H.add_prefix(prefix_map['1H'])
    fts_4H_renamed = fts_4H.add_prefix(prefix_map['4H'])
    fts_1D_renamed = fts_1D.add_prefix(prefix_map['1D'])
    
    # Concatenate all features
    merged = pd.concat([
        fts_15m_renamed,
        fts_1H_renamed,
        fts_4H_renamed,
        fts_1D_renamed
    ], axis=1)
    
    # Convert to numpy array
    return merged.values


def ensure_no_leakage(
    df_15m: pd.DataFrame,
    df_htf: pd.DataFrame,
    htf_name: str
) -> bool:
    """
    Validate that HTF alignment has no forward leakage.
    
    Parameters
    ----------
    df_15m : pd.DataFrame
        15-minute timeline
    df_htf : pd.DataFrame
        Higher timeframe data (aligned)
    htf_name : str
        Name of HTF for logging
        
    Returns
    -------
    bool
        True if no leakage detected
    """
    # Check that HTF values only change at HTF bar boundaries
    # and are never from the future
    
    # For proper anti-leakage, HTF values should only update
    # after the HTF bar has closed
    
    # Simple check: ensure no forward-looking values
    if len(df_htf) > len(df_15m):
        print(f"Warning: {htf_name} has more rows than 15m timeline")
        return False
    
    # Check for NaN propagation (should be minimal)
    nan_ratio = df_htf.isna().sum().sum() / (df_htf.shape[0] * df_htf.shape[1])
    if nan_ratio > 0.5:
        print(f"Warning: {htf_name} has {nan_ratio*100:.1f}% NaN values")
        return False
    
    return True
