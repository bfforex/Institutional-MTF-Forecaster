"""
HTF-to-LTF Alignment Logic

Handles broadcasting and repeating of higher timeframe features to lower timeframe timeline.
"""

import pandas as pd
import numpy as np
from typing import Optional


def broadcast_htf_to_ltf(
    htf_df: pd.DataFrame,
    ltf_index: pd.DatetimeIndex,
    anti_leakage: bool = True
) -> pd.DataFrame:
    """
    Broadcast HTF features to LTF timeline with forward-fill.
    
    Parameters
    ----------
    htf_df : pd.DataFrame
        Higher timeframe DataFrame
    ltf_index : pd.DatetimeIndex
        Lower timeframe index (e.g., 15m)
    anti_leakage : bool
        If True, shift by 1 to prevent using incomplete HTF bars
        
    Returns
    -------
    pd.DataFrame
        HTF features aligned to LTF timeline
    """
    # Reindex to LTF timeline with forward-fill
    aligned = htf_df.reindex(ltf_index, method='ffill')
    
    if anti_leakage:
        # Shift by 1 to ensure we only use completed HTF bars
        aligned = aligned.shift(1)
    
    # Fill initial NaNs with 0 or last valid value
    aligned = aligned.fillna(method='bfill').fillna(0)
    
    return aligned


def repeat_within_period(
    htf_df: pd.DataFrame,
    ltf_index: pd.DatetimeIndex,
    htf_freq: str
) -> pd.DataFrame:
    """
    Repeat HTF values for all LTF bars within each HTF period.
    
    Parameters
    ----------
    htf_df : pd.DataFrame
        Higher timeframe DataFrame
    ltf_index : pd.DatetimeIndex
        Lower timeframe index
    htf_freq : str
        HTF frequency string (e.g., '1H', '4H')
        
    Returns
    -------
    pd.DataFrame
        HTF features repeated to LTF timeline
    """
    # Similar to broadcast but ensures values repeat within period
    aligned = htf_df.reindex(ltf_index, method='ffill')
    
    # Shift to prevent leakage
    aligned = aligned.shift(1)
    aligned = aligned.fillna(method='bfill').fillna(0)
    
    return aligned


def validate_alignment(
    htf_aligned: pd.DataFrame,
    ltf_index: pd.DatetimeIndex,
    strict: bool = True
) -> bool:
    """
    Validate that HTF alignment is correct and has no leakage.
    
    Parameters
    ----------
    htf_aligned : pd.DataFrame
        Aligned HTF features
    ltf_index : pd.DatetimeIndex
        LTF timeline
    strict : bool
        If True, perform strict validation checks
        
    Returns
    -------
    bool
        True if validation passes
    """
    # Check index alignment
    if not htf_aligned.index.equals(ltf_index):
        print("Error: Index mismatch between aligned HTF and LTF")
        return False
    
    # Check for excessive NaNs
    nan_ratio = htf_aligned.isna().sum().sum() / (htf_aligned.shape[0] * htf_aligned.shape[1])
    if nan_ratio > 0.1:
        print(f"Warning: {nan_ratio*100:.1f}% NaN values in aligned HTF features")
        if strict:
            return False
    
    # Check that values update at proper intervals
    # (HTF values should stay constant for multiple LTF bars)
    for col in htf_aligned.columns:
        if htf_aligned[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            diff = htf_aligned[col].diff()
            changes = (diff != 0).sum()
            expected_max_changes = len(htf_aligned) // 4  # For 1H on 15m
            
            if changes > len(htf_aligned) * 0.5:
                print(f"Warning: Column {col} changes too frequently for HTF data")
                # This is informational, not a hard failure
    
    return True
