"""
MTF Feature Builder

Wraps indicators and builds complete feature matrices for all timeframes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
sys.path.append('..')

from ..indicators.fvg import detect_fvg
from ..indicators.smc import compute_smc_features
from ..indicators.regime import detect_regime


def build_features_for_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    fvg_config: Optional[Dict] = None,
    smc_config: Optional[Dict] = None,
    regime_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Build complete feature set for a single timeframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data for the timeframe
    timeframe : str
        Timeframe identifier (e.g., '15min', '1H')
    fvg_config : Dict, optional
        FVG configuration parameters
    smc_config : Dict, optional
        SMC configuration parameters
    regime_config : Dict, optional
        Regime configuration parameters
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all features
    """
    result = df.copy()
    
    # Set defaults
    if fvg_config is None:
        fvg_config = {
            'k': 2.0,
            'atr_len': 14,
            'min_gap_pips': 2.0,
            'max_age': 200,
            'displacement': 'atr'
        }
    
    if smc_config is None:
        smc_config = {
            'swing_w': 3,
            'min_exc_atr': 0.5,
            'equal_band_pips': 2.0,
            'wick_ratio': 0.6,
            'disp_k': 2.0,
            'max_ob_boxes': 3
        }
    
    if regime_config is None:
        regime_config = {
            'lookback': 100,
            'adx_threshold': 25
        }
    
    # Compute FVG features
    result = detect_fvg(result, **fvg_config)
    
    # Compute SMC features
    result = compute_smc_features(result, **smc_config)
    
    # Compute regime features
    if regime_config.get('enabled', True):
        result = detect_regime(result, **regime_config)
    
    return result


def select_features_for_ml(
    df: pd.DataFrame,
    feature_groups: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Select relevant features for ML model.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all computed features
    feature_groups : List[str], optional
        List of feature groups to include
        
    Returns
    -------
    pd.DataFrame
        Filtered feature DataFrame
    """
    if feature_groups is None:
        feature_groups = ['fvg', 'smc', 'regime', 'price']
    
    selected_cols = []
    
    # Price features (always include)
    if 'price' in feature_groups:
        price_cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            price_cols.append('volume')
        selected_cols.extend([c for c in price_cols if c in df.columns])
    
    # FVG features
    if 'fvg' in feature_groups:
        fvg_cols = [
            'fvg_has_bull', 'fvg_has_bear', 'fvg_gap_pips',
            'dist_to_nearest_bull_fvg', 'dist_to_nearest_bear_fvg'
        ]
        selected_cols.extend([c for c in fvg_cols if c in df.columns])
    
    # SMC features
    if 'smc' in feature_groups:
        smc_cols = [
            'trend_bias', 'bos_flag', 'choch_flag', 'bos_age', 'choch_age',
            'dist_to_ob_up', 'dist_to_ob_dn', 'pd_level', 'dist_to_eq_50'
        ]
        selected_cols.extend([c for c in smc_cols if c in df.columns])
    
    # Regime features
    if 'regime' in feature_groups:
        regime_cols = ['regime_numeric', 'adx', 'atr_percentile']
        selected_cols.extend([c for c in regime_cols if c in df.columns])
    
    # Remove duplicates while preserving order
    selected_cols = list(dict.fromkeys(selected_cols))
    
    return df[selected_cols]


def normalize_features(
    df: pd.DataFrame,
    method: str = 'zscore',
    exclude_binary: bool = True
) -> pd.DataFrame:
    """
    Normalize features for neural network input.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    method : str
        Normalization method: 'zscore', 'minmax', or 'robust'
    exclude_binary : bool
        If True, don't normalize binary/categorical features
        
    Returns
    -------
    pd.DataFrame
        Normalized features
    """
    result = df.copy()
    
    # Identify binary columns
    binary_cols = []
    if exclude_binary:
        for col in result.columns:
            unique_vals = result[col].nunique()
            if unique_vals <= 2:
                binary_cols.append(col)
    
    # Normalize non-binary columns
    for col in result.columns:
        if col in binary_cols:
            continue
        
        if method == 'zscore':
            mean = result[col].mean()
            std = result[col].std()
            if std > 0:
                result[col] = (result[col] - mean) / std
        
        elif method == 'minmax':
            min_val = result[col].min()
            max_val = result[col].max()
            if max_val > min_val:
                result[col] = (result[col] - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            median = result[col].median()
            q75 = result[col].quantile(0.75)
            q25 = result[col].quantile(0.25)
            iqr = q75 - q25
            if iqr > 0:
                result[col] = (result[col] - median) / iqr
    
    return result
