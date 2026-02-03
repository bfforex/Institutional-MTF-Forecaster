"""
Confluence Logic Module

Implements SMC/FVG confluence rules for trade filtering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class ConfluenceFilter:
    """
    Filter trades based on SMC/FVG confluence conditions.
    """
    
    def __init__(self, config: Dict):
        """
        Parameters
        ----------
        config : Dict
            Confluence configuration with keys:
            - htf_bias_tf: List of HTF timeframes to check
            - ltf_confirm_tf: List of LTF timeframes to check
            - max_dist_fvg_pips: Max distance to FVG
            - choch_bos_lookback_bars: Lookback for CHOCH/BOS
            - ob_mitigation_lookback_bars: Lookback for OB mitigation
            - require_all: Whether all conditions must be met
        """
        self.config = config
        self.htf_bias_tf = config.get('htf_bias_tf', ['4H', '1D'])
        self.ltf_confirm_tf = config.get('ltf_confirm_tf', ['15m'])
        self.max_dist_fvg_pips = config.get('max_dist_fvg_pips', 5)
        self.choch_bos_lookback = config.get('choch_bos_lookback_bars', 20)
        self.ob_mitigation_lookback = config.get('ob_mitigation_lookback_bars', 60)
        self.require_all = config.get('require_all', False)
    
    def check_long_confluence(
        self,
        features: pd.DataFrame,
        bar_idx: int
    ) -> Tuple[bool, Dict]:
        """
        Check confluence for long entry.
        
        Conditions:
        1. HTF trend bias = +1
        2. LTF bullish CHOCH or BOS within lookback
        3. Fresh/active bullish FVG nearby OR bullish OB mitigation
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with all indicators
        bar_idx : int
            Current bar index
            
        Returns
        -------
        Tuple[bool, Dict]
            (confluence_met, details_dict)
        """
        details = {
            'htf_bias': False,
            'ltf_structure': False,
            'fvg_or_ob': False
        }
        
        if bar_idx >= len(features):
            return False, details
        
        # Check 1: HTF trend bias
        if 'trend_bias' in features.columns:
            htf_bias = features.iloc[bar_idx]['trend_bias']
            details['htf_bias'] = htf_bias > 0
        
        # Check 2: LTF structure (CHOCH or BOS)
        ltf_structure = False
        if 'choch_age' in features.columns and 'bos_age' in features.columns:
            choch_age = features.iloc[bar_idx]['choch_age']
            bos_age = features.iloc[bar_idx]['bos_age']
            
            # Check if recent bullish CHOCH or BOS
            ltf_structure = (choch_age <= self.choch_bos_lookback or 
                           bos_age <= self.choch_bos_lookback)
        
        details['ltf_structure'] = ltf_structure
        
        # Check 3: FVG or OB
        fvg_or_ob = False
        
        # Check for nearby bullish FVG
        if 'dist_to_nearest_bull_fvg' in features.columns:
            dist_fvg = features.iloc[bar_idx]['dist_to_nearest_bull_fvg']
            if dist_fvg <= self.max_dist_fvg_pips and dist_fvg > 0:
                fvg_or_ob = True
        
        # Or check for bullish OB mitigation
        if not fvg_or_ob and 'dist_to_ob_up' in features.columns:
            dist_ob = features.iloc[bar_idx]['dist_to_ob_up']
            if dist_ob <= self.max_dist_fvg_pips:
                fvg_or_ob = True
        
        details['fvg_or_ob'] = fvg_or_ob
        
        # Decide if confluence met
        if self.require_all:
            confluence_met = all(details.values())
        else:
            confluence_met = sum(details.values()) >= 2  # At least 2 conditions
        
        return confluence_met, details
    
    def check_short_confluence(
        self,
        features: pd.DataFrame,
        bar_idx: int
    ) -> Tuple[bool, Dict]:
        """
        Check confluence for short entry.
        
        Conditions:
        1. HTF trend bias = -1
        2. LTF bearish CHOCH or BOS within lookback
        3. Fresh/active bearish FVG nearby OR bearish OB mitigation
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame
        bar_idx : int
            Current bar index
            
        Returns
        -------
        Tuple[bool, Dict]
            (confluence_met, details_dict)
        """
        details = {
            'htf_bias': False,
            'ltf_structure': False,
            'fvg_or_ob': False
        }
        
        if bar_idx >= len(features):
            return False, details
        
        # Check 1: HTF trend bias
        if 'trend_bias' in features.columns:
            htf_bias = features.iloc[bar_idx]['trend_bias']
            details['htf_bias'] = htf_bias < 0
        
        # Check 2: LTF structure
        ltf_structure = False
        if 'choch_age' in features.columns and 'bos_age' in features.columns:
            choch_age = features.iloc[bar_idx]['choch_age']
            bos_age = features.iloc[bar_idx]['bos_age']
            
            ltf_structure = (choch_age <= self.choch_bos_lookback or 
                           bos_age <= self.choch_bos_lookback)
        
        details['ltf_structure'] = ltf_structure
        
        # Check 3: FVG or OB
        fvg_or_ob = False
        
        # Check for nearby bearish FVG
        if 'dist_to_nearest_bear_fvg' in features.columns:
            dist_fvg = features.iloc[bar_idx]['dist_to_nearest_bear_fvg']
            if dist_fvg <= self.max_dist_fvg_pips and dist_fvg > 0:
                fvg_or_ob = True
        
        # Or check for bearish OB mitigation
        if not fvg_or_ob and 'dist_to_ob_dn' in features.columns:
            dist_ob = features.iloc[bar_idx]['dist_to_ob_dn']
            if dist_ob <= self.max_dist_fvg_pips:
                fvg_or_ob = True
        
        details['fvg_or_ob'] = fvg_or_ob
        
        # Decide if confluence met
        if self.require_all:
            confluence_met = all(details.values())
        else:
            confluence_met = sum(details.values()) >= 2
        
        return confluence_met, details


def apply_confluence_filter(
    predictions: np.ndarray,
    features: pd.DataFrame,
    config: Dict,
    threshold: float = 0.55
) -> np.ndarray:
    """
    Apply confluence filtering to predictions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (probabilities)
    features : pd.DataFrame
        Feature DataFrame with indicators
    config : Dict
        Confluence configuration
    threshold : float
        Probability threshold
        
    Returns
    -------
    np.ndarray
        Filtered predictions (positions: -1, 0, +1)
    """
    confluence_filter = ConfluenceFilter(config)
    positions = np.zeros(len(predictions))
    
    for i in range(len(predictions)):
        max_prob = predictions[i].max()
        pred_class = predictions[i].argmax()
        
        if max_prob < threshold:
            continue
        
        if pred_class == 2:  # Up
            long_confluence, _ = confluence_filter.check_long_confluence(features, i)
            if long_confluence:
                positions[i] = 1
        
        elif pred_class == 0:  # Down
            short_confluence, _ = confluence_filter.check_short_confluence(features, i)
            if short_confluence:
                positions[i] = -1
    
    return positions
