"""
Smart Money Concepts (SMC) Module

Implements comprehensive SMC indicator suite including:
- Swing structure (fractals)
- BOS (Break of Structure) and CHOCH (Change of Character)
- Order Blocks (OB)
- Liquidity Sweeps
- Premium/Discount (PD) Zones
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


def compute_structure(
    df: pd.DataFrame,
    swing_w: int = 3,
    min_exc_atr: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute swing structure, BOS and CHOCH events.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    swing_w : int
        Swing window (number of bars on each side)
    min_exc_atr : float
        Minimum excursion in ATR units for valid swing
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (swings, bos_events, choch_events)
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
    
    n_bars = len(df)
    
    # Detect swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(swing_w, n_bars - swing_w):
        # Check swing high
        is_swing_high = True
        for j in range(1, swing_w + 1):
            if high[i] <= high[i-j] or high[i] <= high[i+j]:
                is_swing_high = False
                break
        
        if is_swing_high:
            # Check minimum excursion
            if i > 0 and atr[i] > 0:
                excursion = high[i] - low[max(0, i-swing_w):i+swing_w+1].min()
                if excursion >= min_exc_atr * atr[i]:
                    swing_highs.append({'bar': i, 'price': high[i]})
        
        # Check swing low
        is_swing_low = True
        for j in range(1, swing_w + 1):
            if low[i] >= low[i-j] or low[i] >= low[i+j]:
                is_swing_low = False
                break
        
        if is_swing_low:
            # Check minimum excursion
            if i > 0 and atr[i] > 0:
                excursion = high[max(0, i-swing_w):i+swing_w+1].max() - low[i]
                if excursion >= min_exc_atr * atr[i]:
                    swing_lows.append({'bar': i, 'price': low[i]})
    
    # Create swings DataFrame
    swings = pd.DataFrame({
        'highs': swing_highs,
        'lows': swing_lows
    })
    
    # Detect BOS and CHOCH
    bos_events = []
    choch_events = []
    
    # Track trend state
    trend = 0  # 1 = bullish, -1 = bearish, 0 = undefined
    last_swing_high = None
    last_swing_low = None
    
    # Combine and sort swings
    all_swings = []
    for sh in swing_highs:
        all_swings.append(('high', sh['bar'], sh['price']))
    for sl in swing_lows:
        all_swings.append(('low', sl['bar'], sl['price']))
    all_swings.sort(key=lambda x: x[1])
    
    for i in range(len(all_swings)):
        swing_type, bar, price = all_swings[i]
        
        if swing_type == 'high':
            if last_swing_high is not None and last_swing_low is not None:
                if close[bar:].max() > last_swing_high['price']:
                    # Price broke above last high
                    if trend == 1:
                        # Continuation - BOS
                        bos_events.append({
                            'bar': bar,
                            'type': 'bullish',
                            'price': last_swing_high['price']
                        })
                    else:
                        # Reversal - CHOCH
                        choch_events.append({
                            'bar': bar,
                            'type': 'bullish',
                            'price': last_swing_high['price']
                        })
                        trend = 1
            last_swing_high = {'bar': bar, 'price': price}
        
        elif swing_type == 'low':
            if last_swing_low is not None and last_swing_high is not None:
                if close[bar:].min() < last_swing_low['price']:
                    # Price broke below last low
                    if trend == -1:
                        # Continuation - BOS
                        bos_events.append({
                            'bar': bar,
                            'type': 'bearish',
                            'price': last_swing_low['price']
                        })
                    else:
                        # Reversal - CHOCH
                        choch_events.append({
                            'bar': bar,
                            'type': 'bearish',
                            'price': last_swing_low['price']
                        })
                        trend = -1
            last_swing_low = {'bar': bar, 'price': price}
    
    bos_df = pd.DataFrame(bos_events) if bos_events else pd.DataFrame()
    choch_df = pd.DataFrame(choch_events) if choch_events else pd.DataFrame()
    
    return swings, bos_df, choch_df


def detect_order_blocks(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    bos: pd.DataFrame,
    disp_k: float = 2.0,
    max_boxes: int = 3
) -> pd.DataFrame:
    """
    Detect Order Blocks based on BOS events.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    swings : pd.DataFrame
        Swing structure from compute_structure
    bos : pd.DataFrame
        BOS events from compute_structure
    disp_k : float
        Displacement multiplier
    max_boxes : int
        Maximum number of order blocks to track
        
    Returns
    -------
    pd.DataFrame
        Order blocks with compact encoding
    """
    if len(bos) == 0:
        return pd.DataFrame()
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_price = df['open'].values
    
    order_blocks = []
    
    for _, bos_event in bos.iterrows():
        bos_bar = bos_event['bar']
        bos_type = bos_event['type']
        
        # Find the last opposite candle before BOS
        if bos_type == 'bullish':
            # Look for last bearish candle before bullish BOS
            for i in range(bos_bar - 1, max(0, bos_bar - 20), -1):
                if close[i] < open_price[i]:  # Bearish candle
                    order_blocks.append({
                        'bar': i,
                        'type': 'bullish_ob',
                        'center': (high[i] + low[i]) / 2,
                        'width': high[i] - low[i],
                        'top': high[i],
                        'bottom': low[i],
                        'mitigated': False
                    })
                    break
        else:  # bearish BOS
            # Look for last bullish candle before bearish BOS
            for i in range(bos_bar - 1, max(0, bos_bar - 20), -1):
                if close[i] > open_price[i]:  # Bullish candle
                    order_blocks.append({
                        'bar': i,
                        'type': 'bearish_ob',
                        'center': (high[i] + low[i]) / 2,
                        'width': high[i] - low[i],
                        'top': high[i],
                        'bottom': low[i],
                        'mitigated': False
                    })
                    break
    
    # Keep only max_boxes most recent
    if len(order_blocks) > max_boxes:
        order_blocks = sorted(order_blocks, key=lambda x: x['bar'], reverse=True)[:max_boxes]
    
    return pd.DataFrame(order_blocks)


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    equal_band_pips: float = 2.0,
    wick_ratio: float = 0.6
) -> pd.DataFrame:
    """
    Detect liquidity sweep events.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    equal_band_pips : float
        Band for equal highs/lows detection in pips
    wick_ratio : float
        Minimum wick-to-body ratio for sweep detection
        
    Returns
    -------
    pd.DataFrame
        Sweep events
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_price = df['open'].values
    
    pip_size = 0.0001
    equal_band = equal_band_pips * pip_size
    
    sweeps = []
    n_bars = len(df)
    
    # Look for equal highs
    for i in range(20, n_bars):
        # Find equal highs in recent history
        recent_highs = high[i-20:i]
        for j in range(len(recent_highs) - 1):
            if abs(recent_highs[j] - recent_highs[j+1]) <= equal_band:
                equal_high = max(recent_highs[j], recent_highs[j+1])
                
                # Check if current bar swept the high
                if high[i] > equal_high:
                    # Check if closed back below
                    if close[i] < equal_high:
                        # Calculate wick ratio
                        body = abs(close[i] - open_price[i])
                        upper_wick = high[i] - max(close[i], open_price[i])
                        
                        if body > 0 and upper_wick / body >= wick_ratio:
                            sweeps.append({
                                'bar': i,
                                'type': 'high_sweep',
                                'level': equal_high
                            })
                            break
    
    # Look for equal lows
    for i in range(20, n_bars):
        # Find equal lows in recent history
        recent_lows = low[i-20:i]
        for j in range(len(recent_lows) - 1):
            if abs(recent_lows[j] - recent_lows[j+1]) <= equal_band:
                equal_low = min(recent_lows[j], recent_lows[j+1])
                
                # Check if current bar swept the low
                if low[i] < equal_low:
                    # Check if closed back above
                    if close[i] > equal_low:
                        # Calculate wick ratio
                        body = abs(close[i] - open_price[i])
                        lower_wick = min(close[i], open_price[i]) - low[i]
                        
                        if body > 0 and lower_wick / body >= wick_ratio:
                            sweeps.append({
                                'bar': i,
                                'type': 'low_sweep',
                                'level': equal_low
                            })
                            break
    
    return pd.DataFrame(sweeps)


def compute_pd_zone(
    df: pd.DataFrame,
    active_swing: Dict
) -> pd.DataFrame:
    """
    Compute Premium/Discount zones based on active swing.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    active_swing : Dict
        Current active swing with 'high' and 'low' keys
        
    Returns
    -------
    pd.DataFrame
        DataFrame with pd_zone classification
    """
    result = df.copy()
    close = result['close'].values
    
    if 'high' not in active_swing or 'low' not in active_swing:
        result['pd_zone'] = 'equilibrium'
        result['pd_level'] = 50.0
        return result
    
    swing_high = active_swing['high']
    swing_low = active_swing['low']
    swing_range = swing_high - swing_low
    
    if swing_range == 0:
        result['pd_zone'] = 'equilibrium'
        result['pd_level'] = 50.0
        return result
    
    # Calculate position within swing range
    pd_level = ((close - swing_low) / swing_range) * 100
    
    # Classify zones
    pd_zone = np.where(pd_level > 61.8, 'premium',
               np.where(pd_level < 38.2, 'discount', 'equilibrium'))
    
    result['pd_zone'] = pd_zone
    result['pd_level'] = pd_level
    
    return result


def compute_smc_features(
    df: pd.DataFrame,
    swing_w: int = 3,
    min_exc_atr: float = 0.5,
    equal_band_pips: float = 2.0,
    wick_ratio: float = 0.6,
    disp_k: float = 2.0,
    max_ob_boxes: int = 3
) -> pd.DataFrame:
    """
    Aggregate SMC into per-bar features.
    
    Returns DataFrame with:
    - trend_bias ∈ {+1, 0, −1}
    - bos_flag, choch_flag, bos_age, choch_age
    - ob_up_boxes, ob_dn_boxes (vectorized, N=3 max)
    - ob_mitigated_up/dn, time_since_last_ob_mitigation
    - liq_sweep_high, liq_sweep_low, sweep_age
    - pd_zone ∈ {premium, equilibrium, discount}
    - Distances: dist_to_ob_up, dist_to_ob_dn, dist_to_eq_50, dist_to_swing
    """
    result = df.copy()
    n_bars = len(result)
    
    # Compute structure
    swings, bos_df, choch_df = compute_structure(result, swing_w, min_exc_atr)
    
    # Detect order blocks
    order_blocks = detect_order_blocks(result, swings, bos_df, disp_k, max_ob_boxes)
    
    # Detect liquidity sweeps
    sweeps = detect_liquidity_sweeps(result, equal_band_pips, wick_ratio)
    
    # Initialize feature arrays
    trend_bias = np.zeros(n_bars)
    bos_flag = np.zeros(n_bars, dtype=bool)
    choch_flag = np.zeros(n_bars, dtype=bool)
    bos_age = np.zeros(n_bars)
    choch_age = np.zeros(n_bars)
    
    # Populate BOS flags
    if len(bos_df) > 0:
        for _, bos_event in bos_df.iterrows():
            bar = bos_event['bar']
            if bar < n_bars:
                bos_flag[bar] = True
                bos_type = bos_event['type']
                # Update trend bias
                if bos_type == 'bullish':
                    trend_bias[bar:] = 1
                else:
                    trend_bias[bar:] = -1
    
    # Populate CHOCH flags
    if len(choch_df) > 0:
        for _, choch_event in choch_df.iterrows():
            bar = choch_event['bar']
            if bar < n_bars:
                choch_flag[bar] = True
    
    # Calculate ages
    last_bos = -1
    last_choch = -1
    for i in range(n_bars):
        if bos_flag[i]:
            last_bos = i
        if choch_flag[i]:
            last_choch = i
        
        bos_age[i] = i - last_bos if last_bos >= 0 else 999
        choch_age[i] = i - last_choch if last_choch >= 0 else 999
    
    # Order block features
    ob_up_center = np.zeros(n_bars)
    ob_up_width = np.zeros(n_bars)
    ob_dn_center = np.zeros(n_bars)
    ob_dn_width = np.zeros(n_bars)
    dist_to_ob_up = np.full(n_bars, 999.0)
    dist_to_ob_dn = np.full(n_bars, 999.0)
    
    close = result['close'].values
    
    if len(order_blocks) > 0:
        for _, ob in order_blocks.iterrows():
            ob_type = ob['type']
            center = ob['center']
            width = ob['width']
            
            if ob_type == 'bullish_ob':
                ob_up_center[:] = center
                ob_up_width[:] = width
                dist_to_ob_up[:] = np.abs(close - center)
            else:
                ob_dn_center[:] = center
                ob_dn_width[:] = width
                dist_to_ob_dn[:] = np.abs(close - center)
    
    # Liquidity sweep features
    liq_sweep_high = np.zeros(n_bars, dtype=bool)
    liq_sweep_low = np.zeros(n_bars, dtype=bool)
    sweep_age = np.full(n_bars, 999)
    
    if len(sweeps) > 0:
        for _, sweep in sweeps.iterrows():
            bar = sweep['bar']
            if bar < n_bars:
                if sweep['type'] == 'high_sweep':
                    liq_sweep_high[bar] = True
                else:
                    liq_sweep_low[bar] = True
    
    last_sweep = -1
    for i in range(n_bars):
        if liq_sweep_high[i] or liq_sweep_low[i]:
            last_sweep = i
        sweep_age[i] = i - last_sweep if last_sweep >= 0 else 999
    
    # Premium/Discount zones
    # Use recent swing range
    active_swing = {}
    if len(swings.get('highs', [])) > 0 and len(swings.get('lows', [])) > 0:
        recent_high = swings['highs'][-1]['price'] if swings['highs'] else close[-1]
        recent_low = swings['lows'][-1]['price'] if swings['lows'] else close[-1]
        active_swing = {'high': recent_high, 'low': recent_low}
    
    result_pd = compute_pd_zone(result, active_swing)
    
    # Add all features to result
    result['trend_bias'] = trend_bias
    result['bos_flag'] = bos_flag
    result['choch_flag'] = choch_flag
    result['bos_age'] = bos_age
    result['choch_age'] = choch_age
    result['ob_up_center'] = ob_up_center
    result['ob_up_width'] = ob_up_width
    result['ob_dn_center'] = ob_dn_center
    result['ob_dn_width'] = ob_dn_width
    result['dist_to_ob_up'] = dist_to_ob_up
    result['dist_to_ob_dn'] = dist_to_ob_dn
    result['liq_sweep_high'] = liq_sweep_high
    result['liq_sweep_low'] = liq_sweep_low
    result['sweep_age'] = sweep_age
    result['pd_zone'] = result_pd['pd_zone']
    result['pd_level'] = result_pd['pd_level']
    
    # Distance to equilibrium
    if 'high' in active_swing and 'low' in active_swing:
        eq_50 = (active_swing['high'] + active_swing['low']) / 2
        result['dist_to_eq_50'] = np.abs(close - eq_50)
        result['dist_to_swing'] = np.minimum(
            np.abs(close - active_swing['high']),
            np.abs(close - active_swing['low'])
        )
    else:
        result['dist_to_eq_50'] = 0
        result['dist_to_swing'] = 0
    
    return result
