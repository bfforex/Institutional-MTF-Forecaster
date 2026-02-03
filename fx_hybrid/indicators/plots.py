"""
Visualization Utilities for Indicators

Provides plotting functions for FVG, SMC, and MTF analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def plot_fvg(
    df: pd.DataFrame,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 8)
):
    """
    Plot price action with FVG overlays.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV and FVG features
    start_idx : int
        Start index for plotting
    end_idx : int, optional
        End index for plotting
    figsize : Tuple[int, int]
        Figure size
    """
    if end_idx is None:
        end_idx = len(df)
    
    plot_df = df.iloc[start_idx:end_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot price
    ax1.plot(plot_df.index, plot_df['close'], label='Close', color='black', linewidth=1)
    
    # Highlight FVG zones
    if 'fvg_has_bull' in plot_df.columns:
        bull_fvg_idx = plot_df[plot_df['fvg_has_bull']].index
        ax1.scatter(bull_fvg_idx, plot_df.loc[bull_fvg_idx, 'close'], 
                   color='green', marker='^', s=100, label='Bullish FVG', zorder=5)
    
    if 'fvg_has_bear' in plot_df.columns:
        bear_fvg_idx = plot_df[plot_df['fvg_has_bear']].index
        ax1.scatter(bear_fvg_idx, plot_df.loc[bear_fvg_idx, 'close'], 
                   color='red', marker='v', s=100, label='Bearish FVG', zorder=5)
    
    ax1.set_ylabel('Price')
    ax1.set_title('Fair Value Gaps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot distances
    if 'dist_to_nearest_bull_fvg' in plot_df.columns:
        ax2.plot(plot_df.index, plot_df['dist_to_nearest_bull_fvg'], 
                label='Dist to Bull FVG', color='green', alpha=0.7)
    if 'dist_to_nearest_bear_fvg' in plot_df.columns:
        ax2.plot(plot_df.index, plot_df['dist_to_nearest_bear_fvg'], 
                label='Dist to Bear FVG', color='red', alpha=0.7)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Distance (pips)')
    ax2.set_title('FVG Distances')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_smc(
    df: pd.DataFrame,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot price action with SMC overlays.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV and SMC features
    start_idx : int
        Start index for plotting
    end_idx : int, optional
        End index for plotting
    figsize : Tuple[int, int]
        Figure size
    """
    if end_idx is None:
        end_idx = len(df)
    
    plot_df = df.iloc[start_idx:end_idx]
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Price with BOS/CHOCH
    ax1 = axes[0]
    ax1.plot(plot_df.index, plot_df['close'], label='Close', color='black', linewidth=1)
    
    if 'bos_flag' in plot_df.columns:
        bos_idx = plot_df[plot_df['bos_flag']].index
        ax1.scatter(bos_idx, plot_df.loc[bos_idx, 'close'], 
                   color='blue', marker='o', s=80, label='BOS', zorder=5)
    
    if 'choch_flag' in plot_df.columns:
        choch_idx = plot_df[plot_df['choch_flag']].index
        ax1.scatter(choch_idx, plot_df.loc[choch_idx, 'close'], 
                   color='orange', marker='s', s=80, label='CHOCH', zorder=5)
    
    ax1.set_ylabel('Price')
    ax1.set_title('Structure: BOS & CHOCH')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trend Bias & PD Zones
    ax2 = axes[1]
    if 'trend_bias' in plot_df.columns:
        ax2.plot(plot_df.index, plot_df['trend_bias'], label='Trend Bias', 
                color='purple', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_ylabel('Trend Bias')
    ax2.set_title('Trend Bias')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Premium/Discount Zones
    ax3 = axes[2]
    if 'pd_level' in plot_df.columns:
        ax3.plot(plot_df.index, plot_df['pd_level'], label='PD Level', 
                color='teal', linewidth=1.5)
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
        ax3.axhline(y=61.8, color='red', linestyle=':', alpha=0.5, label='Premium')
        ax3.axhline(y=38.2, color='green', linestyle=':', alpha=0.5, label='Discount')
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('PD Level (%)')
    ax3.set_title('Premium/Discount Zones')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_regime(
    df: pd.DataFrame,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 8)
):
    """
    Plot market regime classification.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regime features
    start_idx : int
        Start index for plotting
    end_idx : int, optional
        End index for plotting
    figsize : Tuple[int, int]
        Figure size
    """
    if end_idx is None:
        end_idx = len(df)
    
    plot_df = df.iloc[start_idx:end_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot price with regime background
    ax1.plot(plot_df.index, plot_df['close'], label='Close', color='black', linewidth=1)
    
    if 'regime_numeric' in plot_df.columns:
        # Color background based on regime
        for i in range(len(plot_df) - 1):
            regime = plot_df['regime_numeric'].iloc[i]
            if regime == 2:  # Trending up
                ax1.axvspan(plot_df.index[i], plot_df.index[i+1], 
                           alpha=0.1, color='green')
            elif regime == -2:  # Trending down
                ax1.axvspan(plot_df.index[i], plot_df.index[i+1], 
                           alpha=0.1, color='red')
    
    ax1.set_ylabel('Price')
    ax1.set_title('Market Regime')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot ADX
    if 'adx' in plot_df.columns:
        ax2.plot(plot_df.index, plot_df['adx'], label='ADX', color='blue', linewidth=1.5)
        ax2.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ADX')
    ax2.set_title('Average Directional Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mtf_features(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot multi-timeframe feature alignment.
    
    Parameters
    ----------
    df_15m : pd.DataFrame
        15-minute features
    df_1h : pd.DataFrame
        1-hour features (aligned)
    df_4h : pd.DataFrame
        4-hour features (aligned)
    start_idx : int
        Start index for plotting
    end_idx : int, optional
        End index for plotting
    figsize : Tuple[int, int]
        Figure size
    """
    if end_idx is None:
        end_idx = len(df_15m)
    
    plot_15m = df_15m.iloc[start_idx:end_idx]
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot closes from different timeframes
    ax1 = axes[0]
    ax1.plot(plot_15m.index, plot_15m['close'], label='15m Close', 
            color='blue', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Price')
    ax1.set_title('15-Minute Timeframe')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 1H aligned
    ax2 = axes[1]
    if len(df_1h) > 0:
        plot_1h = df_1h.iloc[start_idx:min(end_idx, len(df_1h))]
        if 'close' in plot_1h.columns:
            ax2.plot(plot_1h.index, plot_1h['close'], label='1H Close (aligned)', 
                    color='green', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Price')
    ax2.set_title('1-Hour Timeframe (Aligned to 15m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 4H aligned
    ax3 = axes[2]
    if len(df_4h) > 0:
        plot_4h = df_4h.iloc[start_idx:min(end_idx, len(df_4h))]
        if 'close' in plot_4h.columns:
            ax3.plot(plot_4h.index, plot_4h['close'], label='4H Close (aligned)', 
                    color='red', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Price')
    ax3.set_title('4-Hour Timeframe (Aligned to 15m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
