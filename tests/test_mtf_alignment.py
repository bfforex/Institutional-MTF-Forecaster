"""
Unit Tests for MTF Alignment

Tests multi-timeframe alignment and anti-leakage.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_hybrid.indicators.mtf import build_mtf_sets, align_to_15m, ensure_no_leakage


def create_15m_data(n_days=10):
    """Create synthetic 15-minute data."""
    # 96 bars per day (24 hours * 4 bars/hour)
    n_bars = n_days * 96
    
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='15min')
    
    price = 1.1000 + np.cumsum(np.random.normal(0, 0.0001, n_bars))
    
    df = pd.DataFrame({
        'open': price + np.random.normal(0, 0.0001, n_bars),
        'high': price + np.abs(np.random.normal(0, 0.0002, n_bars)),
        'low': price - np.abs(np.random.normal(0, 0.0002, n_bars)),
        'close': price,
        'volume': np.random.randint(1000, 5000, n_bars)
    }, index=dates)
    
    return df


def test_mtf_resampling():
    """Test that MTF resampling works."""
    df_15m = create_15m_data(n_days=5)
    
    mtf_sets = build_mtf_sets(df_15m, frames=['1H', '4H', '1D'])
    
    # Check that all timeframes exist
    assert '15min' in mtf_sets
    assert '1H' in mtf_sets
    assert '4H' in mtf_sets
    assert '1D' in mtf_sets
    
    # Check that HTF has fewer bars
    assert len(mtf_sets['1H']) < len(mtf_sets['15min'])
    assert len(mtf_sets['4H']) < len(mtf_sets['1H'])
    assert len(mtf_sets['1D']) < len(mtf_sets['4H'])


def test_alignment_shape():
    """Test that alignment preserves 15m shape."""
    df_15m = create_15m_data(n_days=5)
    
    mtf_sets = build_mtf_sets(df_15m, frames=['1H'])
    
    # Align 1H to 15m
    aligned_1h = align_to_15m(
        mtf_sets['1H'],
        df_15m.index,
        method='broadcast'
    )
    
    # Should have same length as 15m
    assert len(aligned_1h) == len(df_15m)


def test_anti_leakage():
    """Test that alignment prevents forward leakage."""
    df_15m = create_15m_data(n_days=3)
    
    mtf_sets = build_mtf_sets(df_15m, frames=['1H'])
    
    # Align with anti-leakage
    aligned_1h = align_to_15m(
        mtf_sets['1H'],
        df_15m.index,
        method='broadcast'
    )
    
    # Validate no leakage
    is_valid = ensure_no_leakage(df_15m, aligned_1h, '1H')
    assert is_valid


def test_mtf_values_constant():
    """Test that HTF values stay constant within period."""
    df_15m = create_15m_data(n_days=2)
    
    mtf_sets = build_mtf_sets(df_15m, frames=['1H'])
    
    aligned_1h = align_to_15m(
        mtf_sets['1H'],
        df_15m.index,
        method='broadcast'
    )
    
    # Within each hour, values should be constant (forward-filled)
    # Check that not every bar changes
    changes = (aligned_1h['close'].diff() != 0).sum()
    total_bars = len(aligned_1h)
    
    # Changes should be much less than total bars
    assert changes < total_bars * 0.5


def test_multiple_timeframes():
    """Test alignment of multiple timeframes."""
    df_15m = create_15m_data(n_days=7)
    
    mtf_sets = build_mtf_sets(df_15m, frames=['1H', '4H', '1D'])
    
    # Align all
    aligned_1h = align_to_15m(mtf_sets['1H'], df_15m.index)
    aligned_4h = align_to_15m(mtf_sets['4H'], df_15m.index)
    aligned_1d = align_to_15m(mtf_sets['1D'], df_15m.index)
    
    # All should have same length
    assert len(aligned_1h) == len(df_15m)
    assert len(aligned_4h) == len(df_15m)
    assert len(aligned_1d) == len(df_15m)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
