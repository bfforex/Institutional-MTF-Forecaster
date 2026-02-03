"""
Unit Tests for FVG Detection

Tests the Fair Value Gap detection logic.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_hybrid.indicators.fvg import detect_fvg


def create_test_data_with_fvg():
    """Create synthetic data with known FVG."""
    n = 100
    data = {
        'open': np.linspace(1.1000, 1.1050, n),
        'high': np.linspace(1.1010, 1.1060, n),
        'low': np.linspace(1.0990, 1.1040, n),
        'close': np.linspace(1.1005, 1.1055, n),
        'volume': np.random.randint(1000, 5000, n)
    }
    
    df = pd.DataFrame(data)
    
    # Create a bullish FVG at index 50
    # Gap: low[51] > high[49]
    df.loc[49, 'high'] = 1.1000
    df.loc[50, 'high'] = 1.1020  # Displacement bar
    df.loc[50, 'low'] = 1.1010
    df.loc[51, 'low'] = 1.1025  # Gap starts here
    
    return df


def test_fvg_detection():
    """Test that FVG is detected correctly."""
    df = create_test_data_with_fvg()
    result = detect_fvg(df, k=1.5, min_gap_pips=1.0)
    
    # Check that FVG columns exist
    assert 'fvg_has_bull' in result.columns
    assert 'fvg_has_bear' in result.columns
    assert 'fvg_gap_pips' in result.columns
    
    # Check that at least one FVG was detected
    assert result['fvg_has_bull'].sum() > 0 or result['fvg_has_bear'].sum() > 0


def test_fvg_no_gaps():
    """Test with data that has no gaps."""
    n = 50
    data = {
        'open': np.linspace(1.1000, 1.1010, n),
        'high': np.linspace(1.1005, 1.1015, n),
        'low': np.linspace(1.0995, 1.1005, n),
        'close': np.linspace(1.1002, 1.1012, n),
        'volume': np.ones(n) * 1000
    }
    
    df = pd.DataFrame(data)
    result = detect_fvg(df, k=5.0, min_gap_pips=5.0)  # Very strict
    
    # Should detect very few or no FVGs
    assert result['fvg_has_bull'].sum() == 0
    assert result['fvg_has_bear'].sum() == 0


def test_fvg_distances():
    """Test distance calculations."""
    df = create_test_data_with_fvg()
    result = detect_fvg(df)
    
    # Check that distance columns exist
    assert 'dist_to_nearest_bull_fvg' in result.columns
    assert 'dist_to_nearest_bear_fvg' in result.columns
    
    # Distances should be non-negative
    assert (result['dist_to_nearest_bull_fvg'] >= 0).all()
    assert (result['dist_to_nearest_bear_fvg'] >= 0).all()


def test_fvg_age_tracking():
    """Test that FVG age is tracked."""
    df = create_test_data_with_fvg()
    result = detect_fvg(df, max_age=50)
    
    # Check age column exists
    assert 'fvg_age_bars' in result.columns
    
    # Age should be non-negative
    assert (result['fvg_age_bars'] >= 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
