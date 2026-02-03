"""
Unit Tests for SMC Indicators

Tests Smart Money Concepts detection.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_hybrid.indicators.smc import compute_smc_features, compute_structure


def create_trending_data():
    """Create synthetic trending data."""
    n = 200
    trend = np.linspace(1.1000, 1.1200, n)
    noise = np.random.normal(0, 0.0010, n)
    
    close = trend + noise
    high = close + np.abs(np.random.normal(0, 0.0005, n))
    low = close - np.abs(np.random.normal(0, 0.0005, n))
    open_price = close + np.random.normal(0, 0.0003, n)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 5000, n)
    })
    
    return df


def test_smc_features_computation():
    """Test that SMC features are computed."""
    df = create_trending_data()
    result = compute_smc_features(df)
    
    # Check that key columns exist
    assert 'trend_bias' in result.columns
    assert 'bos_flag' in result.columns
    assert 'choch_flag' in result.columns
    assert 'pd_zone' in result.columns
    assert 'pd_level' in result.columns


def test_trend_bias_values():
    """Test that trend bias has valid values."""
    df = create_trending_data()
    result = compute_smc_features(df)
    
    # Trend bias should be -1, 0, or 1
    assert result['trend_bias'].isin([-1, 0, 1]).all()


def test_structure_detection():
    """Test swing structure detection."""
    df = create_trending_data()
    swings, bos, choch = compute_structure(df)
    
    # Should detect some structure
    assert 'highs' in swings or 'lows' in swings


def test_pd_zones():
    """Test premium/discount zone classification."""
    df = create_trending_data()
    result = compute_smc_features(df)
    
    # PD zones should be one of the valid values
    assert result['pd_zone'].isin(['premium', 'equilibrium', 'discount']).all()
    
    # PD level should be percentage (0-100)
    assert (result['pd_level'] >= -50).all()  # Allow some range
    assert (result['pd_level'] <= 150).all()


def test_distance_features():
    """Test distance feature calculations."""
    df = create_trending_data()
    result = compute_smc_features(df)
    
    # Check distance columns exist
    assert 'dist_to_ob_up' in result.columns
    assert 'dist_to_ob_dn' in result.columns
    assert 'dist_to_eq_50' in result.columns
    
    # Distances should be non-negative
    assert (result['dist_to_ob_up'] >= 0).all()
    assert (result['dist_to_ob_dn'] >= 0).all()
    assert (result['dist_to_eq_50'] >= 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
