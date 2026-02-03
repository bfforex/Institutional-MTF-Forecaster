"""
Integration Tests

End-to-end tests for the complete pipeline.
"""

import pytest
import torch
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_hybrid.indicators.fvg import detect_fvg
from fx_hybrid.indicators.smc import compute_smc_features
from fx_hybrid.data.features_mtf import build_features_for_timeframe
from fx_hybrid.models import HybridForexModel, count_parameters
from fx_hybrid.training import generate_labels
from fx_hybrid.data import create_sequences


def create_test_data(n_bars=500):
    """Create test OHLCV data."""
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


def test_full_feature_pipeline():
    """Test complete feature building pipeline."""
    df = create_test_data(n_bars=300)
    
    # Build features
    features = build_features_for_timeframe(
        df,
        '15min',
        fvg_config={'k': 2.0, 'atr_len': 14, 'min_gap_pips': 2.0},
        smc_config={'swing_w': 3, 'min_exc_atr': 0.5},
        regime_config={'lookback': 100, 'adx_threshold': 25}
    )
    
    # Check that features were added
    assert 'fvg_has_bull' in features.columns
    assert 'trend_bias' in features.columns
    assert 'regime_numeric' in features.columns
    
    # Check no NaN in critical columns
    assert not features['close'].isna().any()


def test_model_initialization():
    """Test model initialization."""
    model = HybridForexModel(
        input_dim=50,
        window=128,
        conv_channels=[32, 64],
        kernel_size=5,
        lstm_hidden=96,
        lstm_layers=2,
        attention_dim=32,
        num_classes=3,
        dropout=0.2
    )
    
    # Check parameter count
    n_params = count_parameters(model)
    assert n_params > 0
    print(f"Model has {n_params:,} parameters")


def test_model_forward_pass():
    """Test model forward pass."""
    batch_size = 4
    window = 128
    input_dim = 50
    
    model = HybridForexModel(
        input_dim=input_dim,
        window=window,
        num_classes=3
    )
    
    # Create dummy input
    x = torch.randn(batch_size, window, input_dim)
    
    # Forward pass
    logits, confidence, attention = model(x, None)
    
    # Check output shapes
    assert logits.shape == (batch_size, 3)
    assert confidence.shape == (batch_size, 1)
    assert attention.shape == (batch_size, window)


def test_label_generation():
    """Test label generation."""
    df = create_test_data(n_bars=200)
    
    labels = generate_labels(
        df,
        horizon_minutes=30,
        band_pct=0.10
    )
    
    # Check label values
    assert labels.min() >= 0
    assert labels.max() <= 2
    
    # Check that all classes are present
    unique_labels = np.unique(labels)
    assert len(unique_labels) >= 2  # At least 2 classes


def test_sequence_creation():
    """Test sequence creation."""
    n_samples = 200
    n_features = 30
    window = 64
    
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 3, n_samples)
    
    X, y = create_sequences(features, labels, window)
    
    # Check shapes
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == window
    assert X.shape[2] == n_features


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_inference():
    """Test model inference on GPU."""
    model = HybridForexModel(input_dim=50, window=128, num_classes=3)
    model = model.cuda()
    
    x = torch.randn(2, 128, 50).cuda()
    
    with torch.no_grad():
        logits, confidence, attention = model(x, None)
    
    assert logits.device.type == 'cuda'
    assert confidence.device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
