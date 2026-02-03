"""
Label Generation for 3-Class Prediction

Generates Up/Neutral/Down labels based on forward returns.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_labels(
    df: pd.DataFrame,
    horizon_minutes: int = 30,
    band_pct: float = 0.10,
    label_smoothing: float = 0.0
) -> np.ndarray:
    """
    Generate 3-class labels based on forward returns.
    
    Classes:
    - 0: Down (return < -band_pct)
    - 1: Neutral (|return| <= band_pct)
    - 2: Up (return > band_pct)
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex
    horizon_minutes : int
        Forward-looking horizon in minutes
    band_pct : float
        Band threshold in percentage (e.g., 0.10 for 0.10%)
    label_smoothing : float
        Label smoothing factor for regularization
        
    Returns
    -------
    np.ndarray
        Array of labels (0, 1, 2)
    """
    close = df['close'].values
    
    # Calculate forward returns
    # Determine number of bars for horizon (assumes 15m bars)
    bars_per_hour = 4
    horizon_bars = (horizon_minutes // 15)
    
    forward_returns = np.full(len(close), np.nan)
    
    for i in range(len(close) - horizon_bars):
        future_price = close[i + horizon_bars]
        current_price = close[i]
        ret = ((future_price - current_price) / current_price) * 100  # percentage
        forward_returns[i] = ret
    
    # Generate labels
    labels = np.full(len(close), 1)  # Default to neutral
    
    labels[forward_returns > band_pct] = 2  # Up
    labels[forward_returns < -band_pct] = 0  # Down
    labels[np.abs(forward_returns) <= band_pct] = 1  # Neutral
    
    # Handle NaN values at the end (no future data)
    # Set to neutral as default
    labels[np.isnan(forward_returns)] = 1
    
    return labels.astype(np.int64)


def generate_label_probabilities(
    df: pd.DataFrame,
    horizon_minutes: int = 30,
    band_pct: float = 0.10,
    smoothing: float = 0.05
) -> np.ndarray:
    """
    Generate soft labels with label smoothing.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    horizon_minutes : int
        Forward horizon
    band_pct : float
        Band threshold
    smoothing : float
        Label smoothing factor (e.g., 0.05)
        
    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 3) with label probabilities
    """
    labels = generate_labels(df, horizon_minutes, band_pct)
    n_samples = len(labels)
    n_classes = 3
    
    # One-hot encode
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), labels] = 1
    
    # Apply label smoothing
    if smoothing > 0:
        one_hot = (1 - smoothing) * one_hot + smoothing / n_classes
    
    return one_hot


def compute_sample_weights(
    labels: np.ndarray,
    method: str = 'balanced'
) -> np.ndarray:
    """
    Compute sample weights for imbalanced classes.
    
    Parameters
    ----------
    labels : np.ndarray
        Label array
    method : str
        Weighting method: 'balanced' or 'sqrt_balanced'
        
    Returns
    -------
    np.ndarray
        Sample weights
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique_classes)
    
    if method == 'balanced':
        # Inverse frequency weighting
        class_weights = n_samples / (n_classes * counts)
    elif method == 'sqrt_balanced':
        # Square root of inverse frequency
        class_weights = np.sqrt(n_samples / (n_classes * counts))
    else:
        class_weights = np.ones(n_classes)
    
    # Map to sample weights
    sample_weights = np.zeros(n_samples)
    for i, cls in enumerate(unique_classes):
        sample_weights[labels == cls] = class_weights[i]
    
    return sample_weights


def validate_labels(
    labels: np.ndarray,
    min_samples_per_class: int = 10
) -> Tuple[bool, str]:
    """
    Validate label distribution.
    
    Parameters
    ----------
    labels : np.ndarray
        Label array
    min_samples_per_class : int
        Minimum required samples per class
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Check if all classes present
    if len(unique_classes) < 3:
        missing = set([0, 1, 2]) - set(unique_classes)
        return False, f"Missing classes: {missing}"
    
    # Check minimum samples
    for cls, count in zip(unique_classes, counts):
        if count < min_samples_per_class:
            return False, f"Class {cls} has only {count} samples (min: {min_samples_per_class})"
    
    # Check for severe imbalance
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 10:
        return False, f"Severe class imbalance: ratio {imbalance_ratio:.1f}"
    
    return True, "Labels are valid"
