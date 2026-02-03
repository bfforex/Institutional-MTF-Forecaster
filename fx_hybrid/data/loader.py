"""
Data Loader Utilities

Handles loading and preprocessing of forex data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import os


def load_csv_data(
    filepath: str,
    datetime_col: Optional[str] = None,
    datetime_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    datetime_col : str, optional
        Name of datetime column
    datetime_format : str, optional
        Format string for datetime parsing
        
    Returns
    -------
    pd.DataFrame
        Loaded OHLCV data with DatetimeIndex
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Try to detect datetime column
    if datetime_col is None:
        # Common datetime column names
        possible_names = ['timestamp', 'datetime', 'time', 'date', 'Date', 'Time', 'Datetime']
        df_temp = pd.read_csv(filepath, nrows=5)
        for name in possible_names:
            if name in df_temp.columns:
                datetime_col = name
                break
    
    if datetime_col:
        df = pd.read_csv(filepath, parse_dates=[datetime_col])
        df = df.set_index(datetime_col)
    else:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            raise ValueError("Could not parse index as datetime")
    
    # Sort by datetime
    df = df.sort_index()
    
    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    embargo_bars: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    train_ratio : float
        Ratio of data to use for training
    embargo_bars : int
        Number of bars to skip between train and test (anti-leakage)
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train + embargo_bars:]
    
    return train_df, test_df


def create_walk_forward_splits(
    df: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 1,
    embargo_bars: int = 120
) -> list:
    """
    Create walk-forward validation splits.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with DatetimeIndex
    train_months : int
        Number of months for training
    test_months : int
        Number of months for testing
    embargo_bars : int
        Number of bars to skip between train and test
        
    Returns
    -------
    list
        List of (train_df, test_df) tuples
    """
    splits = []
    
    start_date = df.index[0]
    end_date = df.index[-1]
    
    current_date = start_date
    
    while current_date < end_date:
        # Define train period
        train_start = current_date
        train_end = train_start + pd.DateOffset(months=train_months)
        
        # Define test period (after embargo)
        test_start = train_end + pd.Timedelta(minutes=15 * embargo_bars)
        test_end = test_start + pd.DateOffset(months=test_months)
        
        # Extract data
        train_df = df[(df.index >= train_start) & (df.index < train_end)]
        test_df = df[(df.index >= test_start) & (df.index < test_end)]
        
        if len(train_df) > 0 and len(test_df) > 0:
            splits.append((train_df, test_df))
        
        # Move to next fold
        current_date = test_start
    
    return splits


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    window: int = 256,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series modeling.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array of shape (n_samples, n_features)
    labels : np.ndarray
        Label array of shape (n_samples,)
    window : int
        Sequence window length
    stride : int
        Stride for sliding window
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (sequences, sequence_labels) where sequences has shape
        (n_sequences, window, n_features)
    """
    n_samples, n_features = features.shape
    
    sequences = []
    sequence_labels = []
    
    for i in range(0, n_samples - window + 1, stride):
        seq = features[i:i+window]
        label = labels[i+window-1]  # Label for last timestep
        
        sequences.append(seq)
        sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels)


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'undersample'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance class distribution in dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Feature array
    y : np.ndarray
        Label array
    method : str
        Balancing method: 'undersample' or 'oversample'
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (balanced_X, balanced_y)
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    
    if method == 'undersample':
        # Undersample to match smallest class
        min_count = counts.min()
        
        balanced_X = []
        balanced_y = []
        
        for cls in unique_classes:
            idx = np.where(y == cls)[0]
            sampled_idx = np.random.choice(idx, min_count, replace=False)
            balanced_X.append(X[sampled_idx])
            balanced_y.append(y[sampled_idx])
        
        balanced_X = np.concatenate(balanced_X)
        balanced_y = np.concatenate(balanced_y)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(balanced_y))
        balanced_X = balanced_X[shuffle_idx]
        balanced_y = balanced_y[shuffle_idx]
        
        return balanced_X, balanced_y
    
    elif method == 'oversample':
        # Oversample to match largest class
        max_count = counts.max()
        
        balanced_X = []
        balanced_y = []
        
        for cls in unique_classes:
            idx = np.where(y == cls)[0]
            sampled_idx = np.random.choice(idx, max_count, replace=True)
            balanced_X.append(X[sampled_idx])
            balanced_y.append(y[sampled_idx])
        
        balanced_X = np.concatenate(balanced_X)
        balanced_y = np.concatenate(balanced_y)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(balanced_y))
        balanced_X = balanced_X[shuffle_idx]
        balanced_y = balanced_y[shuffle_idx]
        
        return balanced_X, balanced_y
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")
