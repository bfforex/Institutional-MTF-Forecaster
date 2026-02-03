"""
Evaluation Metrics for Forex Trading

Includes Sharpe ratio, Sortino ratio, Calmar ratio, and trading-specific metrics.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from typing import Dict, Tuple


def calculate_returns(
    predictions: np.ndarray,
    actual_prices: np.ndarray,
    threshold: float = 0.55
) -> np.ndarray:
    """
    Calculate returns based on predictions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (probabilities for each class)
    actual_prices : np.ndarray
        Actual price series
    threshold : float
        Probability threshold for taking positions
        
    Returns
    -------
    np.ndarray
        Array of returns
    """
    n_samples = len(predictions) - 1
    returns = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Get max probability and class
        max_prob = predictions[i].max()
        pred_class = predictions[i].argmax()
        
        if max_prob >= threshold:
            # Take position
            if pred_class == 2:  # Up
                position = 1
            elif pred_class == 0:  # Down
                position = -1
            else:  # Neutral
                position = 0
        else:
            position = 0
        
        # Calculate return
        price_return = (actual_prices[i+1] - actual_prices[i]) / actual_prices[i]
        returns[i] = position * price_return
    
    return returns


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns
    -------
    float
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / (252 * 96)  # Daily risk-free rate for 15m bars
    return np.sqrt(252 * 96) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (focuses on downside volatility).
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    risk_free_rate : float
        Risk-free rate
        
    Returns
    -------
    float
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / (252 * 96)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    return np.sqrt(252 * 96) * excess_returns.mean() / downside_returns.std()


def max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
        
    Returns
    -------
    float
        Maximum drawdown (as positive percentage)
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    
    return drawdown.max() * 100


def calmar_ratio(returns: np.ndarray) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
        
    Returns
    -------
    float
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    total_return = (np.cumprod(1 + returns)[-1] - 1) * 100
    mdd = max_drawdown(returns)
    
    if mdd == 0:
        return 0.0
    
    return total_return / mdd


def win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of profitable trades).
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
        
    Returns
    -------
    float
        Win rate as percentage
    """
    if len(returns) == 0:
        return 0.0
    
    profitable = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    
    if total_trades == 0:
        return 0.0
    
    return (profitable / total_trades) * 100


def profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
        
    Returns
    -------
    float
        Profit factor
    """
    if len(returns) == 0:
        return 0.0
    
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss == 0:
        return 0.0
    
    return gross_profit / gross_loss


def calibration_error(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (probabilities)
    labels : np.ndarray
        True labels
    n_bins : int
        Number of bins for calibration
        
    Returns
    -------
    float
        Expected Calibration Error
    """
    # Get max probability and predicted class
    confidences = predictions.max(axis=1)
    pred_classes = predictions.argmax(axis=1)
    accuracies = (pred_classes == labels).astype(float)
    
    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
    
    return ece


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    prices: np.ndarray,
    threshold: float = 0.55
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    labels : np.ndarray
        True labels
    prices : np.ndarray
        Price series
    threshold : float
        Decision threshold
        
    Returns
    -------
    Dict[str, float]
        Dictionary of all metrics
    """
    # Classification metrics
    pred_classes = predictions.argmax(axis=1)
    
    accuracy = accuracy_score(labels, pred_classes)
    f1_macro = f1_score(labels, pred_classes, average='macro')
    f1_weighted = f1_score(labels, pred_classes, average='weighted')
    
    # Trading metrics
    returns = calculate_returns(predictions, prices, threshold)
    
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    calmar = calmar_ratio(returns)
    mdd = max_drawdown(returns)
    wr = win_rate(returns)
    pf = profit_factor(returns)
    
    # Total PnL
    total_pnl = (np.cumprod(1 + returns)[-1] - 1) * 100 if len(returns) > 0 else 0
    
    # Calibration
    ece = calibration_error(predictions, labels)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': mdd,
        'win_rate': wr,
        'profit_factor': pf,
        'total_pnl': total_pnl,
        'calibration_error': ece
    }
