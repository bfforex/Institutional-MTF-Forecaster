"""
Training Module

Training loop, metrics, label generation, and continuous learning.
"""

from .trainer import ForexTrainer, ForexDataset, save_model, load_model
from .labels import generate_labels, generate_label_probabilities, compute_sample_weights
from .metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    profit_factor,
    compute_all_metrics
)
from .continuous_learning import MistakeTracker, apply_mistake_tracking

__all__ = [
    'ForexTrainer',
    'ForexDataset',
    'save_model',
    'load_model',
    'generate_labels',
    'generate_label_probabilities',
    'compute_sample_weights',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'win_rate',
    'profit_factor',
    'compute_all_metrics',
    'MistakeTracker',
    'apply_mistake_tracking'
]
