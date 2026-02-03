"""
Models Module

Neural network architectures for forex forecasting.
"""

from .cnn_bilstm_attn import HybridForexModel, count_parameters
from .film_conditioning import FiLMLayer, MultiScaleFiLM, ConditionalBatchNorm
from .fusion import BaselineFusion, DualTowerFusion, AttentionFusion, HierarchicalFusion

__all__ = [
    'HybridForexModel',
    'count_parameters',
    'FiLMLayer',
    'MultiScaleFiLM',
    'ConditionalBatchNorm',
    'BaselineFusion',
    'DualTowerFusion',
    'AttentionFusion',
    'HierarchicalFusion'
]
