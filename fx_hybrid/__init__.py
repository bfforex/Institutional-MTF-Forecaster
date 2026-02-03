"""
FX Hybrid - Institutional Multi-Timeframe Forex Forecaster

Main package initialization.
"""

__version__ = '0.1.0'
__author__ = 'BFForex'

# Import key components for easy access
from . import indicators
from . import data
from . import models
from . import training
from . import backtest
from . import utils

__all__ = [
    'indicators',
    'data',
    'models',
    'training',
    'backtest',
    'utils'
]
