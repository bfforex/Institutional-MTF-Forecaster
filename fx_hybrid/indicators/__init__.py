"""
Indicators Module

Custom institutional indicators for forex trading:
- Fair Value Gaps (FVG)
- Smart Money Concepts (SMC)
- Market Regime Detection
- Multi-Timeframe Pipeline
"""

from .fvg import detect_fvg
from .smc import (
    compute_structure,
    detect_order_blocks,
    detect_liquidity_sweeps,
    compute_pd_zone,
    compute_smc_features
)
from .regime import detect_regime
from .mtf import (
    build_mtf_sets,
    align_to_15m,
    merge_features,
    ensure_no_leakage
)

__all__ = [
    'detect_fvg',
    'compute_structure',
    'detect_order_blocks',
    'detect_liquidity_sweeps',
    'compute_pd_zone',
    'compute_smc_features',
    'detect_regime',
    'build_mtf_sets',
    'align_to_15m',
    'merge_features',
    'ensure_no_leakage'
]
