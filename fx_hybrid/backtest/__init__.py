"""
Backtest Module

Backtesting engine with confluence filtering and comprehensive reporting.
"""

from .engine import BacktestEngine
from .confluence import ConfluenceFilter, apply_confluence_filter
from .reports import (
    generate_report,
    plot_equity_curve,
    plot_returns_distribution,
    save_trades_csv,
    generate_full_report
)

__all__ = [
    'BacktestEngine',
    'ConfluenceFilter',
    'apply_confluence_filter',
    'generate_report',
    'plot_equity_curve',
    'plot_returns_distribution',
    'save_trades_csv',
    'generate_full_report'
]
