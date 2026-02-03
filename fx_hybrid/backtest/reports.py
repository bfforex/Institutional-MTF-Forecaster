"""
Backtest Reporting

Generate comprehensive backtest performance reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os


def generate_report(
    backtest_results: Dict,
    trades_df: Optional[pd.DataFrame] = None,
    equity_curve: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive backtest report.
    
    Parameters
    ----------
    backtest_results : Dict
        Dictionary of backtest metrics
    trades_df : pd.DataFrame, optional
        DataFrame of individual trades
    equity_curve : np.ndarray, optional
        Equity curve array
    save_path : str, optional
        Path to save report
        
    Returns
    -------
    str
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("BACKTEST PERFORMANCE REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Overall Statistics
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 70)
    report_lines.append(f"Total Trades:        {backtest_results.get('total_trades', 0):>10}")
    report_lines.append(f"Winning Trades:      {backtest_results.get('winning_trades', 0):>10}")
    report_lines.append(f"Losing Trades:       {backtest_results.get('losing_trades', 0):>10}")
    report_lines.append(f"Win Rate:            {backtest_results.get('win_rate', 0):>9.2f}%")
    report_lines.append("")
    
    # Profit & Loss
    report_lines.append("PROFIT & LOSS")
    report_lines.append("-" * 70)
    report_lines.append(f"Total PnL:           {backtest_results.get('total_pnl', 0):>9.2f}%")
    report_lines.append(f"Profit Factor:       {backtest_results.get('profit_factor', 0):>10.2f}")
    report_lines.append(f"Average Win:         {backtest_results.get('avg_win', 0):>9.4f}")
    report_lines.append(f"Average Loss:        {backtest_results.get('avg_loss', 0):>9.4f}")
    report_lines.append("")
    
    # Risk Metrics
    report_lines.append("RISK METRICS")
    report_lines.append("-" * 70)
    report_lines.append(f"Sharpe Ratio:        {backtest_results.get('sharpe_ratio', 0):>10.2f}")
    report_lines.append(f"Sortino Ratio:       {backtest_results.get('sortino_ratio', 0):>10.2f}")
    report_lines.append(f"Calmar Ratio:        {backtest_results.get('calmar_ratio', 0):>10.2f}")
    report_lines.append(f"Max Drawdown:        {backtest_results.get('max_drawdown', 0):>9.2f}%")
    report_lines.append("")
    
    # Trade Distribution
    if trades_df is not None and len(trades_df) > 0:
        report_lines.append("TRADE DISTRIBUTION")
        report_lines.append("-" * 70)
        
        long_trades = (trades_df['direction'] == 1).sum()
        short_trades = (trades_df['direction'] == -1).sum()
        
        report_lines.append(f"Long Trades:         {long_trades:>10}")
        report_lines.append(f"Short Trades:        {short_trades:>10}")
        
        if long_trades > 0:
            long_pnl = trades_df[trades_df['direction'] == 1]['pnl'].sum() * 100
            long_win_rate = (trades_df[trades_df['direction'] == 1]['pnl'] > 0).sum() / long_trades * 100
            report_lines.append(f"Long PnL:            {long_pnl:>9.2f}%")
            report_lines.append(f"Long Win Rate:       {long_win_rate:>9.2f}%")
        
        if short_trades > 0:
            short_pnl = trades_df[trades_df['direction'] == -1]['pnl'].sum() * 100
            short_win_rate = (trades_df[trades_df['direction'] == -1]['pnl'] > 0).sum() / short_trades * 100
            report_lines.append(f"Short PnL:           {short_pnl:>9.2f}%")
            report_lines.append(f"Short Win Rate:      {short_win_rate:>9.2f}%")
        
        report_lines.append("")
    
    report_lines.append("=" * 70)
    
    report_text = "\n".join(report_lines)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text


def plot_equity_curve(
    equity_curve: np.ndarray,
    title: str = "Equity Curve",
    save_path: Optional[str] = None
):
    """
    Plot equity curve.
    
    Parameters
    ----------
    equity_curve : np.ndarray
        Equity curve array
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(equity_curve, linewidth=2, color='blue')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Equity')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at starting equity
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Starting Equity')
    
    # Calculate and plot drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (running_max - equity_curve) / running_max
    
    ax2 = ax.twinx()
    ax2.fill_between(range(len(drawdown)), 0, -drawdown * 100, 
                     alpha=0.3, color='red', label='Drawdown')
    ax2.set_ylabel('Drawdown (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_returns_distribution(
    trades_df: pd.DataFrame,
    title: str = "Returns Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of trade returns.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame of trades
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    returns_pct = trades_df['pnl_pct'].values
    ax1.hist(returns_pct, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Returns Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative returns
    cumulative_returns = np.cumsum(returns_pct)
    ax2.plot(cumulative_returns, linewidth=2, color='green')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.set_title('Cumulative Returns')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_trades_csv(
    trades_df: pd.DataFrame,
    save_path: str
):
    """
    Save trades to CSV file.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame of trades
    save_path : str
        Path to save CSV
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trades_df.to_csv(save_path, index=False)
    print(f"Trades saved to {save_path}")


def generate_full_report(
    backtest_results: Dict,
    trades_df: pd.DataFrame,
    equity_curve: np.ndarray,
    output_dir: str = 'backtest_results'
):
    """
    Generate full report with text, plots, and CSV.
    
    Parameters
    ----------
    backtest_results : Dict
        Backtest metrics
    trades_df : pd.DataFrame
        Trades DataFrame
    equity_curve : np.ndarray
        Equity curve
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate text report
    report_text = generate_report(
        backtest_results,
        trades_df,
        equity_curve,
        save_path=os.path.join(output_dir, 'backtest_report.txt')
    )
    print(report_text)
    
    # Plot equity curve
    plot_equity_curve(
        equity_curve,
        save_path=os.path.join(output_dir, 'equity_curve.png')
    )
    
    # Plot returns distribution
    if len(trades_df) > 0:
        plot_returns_distribution(
            trades_df,
            save_path=os.path.join(output_dir, 'returns_distribution.png')
        )
    
    # Save trades CSV
    if len(trades_df) > 0:
        save_trades_csv(
            trades_df,
            os.path.join(output_dir, 'trades.csv')
        )
    
    print(f"\nFull report generated in: {output_dir}")
