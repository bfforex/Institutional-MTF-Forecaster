"""
Backtest Engine

Walk-forward backtesting with optional confluence filtering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from .confluence import apply_confluence_filter


class BacktestEngine:
    """
    Backtesting engine with two modes:
    1. Base mode: Trade if max_prob >= threshold
    2. Confluence mode: Trade only with SMC/FVG confluence
    """
    
    def __init__(self, config: Dict):
        """
        Parameters
        ----------
        config : Dict
            Backtest configuration
        """
        self.config = config
        self.mode = config.get('mode', 'base')
        self.confluence_config = config.get('confluence', {})
        
        # Thresholds
        self.threshold_up = config.get('thresholds', {}).get('up', 0.55)
        self.threshold_down = config.get('thresholds', {}).get('down', 0.55)
        self.neutral_band = config.get('thresholds', {}).get('neutral_band', 0.05)
        
        # Costs
        self.spread_pips = config.get('costs', {}).get('spread_pips', 0.8)
        self.commission_pct = config.get('costs', {}).get('commission_pct', 0.0)
        self.slippage_pips = config.get('costs', {}).get('slippage_pips', 0.3)
        
        # Risk
        self.position_size_pct = config.get('risk', {}).get('position_size_pct', 1.0)
        self.max_positions = config.get('risk', {}).get('max_positions', 1)
        
        # Pip size
        self.pip_size = 0.0001
        
        # Tracking
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(
        self,
        predictions: np.ndarray,
        features: pd.DataFrame,
        prices: np.ndarray
    ) -> Dict:
        """
        Execute backtest.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions (probabilities), shape (n_samples, 3)
        features : pd.DataFrame
            Feature DataFrame with indicators
        prices : np.ndarray
            Price series
            
        Returns
        -------
        Dict
            Backtest results
        """
        self.trades = []
        self.equity_curve = [1.0]  # Start with $1
        
        # Generate positions
        if self.mode == 'base':
            positions = self._base_mode_positions(predictions)
        elif self.mode == 'smc_fvg_confluence':
            positions = apply_confluence_filter(
                predictions,
                features,
                self.confluence_config,
                max(self.threshold_up, self.threshold_down)
            )
        else:
            raise ValueError(f"Unknown backtest mode: {self.mode}")
        
        # Simulate trades
        current_position = 0
        entry_price = 0
        entry_bar = 0
        
        for i in range(len(positions) - 1):
            signal = positions[i]
            current_price = prices[i]
            next_price = prices[i + 1]
            
            # Close existing position if signal changes
            if current_position != 0 and signal != current_position:
                exit_price = current_price
                pnl = self._calculate_pnl(
                    entry_price,
                    exit_price,
                    current_position
                )
                
                self.trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': current_position,
                    'pnl': pnl,
                    'pnl_pct': (pnl / entry_price) * 100 if entry_price > 0 else 0
                })
                
                # Update equity
                self.equity_curve.append(self.equity_curve[-1] * (1 + pnl))
                current_position = 0
            
            # Open new position
            if current_position == 0 and signal != 0:
                entry_price = current_price
                entry_bar = i
                current_position = signal
        
        # Close any remaining position
        if current_position != 0:
            exit_price = prices[-1]
            pnl = self._calculate_pnl(
                entry_price,
                exit_price,
                current_position
            )
            
            self.trades.append({
                'entry_bar': entry_bar,
                'exit_bar': len(positions) - 1,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': current_position,
                'pnl': pnl,
                'pnl_pct': (pnl / entry_price) * 100 if entry_price > 0 else 0
            })
            
            self.equity_curve.append(self.equity_curve[-1] * (1 + pnl))
        
        # Compute metrics
        results = self._compute_metrics()
        
        return results
    
    def _base_mode_positions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Generate positions using base mode (threshold-based).
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
            
        Returns
        -------
        np.ndarray
            Position array (-1, 0, +1)
        """
        positions = np.zeros(len(predictions))
        
        for i in range(len(predictions)):
            max_prob = predictions[i].max()
            pred_class = predictions[i].argmax()
            
            if pred_class == 2 and max_prob >= self.threshold_up:
                positions[i] = 1  # Long
            elif pred_class == 0 and max_prob >= self.threshold_down:
                positions[i] = -1  # Short
            # else: neutral (0)
        
        return positions
    
    def _calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        direction: int
    ) -> float:
        """
        Calculate PnL with costs.
        
        Parameters
        ----------
        entry_price : float
            Entry price
        exit_price : float
            Exit price
        direction : int
            Position direction (+1 for long, -1 for short)
            
        Returns
        -------
        float
            PnL as decimal (e.g., 0.01 for 1%)
        """
        # Raw PnL
        if direction > 0:  # Long
            raw_pnl = exit_price - entry_price
        else:  # Short
            raw_pnl = entry_price - exit_price
        
        # Apply costs
        total_cost = (self.spread_pips + self.slippage_pips) * self.pip_size
        raw_pnl -= total_cost
        
        # Commission
        if self.commission_pct > 0:
            raw_pnl -= (entry_price * self.commission_pct / 100)
        
        # Convert to percentage
        pnl_pct = (raw_pnl / entry_price)
        
        # Apply position sizing
        pnl_pct *= (self.position_size_pct / 100)
        
        return pnl_pct
    
    def _compute_metrics(self) -> Dict:
        """
        Compute backtest metrics.
        
        Returns
        -------
        Dict
            Dictionary of metrics
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] < 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        total_pnl = (self.equity_curve[-1] - 1) * 100
        
        # Returns array
        returns = trades_df['pnl'].values
        
        # Sharpe ratio
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino = 0
        
        # Max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max
        max_dd = drawdown.max() * 100
        
        # Calmar ratio
        calmar = (total_pnl / max_dd) if max_dd > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'avg_win': gross_profit / winning_trades if winning_trades > 0 else 0,
            'avg_loss': gross_loss / losing_trades if losing_trades > 0 else 0
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        return pd.DataFrame(self.trades)
    
    def get_equity_curve(self) -> np.ndarray:
        """Get equity curve."""
        return np.array(self.equity_curve)
