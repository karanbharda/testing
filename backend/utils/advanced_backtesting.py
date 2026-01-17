#!/usr/bin/env python3
"""
Advanced Backtesting and Validation Suite
===========================================

Comprehensive backtesting framework with:
- Walk-forward analysis
- Monte Carlo simulation
- Stress testing
- Out-of-sample validation
- Risk metrics calculation
- Drawdown analysis
- Parameter optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting metrics"""
    total_return: float  # Total return %
    annual_return: float  # Annualized return %
    sharpe_ratio: float  # Risk-adjusted return
    sortino_ratio: float  # Downside risk-adjusted return
    max_drawdown: float  # Maximum drawdown
    win_rate: float  # % of winning trades
    profit_factor: float  # Gross profit / Gross loss
    trades_count: int  # Total trades
    avg_trade_pnl: float  # Average trade P&L
    largest_win: float  # Largest winning trade
    largest_loss: float  # Largest losing trade
    
    # Risk metrics
    volatility: float  # Annual volatility
    calmar_ratio: float  # Return / Max Drawdown
    recovery_factor: float  # Total return / Max Drawdown
    
    # Time-based metrics
    win_streak: int  # Longest winning streak
    lose_streak: int  # Longest losing streak
    avg_holding_period: float  # Average days in position
    
    # Stability metrics
    stability_score: float  # 0-1 score for strategy stability
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_return': round(self.total_return, 4),
            'annual_return': round(self.annual_return, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'sortino_ratio': round(self.sortino_ratio, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'win_rate': round(self.win_rate, 4),
            'profit_factor': round(self.profit_factor, 4),
            'trades_count': self.trades_count,
            'avg_trade_pnl': round(self.avg_trade_pnl, 4),
            'largest_win': round(self.largest_win, 4),
            'largest_loss': round(self.largest_loss, 4),
            'volatility': round(self.volatility, 4),
            'calmar_ratio': round(self.calmar_ratio, 4),
            'recovery_factor': round(self.recovery_factor, 4),
            'stability_score': round(self.stability_score, 4)
        }


class AdvancedBacktester:
    """
    Advanced backtesting with comprehensive validation
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting portfolio value
        """
        self.initial_capital = initial_capital
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
        logger.info(f"Advanced Backtester initialized with ${initial_capital:,.2f}")
    
    def backtest(self, signals: pd.Series, prices: pd.Series, 
                positions_size: float = 0.1, slippage: float = 0.0001,
                commission: float = 0.001) -> BacktestMetrics:
        """
        Run backtest on signals and price data
        
        Args:
            signals: Series of signals (-1, 0, 1) for (SELL, HOLD, BUY)
            prices: Series of prices
            positions_size: Size of each position (fraction of portfolio)
            slippage: Slippage as fraction of price
            commission: Commission as fraction of trade value
        
        Returns:
            BacktestMetrics object
        """
        if len(signals) != len(prices):
            raise ValueError("Signals and prices must have same length")
        
        # Initialize
        position = 0  # Current position
        entry_price = 0
        entry_date = None
        cash = self.initial_capital
        portfolio_value = self.initial_capital
        
        self.equity_curve = [self.initial_capital]
        self.timestamps = [prices.index[0]]
        self.trades = []
        
        # Process signals
        for i in range(1, len(signals)):
            price = prices.iloc[i]
            signal = signals.iloc[i]
            timestamp = prices.index[i]
            
            # Update unrealized P&L
            if position != 0:
                unrealized_pnl = position * (price - entry_price)
                portfolio_value = cash + position * price
            else:
                portfolio_value = cash
            
            # Process trades
            if signal != 0 and position == 0:  # Enter position
                position_amount = int(portfolio_value * positions_size / price)
                if position_amount > 0:
                    slippage_cost = position_amount * price * slippage
                    commission_cost = position_amount * price * commission
                    
                    entry_price = price + (slippage if signal > 0 else -slippage)
                    cash -= position_amount * entry_price + commission_cost
                    position = position_amount * signal
                    entry_date = timestamp
            
            elif signal == 0 and position != 0:  # Exit position
                exit_price = price - (slippage if position > 0 else -slippage)
                pnl = position * (exit_price - entry_price)
                commission_cost = abs(position) * exit_price * commission
                
                cash += position * exit_price - commission_cost
                self.trades.append({
                    'entry_date': entry_date,
                    'exit_date': timestamp,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position,
                    'pnl': pnl,
                    'return_pct': pnl / (abs(position) * entry_price),
                    'days_held': (timestamp - entry_date).days
                })
                
                position = 0
                portfolio_value = cash
            
            self.equity_curve.append(portfolio_value)
            self.timestamps.append(timestamp)
        
        # Close final position
        if position != 0:
            exit_price = prices.iloc[-1]
            pnl = position * (exit_price - entry_price)
            self.trades.append({
                'entry_date': entry_date,
                'exit_date': prices.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position,
                'pnl': pnl,
                'return_pct': pnl / (abs(position) * entry_price),
                'days_held': (prices.index[-1] - entry_date).days
            })
        
        # Calculate metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Basic returns
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Annualized return
        trading_days = len(returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        daily_rf = 0.02 / 252  # 2% annual risk-free
        sharpe = (np.mean(returns) - daily_rf) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        # Sortino ratio (only downside deviation)
        negative_returns = returns[returns < 0]
        downside_dev = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino = (np.mean(returns) - daily_rf) / (downside_dev + 1e-10) * np.sqrt(252)
        
        # Drawdown
        cumulative_max = np.maximum.accumulate(equity)
        drawdown = (equity - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown)
        
        # Trade metrics
        if not self.trades:
            win_rate = 0
            profit_factor = 1
            avg_pnl = 0
            largest_win = 0
            largest_loss = 0
            win_streak = 0
            lose_streak = 0
            avg_holding = 0
        else:
            pnls = [t['pnl'] for t in self.trades]
            win_trades = [p for p in pnls if p > 0]
            lose_trades = [p for p in pnls if p < 0]
            
            win_rate = len(win_trades) / len(self.trades) if self.trades else 0
            
            gross_profit = sum(win_trades) if win_trades else 0
            gross_loss = abs(sum(lose_trades)) if lose_trades else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1
            
            avg_pnl = np.mean(pnls) if pnls else 0
            largest_win = max(pnls) if pnls else 0
            largest_loss = min(pnls) if pnls else 0
            
            # Streaks
            win_streak = self._calculate_streak(pnls, True)
            lose_streak = self._calculate_streak(pnls, False)
            
            holding_periods = [t['days_held'] for t in self.trades]
            avg_holding = np.mean(holding_periods) if holding_periods else 0
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Recovery factor
        total_pnl = sum([t['pnl'] for t in self.trades]) if self.trades else 0
        recovery = total_pnl / abs(total_pnl * max_drawdown) if max_drawdown != 0 else 0
        
        # Stability score (0-1)
        stability = min(1.0, (sharpe + sortino) / 4)  # Normalized
        
        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=abs(max_drawdown),
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades_count=len(self.trades),
            avg_trade_pnl=avg_pnl,
            largest_win=largest_win,
            largest_loss=largest_loss,
            volatility=volatility,
            calmar_ratio=calmar,
            recovery_factor=recovery,
            win_streak=win_streak,
            lose_streak=lose_streak,
            avg_holding_period=avg_holding,
            stability_score=max(0, stability)
        )
    
    @staticmethod
    def _calculate_streak(values: List[float], wins: bool = True) -> int:
        """Calculate longest streak of wins or losses"""
        max_streak = 0
        current_streak = 0
        
        for value in values:
            is_win = value > 0
            if (wins and is_win) or (not wins and not is_win):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def walk_forward_analysis(self, signals: pd.Series, prices: pd.Series,
                             train_periods: int = 252, test_periods: int = 63,
                             step_size: int = 21) -> Dict[str, Any]:
        """
        Walk-forward validation (more realistic backtesting)
        
        Args:
            signals: All signals
            prices: All prices
            train_periods: Training window size
            test_periods: Test window size
            step_size: Step size for rolling window
        
        Returns:
            Dictionary with walk-forward results
        """
        results = {
            'periods': [],
            'metrics': [],
            'avg_metrics': None
        }
        
        total_length = len(signals)
        
        idx = 0
        while idx + train_periods + test_periods <= total_length:
            train_end = idx + train_periods
            test_end = train_end + test_periods
            
            # Test period
            test_signals = signals.iloc[train_end:test_end]
            test_prices = prices.iloc[train_end:test_end]
            
            # Run backtest
            metrics = self.backtest(test_signals, test_prices)
            
            results['periods'].append({
                'start': test_prices.index[0],
                'end': test_prices.index[-1]
            })
            results['metrics'].append(metrics.to_dict())
            
            idx += step_size
        
        # Calculate average metrics
        if results['metrics']:
            avg_metrics = {}
            for key in results['metrics'][0].keys():
                values = [m[key] for m in results['metrics']]
                avg_metrics[key] = np.mean(values)
            results['avg_metrics'] = avg_metrics
        
        logger.info(f"Walk-forward analysis complete: {len(results['metrics'])} periods tested")
        
        return results
    
    def monte_carlo_simulation(self, returns: np.ndarray, n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Monte Carlo simulation for robustness analysis
        
        Args:
            returns: Returns series
            n_simulations: Number of simulations to run
        
        Returns:
            Simulation results
        """
        results = {
            'simulations': [],
            'percentile_5': 0,
            'percentile_25': 0,
            'median': 0,
            'percentile_75': 0,
            'percentile_95': 0
        }
        
        # Generate random return sequences
        for _ in range(n_simulations):
            # Resample returns with replacement
            sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate cumulative return
            cum_returns = np.cumprod(1 + sampled_returns) - 1
            final_return = cum_returns[-1]
            
            results['simulations'].append(final_return)
        
        # Calculate percentiles
        sorted_returns = np.sort(results['simulations'])
        results['percentile_5'] = np.percentile(sorted_returns, 5)
        results['percentile_25'] = np.percentile(sorted_returns, 25)
        results['median'] = np.percentile(sorted_returns, 50)
        results['percentile_75'] = np.percentile(sorted_returns, 75)
        results['percentile_95'] = np.percentile(sorted_returns, 95)
        
        logger.info(f"Monte Carlo simulation complete: {n_simulations} simulations")
        
        return results
    
    def stress_test(self, signals: pd.Series, prices: pd.Series,
                   shock_magnitude: float = 0.2) -> Dict[str, Any]:
        """
        Stress test strategy with price shocks
        
        Args:
            signals: Trading signals
            prices: Price data
            shock_magnitude: Magnitude of price shock (e.g., 0.2 for 20%)
        
        Returns:
            Stress test results
        """
        results = {
            'baseline_metrics': None,
            'shock_scenarios': []
        }
        
        # Baseline
        baseline = self.backtest(signals, prices)
        results['baseline_metrics'] = baseline.to_dict()
        
        # Create shocks
        scenarios = [
            {'name': 'Positive Shock', 'multiplier': 1 + shock_magnitude},
            {'name': 'Negative Shock', 'multiplier': 1 - shock_magnitude},
            {'name': 'Volatility Surge', 'multiplier': 1 + (shock_magnitude / 2)}
        ]
        
        for scenario in scenarios:
            shocked_prices = prices * scenario['multiplier']
            metrics = self.backtest(signals, shocked_prices)
            
            results['shock_scenarios'].append({
                'scenario': scenario['name'],
                'metrics': metrics.to_dict(),
                'return_impact': metrics.total_return - baseline.total_return
            })
        
        logger.info(f"Stress testing complete: {len(results['shock_scenarios'])} scenarios")
        
        return results
    
    def generate_report(self, metrics: BacktestMetrics) -> str:
        """Generate human-readable backtest report"""
        report = f"\n{'='*60}\nBACKTEST REPORT\n{'='*60}\n"
        
        report += f"\nPerformance Metrics:\n"
        report += f"  Total Return: {metrics.total_return:+.2%}\n"
        report += f"  Annual Return: {metrics.annual_return:+.2%}\n"
        report += f"  Volatility: {metrics.volatility:.2%}\n"
        report += f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n"
        report += f"  Sortino Ratio: {metrics.sortino_ratio:.2f}\n"
        
        report += f"\nRisk Metrics:\n"
        report += f"  Max Drawdown: {metrics.max_drawdown:-.2%}\n"
        report += f"  Calmar Ratio: {metrics.calmar_ratio:.2f}\n"
        report += f"  Recovery Factor: {metrics.recovery_factor:.2f}\n"
        
        report += f"\nTrade Statistics:\n"
        report += f"  Total Trades: {metrics.trades_count}\n"
        report += f"  Win Rate: {metrics.win_rate:.2%}\n"
        report += f"  Profit Factor: {metrics.profit_factor:.2f}\n"
        report += f"  Avg Trade P&L: {metrics.avg_trade_pnl:+.2f}\n"
        report += f"  Largest Win: {metrics.largest_win:+.2f}\n"
        report += f"  Largest Loss: {metrics.largest_loss:+.2f}\n"
        report += f"  Avg Holding Period: {metrics.avg_holding_period:.1f} days\n"
        
        report += f"\nStrategy Stability: {metrics.stability_score:.2%}\n"
        report += f"\n{'='*60}\n"
        
        return report


# Singleton instance
_backtester = None

def get_backtester(initial_capital: float = 100000) -> AdvancedBacktester:
    """Get or create backtester instance"""
    global _backtester
    if _backtester is None:
        _backtester = AdvancedBacktester(initial_capital)
    return _backtester


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    backtester = get_backtester()
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(500) * 0.5),
        index=dates
    )
    
    # Generate simple signals
    sma_20 = prices.rolling(window=20).mean()
    signals = pd.Series(np.where(prices > sma_20, 1, -1), index=dates)
    
    # Run backtest
    metrics = backtester.backtest(signals, prices)
    
    # Print report
    report = backtester.generate_report(metrics)
    print(report)
    
    # Walk-forward analysis
    wf_results = backtester.walk_forward_analysis(signals, prices)
    print(f"Walk-forward avg return: {wf_results['avg_metrics'].get('total_return', 0):+.2%}")
