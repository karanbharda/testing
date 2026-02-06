#!/usr/bin/env python3
"""
Comprehensive Backtesting Pipeline
==================================
Production-grade backtesting system with:
- Multiple strategy validation methods
- Walk-forward analysis
- Cross-validation techniques
- Performance attribution
- Risk-adjusted metrics
- Realistic market simulation
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
from enum import Enum
import json
import os
from pathlib import Path

# Import existing backtesting utilities
from .advanced_backtesting import BacktestMetrics, AdvancedBacktester
from .model_validation import ModelValidator

logger = logging.getLogger(__name__)

class BacktestMode(Enum):
    """Different backtesting modes"""
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
    CROSS_VALIDATION = "cross_validation"
    OOS_VALIDATION = "out_of_sample"

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    entry_rules: Dict[str, Any]
    exit_rules: Dict[str, Any]
    risk_params: Dict[str, Any]
    position_sizing: str
    signal_weights: Optional[Dict[str, float]] = None

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    mode: BacktestMode
    metrics: BacktestMetrics
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    drawdown_curve: List[float]
    parameters_used: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class ComprehensiveBacktestingPipeline:
    """Production-grade comprehensive backtesting pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        
        # Configuration
        self.data_directory = Path(config.get("data_directory", "data/backtesting"))
        self.results_directory = Path(config.get("results_directory", "results/backtesting"))
        self.benchmark_symbol = config.get("benchmark_symbol", "NIFTY")
        
        # Performance requirements
        self.min_sharpe_ratio = config.get("min_sharpe_ratio", 1.0)
        self.max_drawdown_limit = config.get("max_drawdown_limit", -0.20)  # 20% max drawdown
        self.min_win_rate = config.get("min_win_rate", 0.50)  # 50% win rate
        self.min_profit_factor = config.get("min_profit_factor", 1.5)
        
        # Validation parameters
        self.walk_forward_window = config.get("walk_forward_window", 252)  # 1 year
        self.oos_validation_size = config.get("oos_validation_size", 0.2)  # 20% for testing
        self.monte_carlo_simulations = config.get("monte_carlo_simulations", 1000)
        
        # Ensure directories exist
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.results_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_validator = ModelValidator()
        self.results_cache: Dict[str, BacktestResult] = {}
        
        logger.info("Comprehensive Backtesting Pipeline initialized")
    
    def run_comprehensive_validation(self, 
                                   strategy_config: StrategyConfig,
                                   historical_data: pd.DataFrame,
                                   market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation including multiple testing methods
        """
        results = {}
        
        logger.info(f"Running comprehensive validation for strategy: {strategy_config.name}")
        
        # 1. Walk-forward analysis
        results["walk_forward"] = self._run_walk_forward_analysis(
            strategy_config, historical_data, market_data
        )
        
        # 2. Out-of-sample validation
        results["out_of_sample"] = self._run_out_of_sample_validation(
            strategy_config, historical_data, market_data
        )
        
        # 3. Cross-validation
        results["cross_validation"] = self._run_cross_validation(
            strategy_config, historical_data, market_data
        )
        
        # 4. Stress testing
        results["stress_test"] = self._run_stress_testing(
            strategy_config, historical_data, market_data
        )
        
        # 5. Monte Carlo simulation
        results["monte_carlo"] = self._run_monte_carlo_simulation(
            strategy_config, historical_data, market_data
        )
        
        # 6. Generate comprehensive report
        results["comprehensive_report"] = self._generate_comprehensive_report(
            strategy_config, results
        )
        
        # 7. Save results
        self._save_backtest_results(strategy_config.name, results)
        
        return results
    
    def _run_walk_forward_analysis(self, 
                                 strategy_config: StrategyConfig,
                                 historical_data: pd.DataFrame,
                                 market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run walk-forward analysis for robust validation"""
        try:
            logger.info("Running walk-forward analysis...")
            
            # Prepare data
            data_length = len(historical_data)
            window_size = self.walk_forward_window
            step_size = window_size // 4  # 25% overlap
            
            results = []
            start_idx = 0
            
            while start_idx + window_size <= data_length:
                # Define training and testing periods
                train_end = start_idx + int(window_size * 0.8)  # 80% training
                test_end = start_idx + window_size
                
                train_data = historical_data.iloc[start_idx:train_end]
                test_data = historical_data.iloc[train_end:test_end]
                
                if market_data is not None:
                    train_market = market_data.iloc[start_idx:train_end]
                    test_market = market_data.iloc[train_end:test_end]
                else:
                    train_market = test_market = None
                
                # Run backtest on test period
                backtester = AdvancedBacktester(
                    initial_capital=strategy_config.risk_params.get("initial_capital", 100000),
                    commission_rate=strategy_config.risk_params.get("commission_rate", 0.001),
                    slippage_rate=strategy_config.risk_params.get("slippage_rate", 0.0005)
                )
                
                trades = self._generate_trades(strategy_config, test_data, test_market)
                metrics = backtester.calculate_metrics(trades, test_data)
                
                results.append({
                    "period_start": test_data.index[0],
                    "period_end": test_data.index[-1],
                    "metrics": metrics,
                    "trade_count": len(trades)
                })
                
                start_idx += step_size
            
            # Calculate aggregate statistics
            if results:
                sharpe_ratios = [r["metrics"].sharpe_ratio for r in results]
                returns = [r["metrics"].total_return for r in results]
                drawdowns = [r["metrics"].max_drawdown for r in results]
                
                aggregate_metrics = {
                    "avg_sharpe": np.mean(sharpe_ratios),
                    "sharpe_std": np.std(sharpe_ratios),
                    "avg_return": np.mean(returns),
                    "max_drawdown": min(drawdowns),
                    "consistency_score": self._calculate_consistency_score(results),
                    "period_count": len(results),
                    "individual_periods": results
                }
                
                logger.info(f"Walk-forward analysis completed: {len(results)} periods")
                return aggregate_metrics
            else:
                logger.warning("No valid periods for walk-forward analysis")
                return {"error": "No valid periods"}
                
        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}")
            return {"error": str(e)}
    
    def _run_out_of_sample_validation(self,
                                    strategy_config: StrategyConfig,
                                    historical_data: pd.DataFrame,
                                    market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run out-of-sample validation"""
        try:
            logger.info("Running out-of-sample validation...")
            
            # Split data
            split_idx = int(len(historical_data) * (1 - self.oos_validation_size))
            train_data = historical_data.iloc[:split_idx]
            test_data = historical_data.iloc[split_idx:]
            
            if market_data is not None:
                train_market = market_data.iloc[:split_idx]
                test_market = market_data.iloc[split_idx:]
            else:
                train_market = test_market = None
            
            # Run backtest on test data only
            backtester = AdvancedBacktester(
                initial_capital=strategy_config.risk_params.get("initial_capital", 100000),
                commission_rate=strategy_config.risk_params.get("commission_rate", 0.001),
                slippage_rate=strategy_config.risk_params.get("slippage_rate", 0.0005)
            )
            
            trades = self._generate_trades(strategy_config, test_data, test_market)
            metrics = backtester.calculate_metrics(trades, test_data)
            
            # Validate against performance criteria
            validation_result = self._validate_performance_criteria(metrics)
            
            result = {
                "metrics": metrics,
                "validation_passed": validation_result["passed"],
                "validation_details": validation_result["details"],
                "trade_count": len(trades),
                "data_period": {
                    "start": str(test_data.index[0]),
                    "end": str(test_data.index[-1])
                }
            }
            
            logger.info(f"Out-of-sample validation: {'PASSED' if validation_result['passed'] else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Out-of-sample validation failed: {e}")
            return {"error": str(e)}
    
    def _run_cross_validation(self,
                            strategy_config: StrategyConfig,
                            historical_data: pd.DataFrame,
                            market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run time-series cross-validation"""
        try:
            logger.info("Running cross-validation...")
            
            # K-fold time series cross-validation
            k_folds = 5
            data_length = len(historical_data)
            fold_size = data_length // k_folds
            
            fold_results = []
            
            for fold in range(k_folds):
                # Define validation fold (last portion of each fold)
                val_start = fold * fold_size
                val_end = min((fold + 1) * fold_size, data_length)
                
                # Training data is everything except validation fold
                train_indices = list(range(0, val_start)) + list(range(val_end, data_length))
                train_data = historical_data.iloc[train_indices]
                
                val_data = historical_data.iloc[val_start:val_end]
                
                if market_data is not None:
                    train_market = market_data.iloc[train_indices]
                    val_market = market_data.iloc[val_start:val_end]
                else:
                    train_market = val_market = None
                
                # Run validation
                backtester = AdvancedBacktester(
                    initial_capital=strategy_config.risk_params.get("initial_capital", 100000),
                    commission_rate=strategy_config.risk_params.get("commission_rate", 0.001),
                    slippage_rate=strategy_config.risk_params.get("slippage_rate", 0.0005)
                )
                
                trades = self._generate_trades(strategy_config, val_data, val_market)
                metrics = backtester.calculate_metrics(trades, val_data)
                
                fold_results.append({
                    "fold": fold,
                    "metrics": metrics,
                    "trade_count": len(trades),
                    "period": {
                        "start": str(val_data.index[0]),
                        "end": str(val_data.index[-1])
                    }
                })
            
            # Calculate cross-validation statistics
            if fold_results:
                sharpe_scores = [f["metrics"].sharpe_ratio for f in fold_results]
                return_scores = [f["metrics"].total_return for f in fold_results]
                
                cv_metrics = {
                    "avg_sharpe": np.mean(sharpe_scores),
                    "sharpe_std": np.std(sharpe_scores),
                    "sharpe_cv": np.std(sharpe_scores) / (np.mean(sharpe_scores) + 1e-10),
                    "avg_return": np.mean(return_scores),
                    "return_std": np.std(return_scores),
                    "fold_count": len(fold_results),
                    "individual_folds": fold_results
                }
                
                logger.info(f"Cross-validation completed: {len(fold_results)} folds")
                return cv_metrics
            else:
                return {"error": "No valid folds"}
                
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {"error": str(e)}
    
    def _run_stress_testing(self,
                          strategy_config: StrategyConfig,
                          historical_data: pd.DataFrame,
                          market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run stress testing with extreme market scenarios"""
        try:
            logger.info("Running stress testing...")
            
            # Define stress scenarios
            scenarios = [
                {"name": "2008_crisis", "start": "2008-09-01", "end": "2009-03-01"},
                {"name": "covid_crash", "start": "2020-02-01", "end": "2020-05-01"},
                {"name": "high_volatility", "start": "2011-08-01", "end": "2011-10-01"},
                {"name": "dotcom_bubble", "start": "2000-03-01", "end": "2002-10-01"}
            ]
            
            scenario_results = []
            
            for scenario in scenarios:
                try:
                    # Filter data for scenario period
                    scenario_data = historical_data[
                        (historical_data.index >= scenario["start"]) & 
                        (historical_data.index <= scenario["end"])
                    ]
                    
                    if len(scenario_data) < 10:  # Minimum data requirement
                        continue
                    
                    if market_data is not None:
                        scenario_market = market_data[
                            (market_data.index >= scenario["start"]) & 
                            (market_data.index <= scenario["end"])
                        ]
                    else:
                        scenario_market = None
                    
                    # Run backtest on scenario
                    backtester = AdvancedBacktester(
                        initial_capital=strategy_config.risk_params.get("initial_capital", 100000),
                        commission_rate=strategy_config.risk_params.get("commission_rate", 0.001),
                        slippage_rate=strategy_config.risk_params.get("slippage_rate", 0.0005)
                    )
                    
                    trades = self._generate_trades(strategy_config, scenario_data, scenario_market)
                    metrics = backtester.calculate_metrics(trades, scenario_data)
                    
                    scenario_results.append({
                        "scenario": scenario["name"],
                        "metrics": metrics,
                        "trade_count": len(trades),
                        "period": {
                            "start": str(scenario_data.index[0]),
                            "end": str(scenario_data.index[-1])
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"Scenario {scenario['name']} failed: {e}")
                    continue
            
            # Analyze stress test results
            if scenario_results:
                stress_metrics = {
                    "scenario_count": len(scenario_results),
                    "max_drawdown_in_stress": min([s["metrics"].max_drawdown for s in scenario_results]),
                    "avg_sharpe_in_stress": np.mean([s["metrics"].sharpe_ratio for s in scenario_results]),
                    "worst_scenario": min(scenario_results, key=lambda x: x["metrics"].total_return),
                    "best_scenario": max(scenario_results, key=lambda x: x["metrics"].total_return),
                    "scenarios": scenario_results
                }
                
                logger.info(f"Stress testing completed: {len(scenario_results)} scenarios")
                return stress_metrics
            else:
                return {"error": "No valid scenarios"}
                
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            return {"error": str(e)}
    
    def _run_monte_carlo_simulation(self,
                                  strategy_config: StrategyConfig,
                                  historical_data: pd.DataFrame,
                                  market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run Monte Carlo simulation for robustness testing"""
        try:
            logger.info("Running Monte Carlo simulation...")
            
            # Calculate historical returns
            returns = historical_data['close'].pct_change().dropna()
            
            # Generate simulations
            simulation_length = len(returns)
            simulations = []
            
            for i in range(self.monte_carlo_simulations):
                # Bootstrap sampling with replacement
                simulated_returns = np.random.choice(returns, size=simulation_length, replace=True)
                
                # Create simulated price series
                simulated_prices = [historical_data['close'].iloc[0]]
                for ret in simulated_returns:
                    simulated_prices.append(simulated_prices[-1] * (1 + ret))
                
                simulated_df = pd.DataFrame({
                    'close': simulated_prices[1:],
                    'open': simulated_prices[:-1],
                    'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in simulated_prices[1:]],
                    'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in simulated_prices[1:]],
                    'volume': [np.random.normal(1000000, 300000) for _ in range(simulation_length)]
                }, index=historical_data.index[:simulation_length])
                
                # Run backtest on simulated data
                trades = self._generate_trades(strategy_config, simulated_df, None)
                
                if trades:  # Only include simulations with trades
                    backtester = AdvancedBacktester(
                        initial_capital=strategy_config.risk_params.get("initial_capital", 100000)
                    )
                    metrics = backtester.calculate_metrics(trades, simulated_df)
                    simulations.append(metrics)
            
            # Analyze simulation results
            if simulations:
                final_equities = [s.total_return for s in simulations]
                sharpe_ratios = [s.sharpe_ratio for s in simulations]
                max_drawdowns = [s.max_drawdown for s in simulations]
                
                mc_metrics = {
                    "simulation_count": len(simulations),
                    "percentile_5_equity": np.percentile(final_equities, 5),
                    "percentile_25_equity": np.percentile(final_equities, 25),
                    "percentile_50_equity": np.percentile(final_equities, 50),
                    "percentile_75_equity": np.percentile(final_equities, 75),
                    "percentile_95_equity": np.percentile(final_equities, 95),
                    "avg_sharpe": np.mean(sharpe_ratios),
                    "sharpe_std": np.std(sharpe_ratios),
                    "max_drawdown_avg": np.mean(max_drawdowns),
                    "probability_positive_return": len([e for e in final_equities if e > 0]) / len(final_equities)
                }
                
                logger.info(f"Monte Carlo simulation completed: {len(simulations)} valid simulations")
                return mc_metrics
            else:
                return {"error": "No valid simulations"}
                
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            return {"error": str(e)}
    
    def _generate_trades(self, 
                        strategy_config: StrategyConfig,
                        data: pd.DataFrame,
                        market_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Generate trades based on strategy rules"""
        # This is a simplified implementation - in practice, this would
        # integrate with your actual trading logic
        trades = []
        
        # Simple moving average crossover example
        if 'close' in data.columns:
            data = data.copy()
            data['sma_fast'] = data['close'].rolling(10).mean()
            data['sma_slow'] = data['close'].rolling(30).mean()
            
            position = 0
            for i in range(30, len(data)):
                current_price = data['close'].iloc[i]
                fast_sma = data['sma_fast'].iloc[i]
                slow_sma = data['sma_slow'].iloc[i]
                
                # Simple entry/exit logic
                if fast_sma > slow_sma and position <= 0:  # Buy signal
                    trades.append({
                        'timestamp': data.index[i],
                        'action': 'buy',
                        'price': current_price,
                        'quantity': 100  # Fixed quantity for example
                    })
                    position = 100
                elif fast_sma < slow_sma and position > 0:  # Sell signal
                    trades.append({
                        'timestamp': data.index[i],
                        'action': 'sell',
                        'price': current_price,
                        'quantity': 100
                    })
                    position = 0
        
        return trades
    
    def _validate_performance_criteria(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """Validate strategy against performance criteria"""
        validation_details = {
            "sharpe_ratio": {
                "value": metrics.sharpe_ratio,
                "threshold": self.min_sharpe_ratio,
                "passed": metrics.sharpe_ratio >= self.min_sharpe_ratio
            },
            "max_drawdown": {
                "value": metrics.max_drawdown,
                "threshold": self.max_drawdown_limit,
                "passed": metrics.max_drawdown >= self.max_drawdown_limit
            },
            "win_rate": {
                "value": getattr(metrics, 'win_rate', 0),
                "threshold": self.min_win_rate,
                "passed": getattr(metrics, 'win_rate', 0) >= self.min_win_rate
            },
            "profit_factor": {
                "value": getattr(metrics, 'profit_factor', 0),
                "threshold": self.min_profit_factor,
                "passed": getattr(metrics, 'profit_factor', 0) >= self.min_profit_factor
            }
        }
        
        all_passed = all(detail["passed"] for detail in validation_details.values())
        
        return {
            "passed": all_passed,
            "details": validation_details,
            "overall_score": sum(detail["passed"] for detail in validation_details.values()) / len(validation_details)
        }
    
    def _calculate_consistency_score(self, results: List[Dict]) -> float:
        """Calculate consistency score across periods"""
        if len(results) < 2:
            return 1.0
        
        sharpe_ratios = [r["metrics"].sharpe_ratio for r in results]
        returns = [r["metrics"].total_return for r in results]
        
        # Calculate coefficient of variation (lower is better)
        sharpe_cv = np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-10)
        return_cv = np.std(returns) / (np.mean(returns) + 1e-10)
        
        # Convert to consistency score (0-1, higher is better)
        consistency = 1.0 - min(1.0, (sharpe_cv + return_cv) / 2)
        return max(0.0, consistency)
    
    def _generate_comprehensive_report(self, 
                                     strategy_config: StrategyConfig,
                                     all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            "strategy_name": strategy_config.name,
            "timestamp": datetime.now().isoformat(),
            "validation_summary": {},
            "detailed_results": all_results,
            "recommendations": []
        }
        
        # Summarize validation results
        validation_passed = []
        for test_name, results in all_results.items():
            if test_name != "comprehensive_report" and "error" not in results:
                if isinstance(results, dict) and "validation_passed" in results:
                    validation_passed.append(results["validation_passed"])
                elif isinstance(results, dict) and "avg_sharpe" in results:
                    # For tests that return metrics, check if they meet criteria
                    avg_sharpe = results.get("avg_sharpe", 0)
                    validation_passed.append(avg_sharpe >= self.min_sharpe_ratio)
        
        report["validation_summary"] = {
            "total_tests": len(validation_passed),
            "passed_tests": sum(validation_passed),
            "pass_rate": sum(validation_passed) / len(validation_passed) if validation_passed else 0,
            "overall_status": "PASSED" if all(validation_passed) else "FAILED" if validation_passed else "INCOMPLETE"
        }
        
        # Generate recommendations
        if report["validation_summary"]["overall_status"] == "PASSED":
            report["recommendations"].append("Strategy passed all validation tests - ready for production")
        else:
            failed_tests = [test for test, passed in zip(all_results.keys(), validation_passed) if not passed]
            report["recommendations"].append(f"Strategy failed tests: {', '.join(failed_tests)}")
            report["recommendations"].append("Consider parameter optimization or strategy redesign")
        
        return report
    
    def _save_backtest_results(self, strategy_name: str, results: Dict[str, Any]):
        """Save backtest results to file"""
        try:
            filename = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.results_directory / filename
            
            # Convert to JSON-serializable format
            serializable_results = self._make_serializable(results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (BacktestMetrics, StrategyConfig)):
            return obj.__dict__
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

# Usage example
if __name__ == "__main__":
    # Example configuration
    config = {
        "data_directory": "data/backtesting",
        "results_directory": "results/backtesting",
        "min_sharpe_ratio": 1.0,
        "max_drawdown_limit": -0.20,
        "walk_forward_window": 252
    }
    
    # Initialize pipeline
    pipeline = ComprehensiveBacktestingPipeline(config)
    
    # Example strategy configuration
    strategy_config = StrategyConfig(
        name="sma_crossover_strategy",
        entry_rules={"type": "sma_crossover", "fast_period": 10, "slow_period": 30},
        exit_rules={"type": "sma_crossover", "fast_period": 10, "slow_period": 30},
        risk_params={"initial_capital": 100000, "max_position_size": 0.1},
        position_sizing="fixed_fractional"
    )
    
    # Run comprehensive validation (you would provide actual historical data)
    # results = pipeline.run_comprehensive_validation(strategy_config, historical_data)