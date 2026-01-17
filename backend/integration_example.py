#!/usr/bin/env python3
"""
Quick-Start Integration Example
================================

Demonstrates how to use the advanced ML and sentiment modules
in a real trading system.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Import new modules
from utils.advanced_ml_optimizer import get_advanced_ml_optimizer, ModelMetrics
from utils.advanced_sentiment_fusion import (
    get_sentiment_fusion_engine,
    SentimentSignal,
    SentimentDirection
)
from utils.advanced_scraping_pipeline import get_scraping_pipeline
from utils.advanced_feature_engineering import get_feature_engineer
from utils.model_explainability import (
    get_explainability_engine,
    ModelDecision
)
from utils.advanced_backtesting import get_backtester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedQuantTradingSystem:
    """
    Integrated trading system using all advanced modules
    """
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize integrated trading system"""
        logger.info("Initializing Integrated Quant Trading System...")
        
        # Core components
        self.ml_optimizer = get_advanced_ml_optimizer()
        self.sentiment_engine = get_sentiment_fusion_engine()
        self.scraper = get_scraping_pipeline()
        self.feature_engineer = get_feature_engineer()
        self.explainability = get_explainability_engine()
        self.backtester = get_backtester(initial_capital)
        
        # Models (placeholder - would be actual trained models)
        self.models = {
            'xgb': None,      # XGBoost model
            'lgb': None,      # LightGBM model
            'cb': None,       # CatBoost model
            'ensemble': None  # Ensemble model
        }
        
        logger.info("System initialized successfully")
    
    def register_models(self, models_dict: dict):
        """Register trained models"""
        self.models.update(models_dict)
        
        # Register with optimizer
        for name, model in models_dict.items():
            if model is not None:
                self.ml_optimizer.register_model(name, model)
        
        logger.info(f"Registered {len([m for m in models_dict.values() if m])} models")
    
    async def analyze_symbol(self, symbol: str, price_data: pd.DataFrame) -> dict:
        """
        Complete analysis pipeline for a symbol
        
        Args:
            symbol: Trading symbol
            price_data: DataFrame with OHLCV data
        
        Returns:
            Analysis results with decision
        """
        logger.info(f"\nAnalyzing {symbol}...")
        
        # Step 1: Detect market regime
        logger.info("  [1/6] Detecting market regime...")
        regime = self.ml_optimizer.detect_market_regime(price_data)
        
        # Step 2: Engineer features
        logger.info("  [2/6] Engineering features...")
        feature_set = self.feature_engineer.engineer_features(
            df=price_data,
            sentiment_series=None,  # Would add sentiment here
            include_pca=True,
            n_pca_components=20
        )
        
        # Step 3: Aggregate sentiment
        logger.info("  [3/6] Aggregating sentiment signals...")
        
        # Simulate sentiment signals (in real system, would fetch from sources)
        sentiment_signals = [
            SentimentSignal(
                'finbert_news',
                direction=0.7,
                confidence=0.85,
                weight=0.25,
                timestamp=datetime.now(),
                content="Positive news about company"
            ),
            SentimentSignal(
                'market_fear_greed',
                direction=0.5,
                confidence=0.75,
                weight=0.20,
                timestamp=datetime.now(),
                content="Moderate market sentiment"
            )
        ]
        
        for signal in sentiment_signals:
            self.sentiment_engine.add_sentiment_signal(signal)
        
        sentiment = self.sentiment_engine.aggregate_sentiment()
        sentiment_trend = self.sentiment_engine.calculate_sentiment_trend()
        
        # Step 4: Select best models for regime
        logger.info("  [4/6] Selecting best models for regime...")
        best_models = self.ml_optimizer.select_best_models_for_regime(regime, k=3)
        
        # Step 5: Get model predictions
        logger.info("  [5/6] Getting model predictions...")
        
        # Simulate predictions (in real system, would use actual features)
        predictions = {
            'xgb': price_data['close'].iloc[-1] * 1.02,      # 2% up
            'lgb': price_data['close'].iloc[-1] * 1.015,     # 1.5% up
            'cb': price_data['close'].iloc[-1] * 1.025,      # 2.5% up
        }
        
        # Get ensemble prediction
        ensemble_pred, confidence = self.ml_optimizer.ensemble_predictions(
            predictions,
            method='weighted',
            regime=regime
        )
        
        # Step 6: Create decision with explanation
        logger.info("  [6/6] Creating decision...")
        
        decision = ModelDecision(
            decision_id=f"{symbol}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type="BUY" if ensemble_pred > price_data['close'].iloc[-1] else "SELL",
            confidence=confidence,
            prediction_value=ensemble_pred,
            top_features={'rsi_10': 0.3, 'macd_20': 0.25, 'volume_signal': 0.2},
            feature_importance={'rsi_10': 0.4, 'macd_20': 0.3},
            model_contributions=predictions,
            ensemble_method='weighted',
            market_regime=regime.regime_type,
            sentiment_score=sentiment.overall,
            technical_score=0.75,
            volume_score=0.8
        )
        
        # Record decision
        self.explainability.record_decision(decision)
        
        # Generate explanation
        explanation = self.explainability.explain_decision(decision)
        
        # Compile results
        results = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'regime': regime.regime_type,
            'sentiment': {
                'overall': sentiment.overall,
                'direction': sentiment.direction.name,
                'confidence': sentiment.confidence,
                'consensus': sentiment.consensus_level,
                'trend': sentiment_trend['direction']
            },
            'prediction': {
                'value': ensemble_pred,
                'signal': decision.signal_type,
                'confidence': confidence
            },
            'explanation': explanation.explanation_text,
            'reasons': explanation.reason_breakdown,
            'risk_factors': explanation.risk_factors
        }
        
        return results
    
    def backtest_strategy(self, symbols: list, price_data_dict: dict,
                         signals_dict: dict) -> dict:
        """
        Backtest strategy across multiple symbols
        
        Args:
            symbols: List of symbols
            price_data_dict: Dict of {symbol: price_series}
            signals_dict: Dict of {symbol: signals_series}
        
        Returns:
            Backtest results
        """
        logger.info(f"\nBacktesting strategy on {len(symbols)} symbols...")
        
        results = {}
        
        for symbol in symbols:
            if symbol in signals_dict and symbol in price_data_dict:
                signals = signals_dict[symbol]
                prices = price_data_dict[symbol]
                
                metrics = self.backtester.backtest(signals, prices)
                
                results[symbol] = metrics.to_dict()
                
                logger.info(f"  {symbol}: Return {metrics.total_return:+.2%}, "
                           f"Sharpe {metrics.sharpe_ratio:.2f}, "
                           f"Win Rate {metrics.win_rate:.2%}")
        
        # Calculate aggregate metrics
        if results:
            returns = [r['total_return'] for r in results.values()]
            sharpes = [r['sharpe_ratio'] for r in results.values()]
            
            results['_aggregate'] = {
                'symbols_count': len(results),
                'avg_return': np.mean(returns),
                'avg_sharpe': np.mean(sharpes),
                'total_trades': sum(r['trades_count'] for r in results.values())
            }
        
        return results
    
    def generate_performance_report(self) -> str:
        """Generate performance report from decision history"""
        logger.info("\nGenerating performance report...")
        
        # Get summary
        summary = self.explainability.get_decision_summary(last_n=50)
        
        # Get accuracy metrics
        buy_accuracy = self.explainability.calculate_signal_type_accuracy("BUY")
        sell_accuracy = self.explainability.calculate_signal_type_accuracy("SELL")
        
        # Generate report
        report = "\n" + "="*60
        report += "\nPERFORMANCE REPORT\n"
        report += "="*60 + "\n"
        
        report += f"\nDecision Statistics (Last 50):\n"
        report += f"  Total Decisions: {summary.get('decision_count', 0)}\n"
        report += f"  Total P&L: {summary.get('total_pnl', 0):+.2f}\n"
        report += f"  Win Rate: {summary.get('win_rate', 0):.2%}\n"
        report += f"  Avg Confidence: {summary.get('avg_confidence', 0):.2%}\n"
        
        if buy_accuracy:
            report += f"\nBUY Signals:\n"
            report += f"  Accuracy: {buy_accuracy.get('accuracy', 0):.2%}\n"
            report += f"  Total P&L: {buy_accuracy.get('total_pnl', 0):+.2f}\n"
            report += f"  Sharpe Ratio: {buy_accuracy.get('sharpe_ratio', 0):.2f}\n"
        
        if sell_accuracy:
            report += f"\nSELL Signals:\n"
            report += f"  Accuracy: {sell_accuracy.get('accuracy', 0):.2%}\n"
            report += f"  Total P&L: {sell_accuracy.get('total_pnl', 0):+.2f}\n"
            report += f"  Sharpe Ratio: {sell_accuracy.get('sharpe_ratio', 0):.2f}\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report


async def main():
    """Main example execution"""
    
    # Initialize system
    system = IntegratedQuantTradingSystem(initial_capital=100000)
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    
    symbols = ['RELIANCE.NS', 'TCS.NS']
    
    price_data = {}
    signals = {}
    
    for symbol in symbols:
        # Simulate price data
        prices = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(252) * 0.5),
            'high': 102 + np.cumsum(np.random.randn(252) * 0.5),
            'low': 98 + np.cumsum(np.random.randn(252) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(252) * 0.5),
            'volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)
        
        price_data[symbol] = prices['close']
        
        # Simulate signals (would come from actual strategy)
        sma_20 = prices['close'].rolling(20).mean()
        signals[symbol] = pd.Series(
            np.where(prices['close'] > sma_20, 1, -1),
            index=dates
        )
        
        # Analyze each symbol
        results = await system.analyze_symbol(symbol, prices)
        
        logger.info(f"\nResults for {symbol}:")
        logger.info(f"  Regime: {results['regime']}")
        logger.info(f"  Signal: {results['prediction']['signal']} "
                   f"(Confidence: {results['prediction']['confidence']:.2%})")
        logger.info(f"  Sentiment: {results['sentiment']['direction']} "
                   f"({results['sentiment']['overall']:+.2f})")
        logger.info(f"  Explanation: {results['explanation']}")
    
    # Backtest strategy
    backtest_results = system.backtest_strategy(symbols, price_data, signals)
    
    logger.info("\nBacktest Results:")
    logger.info(f"  Symbols: {backtest_results.get('_aggregate', {}).get('symbols_count', 0)}")
    logger.info(f"  Avg Return: {backtest_results.get('_aggregate', {}).get('avg_return', 0):+.2%}")
    logger.info(f"  Avg Sharpe: {backtest_results.get('_aggregate', {}).get('avg_sharpe', 0):.2f}")
    logger.info(f"  Total Trades: {backtest_results.get('_aggregate', {}).get('total_trades', 0)}")
    
    # Generate report
    report = system.generate_performance_report()
    logger.info(report)


if __name__ == "__main__":
    asyncio.run(main())
