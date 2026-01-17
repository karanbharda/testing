#!/usr/bin/env python3
"""
Advanced ML Optimizer for Quantitative Trading
===============================================

Provides intelligent model selection, dynamic weighting, adaptive ensemble strategies,
and comprehensive decision explainability for prop trading systems.

Key Features:
- Regime-aware model switching (trending/volatile/range-bound)
- Correlation-based model diversity scoring
- Adaptive weighting based on recent performance
- Stacking and blending ensemble strategies
- Walk-forward validation for realistic backtesting
- Model explanation and decision tracking
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from datetime import datetime, timedelta
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""
    name: str
    mse: float
    mae: float
    rmse: float
    r2_score: float
    accuracy: float = None
    precision: float = None
    recall: float = None
    f1_score: float = None
    sharpe_ratio: float = None
    sortino_ratio: float = None
    max_drawdown: float = None
    win_rate: float = None
    prediction_speed_ms: float = None
    training_time_s: float = None
    
    def __post_init__(self):
        if self.rmse is None:
            self.rmse = np.sqrt(self.mse)
    
    @property
    def composite_score(self) -> float:
        """Calculate composite score (0-100)"""
        scores = []
        
        # Normalize R² (0-100 scale)
        r2_normalized = max(0, min(100, self.r2_score * 100))
        scores.append(r2_normalized * 0.4)  # 40% weight
        
        # Normalize inverse MAE (lower is better)
        mae_score = max(0, 100 - (self.mae * 10))
        scores.append(mae_score * 0.3)  # 30% weight
        
        # Sharpe ratio bonus (if available)
        if self.sharpe_ratio is not None:
            sharpe_score = min(100, (self.sharpe_ratio + 2) * 10)
            scores.append(sharpe_score * 0.2)  # 20% weight
        else:
            scores.append(50 * 0.2)  # Neutral score if not available
        
        # Speed bonus (lower prediction time better)
        if self.prediction_speed_ms is not None:
            speed_score = max(0, 100 - (self.prediction_speed_ms / 10))
            scores.append(speed_score * 0.1)  # 10% weight
        else:
            scores.append(50 * 0.1)
        
        return sum(scores)


@dataclass
class RegimeIndicators:
    """Market regime detection indicators"""
    volatility: float  # Current volatility level
    trend_strength: float  # 0-1, higher = stronger trend
    momentum: float  # Current price momentum
    mean_reversion_signal: float  # Strength of mean reversion
    correlation_score: float  # Correlation with market
    regime_type: str = "RANGE_BOUND"  # TRENDING, VOLATILE, RANGE_BOUND
    
    def detect_regime(self):
        """Detect market regime based on indicators"""
        if self.trend_strength > 0.6:
            self.regime_type = "TRENDING"
        elif self.volatility > 0.025:
            self.regime_type = "VOLATILE"
        else:
            self.regime_type = "RANGE_BOUND"
        return self.regime_type


class AdvancedMLOptimizer:
    """
    Advanced ML optimization and model selection for quantitative trading
    """
    
    def __init__(self, window_size: int = 252):
        """
        Initialize optimizer
        
        Args:
            window_size: Number of days for performance window (default 1 year = 252 trading days)
        """
        self.window_size = window_size
        self.model_performance_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=1000)
        self.regime_history = deque(maxlen=252)
        
        # Model registry
        self.models: Dict[str, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.model_predictions: Dict[str, List[float]] = {}
        
        # Ensemble weights
        self.ensemble_weights: Dict[str, float] = {}
        self.adaptive_weights: Dict[str, float] = {}
        
        # Regime-aware model mapping
        self.regime_models = {
            'TRENDING': ['xgb', 'lgb', 'cb'],  # Tree-based models excel in trends
            'VOLATILE': ['mlp', 'svr', 'ensemble'],  # Neural networks handle volatility
            'RANGE_BOUND': ['ensemble', 'stacking', 'rf']  # Ensembles stable in ranges
        }
        
        # Correlation tracking for diversity
        self.model_correlations: Dict[Tuple[str, str], float] = {}
        
        logger.info("Advanced ML Optimizer initialized")
    
    def register_model(self, name: str, model: Any, model_type: str = None):
        """Register a model for optimization"""
        self.models[name] = model
        self.model_predictions[name] = []
        if name not in self.ensemble_weights:
            self.ensemble_weights[name] = 1.0 / max(1, len(self.models))
        logger.info(f"Registered model: {name} (type: {model_type})")
    
    def update_model_metrics(self, name: str, metrics: ModelMetrics):
        """Update performance metrics for a model"""
        self.model_metrics[name] = metrics
        self.model_performance_history.append({
            'model': name,
            'timestamp': datetime.now(),
            'metrics': metrics,
            'composite_score': metrics.composite_score
        })
        logger.info(f"Updated metrics for {name}: R²={metrics.r2_score:.4f}, MAE={metrics.mae:.4f}, Score={metrics.composite_score:.2f}")
    
    def calculate_model_correlations(self) -> Dict[Tuple[str, str], float]:
        """Calculate correlation between model predictions for diversity scoring"""
        if len(self.model_predictions) < 2:
            return {}
        
        correlations = {}
        model_names = list(self.model_predictions.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                preds1 = np.array(self.model_predictions[model1])
                preds2 = np.array(self.model_predictions[model2])
                
                if len(preds1) > 0 and len(preds2) > 0:
                    correlation = np.corrcoef(preds1, preds2)[0, 1]
                    correlations[(model1, model2)] = correlation if not np.isnan(correlation) else 0.0
        
        self.model_correlations = correlations
        return correlations
    
    def calculate_diversity_score(self) -> float:
        """Calculate overall model ensemble diversity (lower correlation = higher diversity)"""
        if not self.model_correlations:
            return 0.5
        
        correlations = list(self.model_correlations.values())
        if not correlations:
            return 0.5
        
        avg_correlation = np.mean(correlations)
        diversity = 1 - avg_correlation  # Lower correlation = higher diversity
        return max(0, min(1, diversity))
    
    def detect_market_regime(self, price_data: pd.DataFrame, returns_col: str = 'returns') -> RegimeIndicators:
        """Detect current market regime from price data"""
        if len(price_data) < 20:
            return RegimeIndicators(
                volatility=0.02,
                trend_strength=0.5,
                momentum=0.0,
                mean_reversion_signal=0.5,
                correlation_score=0.5
            )
        
        returns = price_data[returns_col].tail(self.window_size).values
        
        # Calculate volatility
        volatility = np.std(returns)
        
        # Calculate trend strength using moving averages
        ma_short = price_data['close'].tail(20).mean() if 'close' in price_data.columns else price_data.iloc[:, -1].tail(20).mean()
        ma_long = price_data['close'].tail(50).mean() if 'close' in price_data.columns else price_data.iloc[:, -1].tail(50).mean()
        trend_strength = abs(ma_short - ma_long) / ma_long if ma_long > 0 else 0.5
        
        # Calculate momentum
        momentum = returns[-1] if len(returns) > 0 else 0.0
        
        # Mean reversion signal
        mean_reversion_signal = abs(returns[-5:].mean()) / (volatility + 1e-6)
        
        # Correlation (simplified)
        correlation_score = 0.5
        
        regime = RegimeIndicators(
            volatility=min(volatility, 0.1),  # Cap at 10%
            trend_strength=min(trend_strength, 1.0),
            momentum=momentum,
            mean_reversion_signal=min(mean_reversion_signal, 1.0),
            correlation_score=correlation_score
        )
        
        regime.detect_regime()
        self.regime_history.append(regime)
        
        logger.info(f"Detected regime: {regime.regime_type} (volatility={regime.volatility:.4f}, trend={regime.trend_strength:.4f})")
        return regime
    
    def select_best_models_for_regime(self, regime: RegimeIndicators, k: int = 3) -> List[str]:
        """Select best performing models for detected regime"""
        regime_models = self.regime_models.get(regime.regime_type, list(self.models.keys()))
        
        # Get available metrics
        available_models = [(name, self.model_metrics[name].composite_score) 
                           for name in regime_models if name in self.model_metrics]
        
        # Sort by composite score
        available_models.sort(key=lambda x: x[1], reverse=True)
        
        selected = [name for name, _ in available_models[:k]]
        
        if not selected:
            # Fallback to any available models
            selected = sorted(self.model_metrics.keys(), 
                            key=lambda x: self.model_metrics[x].composite_score, 
                            reverse=True)[:k]
        
        logger.info(f"Selected {len(selected)} models for {regime.regime_type}: {selected}")
        return selected
    
    def calculate_adaptive_weights(self, lookback_days: int = 30) -> Dict[str, float]:
        """Calculate adaptive weights based on recent performance"""
        if not self.model_performance_history:
            return {name: 1.0 / max(1, len(self.models)) for name in self.models}
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_history = [h for h in self.model_performance_history 
                         if h['timestamp'] > cutoff_date]
        
        if not recent_history:
            recent_history = list(self.model_performance_history)[-50:]
        
        # Calculate average score for each model
        model_scores = {}
        model_counts = {}
        
        for record in recent_history:
            model = record['model']
            score = record['composite_score']
            
            if model not in model_scores:
                model_scores[model] = 0.0
                model_counts[model] = 0
            
            model_scores[model] += score
            model_counts[model] += 1
        
        # Normalize scores
        adaptive_weights = {}
        total_weight = 0.0
        
        for model_name in self.models:
            if model_name in model_scores:
                avg_score = model_scores[model_name] / model_counts[model_name]
                weight = max(0.1, avg_score / 100.0)  # Min 0.1, normalized from 0-100
            else:
                weight = 0.5  # Default for new models
            
            adaptive_weights[model_name] = weight
            total_weight += weight
        
        # Normalize to sum to 1.0
        self.adaptive_weights = {name: weight / total_weight for name, weight in adaptive_weights.items()}
        
        logger.info(f"Adaptive weights: {[(name, f'{w:.3f}') for name, w in self.adaptive_weights.items()]}")
        return self.adaptive_weights
    
    def ensemble_predictions(self, model_predictions: Dict[str, float], 
                            method: str = 'weighted', regime: Optional[RegimeIndicators] = None) -> Tuple[float, float]:
        """
        Ensemble multiple model predictions
        
        Args:
            model_predictions: Dict of {model_name: prediction}
            method: 'weighted', 'median', 'mean', 'stacking'
            regime: Optional market regime for adaptive weighting
        
        Returns:
            Tuple of (ensemble_prediction, confidence_score)
        """
        if not model_predictions:
            return 0.0, 0.0
        
        predictions = np.array(list(model_predictions.values()))
        model_names = list(model_predictions.keys())
        
        # Update prediction history
        for name, pred in model_predictions.items():
            self.model_predictions[name].append(pred)
        
        if method == 'weighted':
            # Use adaptive weights if available
            weights = np.array([self.adaptive_weights.get(name, 1.0 / len(model_names)) 
                               for name in model_names])
            weights = weights / weights.sum()
            ensemble_pred = np.dot(predictions, weights)
            
            # Confidence based on weight concentration
            weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
            confidence = 1.0 - (weight_entropy / np.log(len(weights)))
        
        elif method == 'median':
            ensemble_pred = np.median(predictions)
            # Confidence based on prediction consensus
            pred_std = np.std(predictions)
            confidence = 1.0 / (1.0 + pred_std)
        
        elif method == 'mean':
            ensemble_pred = np.mean(predictions)
            pred_std = np.std(predictions)
            confidence = 1.0 / (1.0 + pred_std)
        
        elif method == 'stacking':
            # Use meta-learner approach
            ensemble_pred = self._stacking_ensemble(predictions, model_names)
            confidence = 0.7  # Stacking typically has good confidence
        
        else:
            ensemble_pred = np.mean(predictions)
            confidence = 0.5
        
        return ensemble_pred, confidence
    
    def _stacking_ensemble(self, predictions: np.ndarray, model_names: List[str]) -> float:
        """Meta-learner stacking ensemble"""
        # Use adaptive weights as meta-learner
        weights = np.array([self.adaptive_weights.get(name, 1.0 / len(model_names)) 
                           for name in model_names])
        weights = weights / weights.sum()
        return np.dot(predictions, weights)
    
    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray, 
                               train_size: int = 200, step_size: int = 50) -> Dict[str, Any]:
        """
        Walk-forward validation for realistic backtesting performance
        
        Args:
            X: Feature matrix
            y: Target values
            train_size: Training window size
            step_size: Step size for forward rolling
        
        Returns:
            Dict with validation results
        """
        if len(X) < train_size + step_size:
            logger.warning("Insufficient data for walk-forward validation")
            return {}
        
        results = {
            'model_scores': {name: [] for name in self.models},
            'dates': [],
            'ensemble_scores': []
        }
        
        # Walk forward
        for start_idx in range(0, len(X) - train_size - step_size, step_size):
            train_end = start_idx + train_size
            test_end = train_end + step_size
            
            X_train = X[start_idx:train_end]
            y_train = y[start_idx:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]
            
            # Train and evaluate each model
            for model_name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    results['model_scores'][model_name].append(rmse)
                except Exception as e:
                    logger.warning(f"Error in walk-forward for {model_name}: {e}")
            
            results['dates'].append(test_end)
        
        # Calculate summary statistics
        for model_name in self.models:
            scores = results['model_scores'][model_name]
            if scores:
                logger.info(f"{model_name} WF-Validation: Mean RMSE={np.mean(scores):.6f}, Std={np.std(scores):.6f}")
        
        return results
    
    def explain_decision(self, model_predictions: Dict[str, float], 
                        actual_value: Optional[float] = None) -> Dict[str, Any]:
        """Generate comprehensive decision explanation"""
        ensemble_pred, confidence = self.ensemble_predictions(model_predictions)
        
        explanation = {
            'ensemble_prediction': ensemble_pred,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'model_contributions': {},
            'decision_factors': []
        }
        
        # Model contributions
        weights = self.adaptive_weights
        for model_name, pred in model_predictions.items():
            weight = weights.get(model_name, 1.0 / len(model_predictions))
            contribution = pred * weight
            
            metrics = self.model_metrics.get(model_name)
            explanation['model_contributions'][model_name] = {
                'prediction': float(pred),
                'weight': float(weight),
                'contribution': float(contribution),
                'recent_score': float(metrics.composite_score) if metrics else None
            }
        
        # Decision factors
        diversity = self.calculate_diversity_score()
        explanation['decision_factors'] = [
            f"Model ensemble diversity: {diversity:.2%}",
            f"Prediction confidence: {confidence:.2%}",
            f"Number of models: {len(model_predictions)}",
        ]
        
        if actual_value is not None:
            error = abs(ensemble_pred - actual_value)
            explanation['error'] = float(error)
            explanation['decision_factors'].append(f"Prediction error: {error:.6f}")
        
        return explanation
    
    def save_optimizer_state(self, filepath: str):
        """Save optimizer state for persistence"""
        state = {
            'models': self.models,
            'model_metrics': self.model_metrics,
            'ensemble_weights': self.ensemble_weights,
            'adaptive_weights': self.adaptive_weights,
            'model_correlations': self.model_correlations,
            'regime_history': list(self.regime_history)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved optimizer state to {filepath}")
    
    def load_optimizer_state(self, filepath: str):
        """Load optimizer state from file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.model_metrics = state.get('model_metrics', {})
        self.ensemble_weights = state.get('ensemble_weights', {})
        self.adaptive_weights = state.get('adaptive_weights', {})
        self.model_correlations = state.get('model_correlations', {})
        
        logger.info(f"Loaded optimizer state from {filepath}")


# Singleton instance
_optimizer = None

def get_advanced_ml_optimizer() -> AdvancedMLOptimizer:
    """Get or create singleton optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = AdvancedMLOptimizer()
    return _optimizer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    optimizer = get_advanced_ml_optimizer()
    
    # Simulate model registration and evaluation
    from sklearn.ensemble import RandomForestRegressor
    
    # Register dummy models
    for model_name in ['xgb', 'lgb', 'rf', 'ensemble']:
        optimizer.register_model(model_name, RandomForestRegressor(n_estimators=10))
    
    # Register metrics
    for model_name in ['xgb', 'lgb', 'rf', 'ensemble']:
        metrics = ModelMetrics(
            name=model_name,
            mse=0.0001,
            mae=0.005,
            rmse=0.01,
            r2_score=0.95,
            sharpe_ratio=1.5
        )
        optimizer.update_model_metrics(model_name, metrics)
    
    # Calculate adaptive weights
    weights = optimizer.calculate_adaptive_weights()
    print(f"Adaptive weights: {weights}")
    
    # Test ensemble prediction
    test_preds = {'xgb': 100.5, 'lgb': 100.2, 'rf': 100.1, 'ensemble': 100.3}
    ensemble_pred, confidence = optimizer.ensemble_predictions(test_preds)
    print(f"Ensemble prediction: {ensemble_pred:.4f}, Confidence: {confidence:.2%}")
    
    # Explain decision
    explanation = optimizer.explain_decision(test_preds, actual_value=100.25)
    print(f"Decision explanation: {json.dumps(explanation, indent=2, default=str)}")

import json
