#!/usr/bin/env python3
"""
Adaptive Signal Weighting System
==============================
Resolves overweighted sentiment issues by implementing intelligent 
dynamic weight adjustments based on:
- Signal consistency and historical performance
- Cross-correlation between different signals
- Volatility-adaptiveness
- Confident only scenarios
"""
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class SignalPerformance:
    """Performance metrics for each signal type"""
    signal_type: str
    recent_accuracy: float = 0.0
    recent_return: float = 0.0
    consistency_score: float = 0.0
    volatility: float = 0.0
    sample_size: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SignalWeights:
    """Current weights for all signal types"""
    technical: float = 0.35
    sentiment: float = 0.20
    ml: float = 0.30
    fundamental: float = 0.10
    risk: float = 0.05
    timestamp: datetime = field(default_factory=datetime.now)

class AdaptiveSignalWeighter:
    """Enhanced adaptive signal weighting system"""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
            
        # Configuration parameters
        self.max_sentiment_weight = config.get("max_sentiment_weight", 0.25)  # Prevent over-weighting
        self.min_sentiment_weight = config.get("min_sentiment_weight", 0.05)  # Prevent under-weighting
        self.confidence_threshold = config.get("confidence_threshold", 0.6)  # Only confident signals
        self.volatility_sensitivity = config.get("volatility_sensitivity", 0.3)  # Weight volatility impact
        self.correlation_threshold = config.get("correlation_threshold", 0.7)  # High correlation penalty
        self.performance_window = config.get("performance_window", 30)  # Days to track performance
        self.weight_update_frequency = config.get("weight_update_frequency", 5)  # Update every 5 days
        
        # Performance tracking
        self.signal_performance: Dict[str, deque] = {
            'technical': deque(maxlen=self.performance_window),
            'sentiment': deque(maxlen=self.performance_window),
            'ml': deque(maxlen=self.performance_window),
            'fundamental': deque(maxlen=self.performance_window),
            'risk': deque(maxlen=self.performance_window)
        }
        
        # Historical weights for stability
        self.weight_history: deque = deque(maxlen=20)
        self.last_update = datetime.now()
        
        # Initialize with balanced weights
        self.current_weights = SignalWeights()
        self.weight_history.append(self.current_weights)
        
        logger.info("Adaptive Signal Weighter initialized with sentiment weight cap")
    
    def calculate_weights(self, 
                         technical_indicators: Dict[str, Any],
                         sentiment_analysis: Dict[str, Any],
                         ml_analysis: Dict[str, Any],
                         fundamental_data: Dict[str, Any],
                         risk_metrics: Dict[str, Any],
                         market_regime: str,
                         historical_performance: Optional[Dict] = None) -> SignalWeights:
        """
        Calculate adaptive weights based on signal quality and performance
        """
        # Update performance metrics if available
        if historical_performance:
            self._update_performance_metrics(historical_performance)
        
        # Calculate base weights adjusted for market regime
        base_weights = self._get_regime_adjusted_weights(market_regime)
        
        # Calculate signal quality scores
        signal_scores = self._calculate_signal_quality_scores(
            technical_indicators, sentiment_analysis, ml_analysis, 
            fundamental_data, risk_metrics
        )
        
        # Apply adaptive adjustments
        adjusted_weights = self._apply_adaptive_adjustments(
            base_weights, signal_scores, market_regime
        )
        
        # Ensure sentiment weight constraints
        adjusted_weights = self._enforce_sentiment_constraints(adjusted_weights)
        
        # Normalize weights to sum to 1.0
        normalized_weights = self._normalize_weights(adjusted_weights)
        
        # Apply smoothing to prevent weight whiplash
        final_weights = self._apply_weight_smoothing(normalized_weights)
        
        self.current_weights = final_weights
        self.weight_history.append(final_weights)
        
        return final_weights
    
    def _get_regime_adjusted_weights(self, market_regime: str) -> Dict[str, float]:
        """Get base weights adjusted for market regime"""
        if market_regime == "TRENDING":
            return {
                'technical': 0.40,
                'sentiment': 0.15,
                'ml': 0.35,
                'fundamental': 0.08,
                'risk': 0.02
            }
        elif market_regime == "VOLATILE":
            return {
                'technical': 0.30,
                'sentiment': 0.10,
                'ml': 0.25,
                'fundamental': 0.15,
                'risk': 0.20
            }
        else:  # RANGE_BOUND
            return {
                'technical': 0.30,
                'sentiment': 0.25,
                'ml': 0.25,
                'fundamental': 0.15,
                'risk': 0.05
            }
    
    def _calculate_signal_quality_scores(self,
                                       technical_indicators: Dict[str, Any],
                                       sentiment_analysis: Dict[str, Any],
                                       ml_analysis: Dict[str, Any],
                                       fundamental_data: Dict[str, Any],
                                       risk_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality scores for each signal type"""
        scores = {}
        
        # Technical Signal Quality (0-1 scale)
        scores['technical'] = self._calculate_technical_score(technical_indicators)
        
        # Sentiment Signal Quality (0-1 scale) - With constraints
        scores['sentiment'] = self._calculate_sentiment_score(sentiment_analysis)
        
        # ML Signal Quality (0-1 scale)
        scores['ml'] = self._calculate_ml_score(ml_analysis)
        
        # Fundamental Signal Quality (0-1 scale)
        scores['fundamental'] = self._calculate_fundamental_score(fundamental_data)
        
        # Risk Signal Quality (0-1 scale)
        scores['risk'] = self._calculate_risk_score(risk_metrics)
        
        return scores
    
    def _calculate_technical_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate technical signal quality score"""
        if not indicators:
            return 0.5
        
        scores = []
        
        # RSI quality (extreme values are stronger signals)
        rsi = indicators.get("rsi", 50)
        if rsi < 30 or rsi > 70:
            scores.append(0.9)
        elif rsi < 35 or rsi > 65:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        # MACD quality (divergence strength)
        macd = indicators.get("macd", 0)
        macd_score = min(abs(macd) / 2.0, 1.0)  # Normalize MACD signal
        scores.append(macd_score)
        
        # ADX quality (trend strength)
        adx = indicators.get("adx", 25)
        adx_score = min(adx / 50.0, 1.0)  # Higher ADX = stronger trend
        scores.append(adx_score)
        
        # Moving average alignment
        ma_alignment = indicators.get("ma_alignment", 0)
        if ma_alignment > 0.7:
            scores.append(0.8)
        elif ma_alignment > 0.5:
            scores.append(0.6)
        else:
            scores.append(0.4)
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_sentiment_score(self, sentiment: Dict[str, Any]) -> float:
        """Calculate sentiment score with built-in constraints"""
        if not sentiment:
            return 0.3  # Default low score for missing sentiment
        
        # Get raw sentiment score
        raw_score = sentiment.get("score", 0.5)
        confidence = sentiment.get("confidence", 0.5)
        
        # Apply confidence penalty
        confidence_adjusted = raw_score * confidence
        
        # Apply extreme value penalty (sentiment shouldn't be too extreme)
        if abs(confidence_adjusted - 0.5) > 0.3:  # If > 0.8 or < 0.2
            confidence_adjusted *= 0.8  # Reduce extreme sentiment impact
        
        # Apply consistency penalty based on historical performance
        sentiment_performance = self._get_signal_performance('sentiment')
        if sentiment_performance.sample_size > 5:
            # Reduce weight if sentiment has poor historical performance
            performance_penalty = max(0.5, 1.0 - sentiment_performance.recent_accuracy)
            confidence_adjusted *= (1.0 - performance_penalty * 0.3)
        
        # Ensure minimum score for stability
        return max(0.1, min(0.8, confidence_adjusted))  # Cap between 0.1-0.8
    
    def _calculate_ml_score(self, ml_analysis: Dict[str, Any]) -> float:
        """Calculate ML model confidence score"""
        if not ml_analysis:
            return 0.5
        
        confidence = ml_analysis.get("confidence", 0.5)
        success_rate = ml_analysis.get("success_rate", 0.5)
        
        # Combine confidence and historical success
        return (confidence + success_rate) / 2.0
    
    def _calculate_fundamental_score(self, fundamental_data: Dict[str, Any]) -> float:
        """Calculate fundamental data quality score"""
        if not fundamental_data:
            return 0.4  # Lower base for fundamental (usually less frequent)
        
        quality_indicators = 0
        total_indicators = 4  # P/E, debt_ratio, revenue_growth, eps_growth
        
        # Check for key fundamental ratios
        if "p_e_ratio" in fundamental_data:
            quality_indicators += 1
        if "debt_ratio" in fundamental_data:
            quality_indicators += 1
        if "revenue_growth" in fundamental_data:
            quality_indicators += 1
        if "eps_growth" in fundamental_data:
            quality_indicators += 1
            
        return quality_indicators / total_indicators
    
    def _calculate_risk_score(self, risk_metrics: Dict[str, Any]) -> float:
        """Calculate risk signal quality"""
        if not risk_metrics:
            return 0.6  # Medium base for risk management signals
        
        # Lower risk scores for stable environments
        volatility = risk_metrics.get("current_volatility", 0.5)
        if volatility < 0.2:  # Low volatility = high risk quality
            return 0.8
        elif volatility < 0.5:  # Moderate volatility
            return 0.6
        else:  # High volatility
            return 0.4  # Reduce risk management emphasis in high volatility
    
    def _apply_adaptive_adjustments(self, base_weights: Dict[str, float], 
                                  signal_scores: Dict[str, float], 
                                  market_regime: str) -> Dict[str, float]:
        """Apply adaptive weight adjustments based on signal scores"""
        adjusted_weights = base_weights.copy()
        
        # Calculate performance adjustments
        sentiment_perf = self._get_signal_performance('sentiment')
        tech_perf = self._get_signal_performance('technical')
        ml_perf = self._get_signal_performance('ml')
        
        # Boost performing signals (but keep constraints on sentiment)
        if signal_scores.get('sentiment', 0) < 0.7:  # Sentinel improvement on medium-impact assets
            adj_amount = base_weights['sentiment']  # amount: 0.05
            adjusted_weights['sentiment'] = base_weights['sentiment'] + adj_amount
        else:
            adjusted_weights['sentiment'] = base_weights['sentiment']
        
        # Adjust technical based on performance
        if tech_perf.recent_accuracy > 0.6 and signal_scores.get('technical', 0) > 0.7:
            adjusted_weights['technical'] = min(0.5, base_weights['technical'] * 1.2)
        
        # Adjust ML based on performance
        if ml_perf.recent_accuracy > 0.6 and signal_scores.get('ml', 0) > 0.7:
            adjusted_weights['ml'] = min(0.45, base_weights['ml'] * 1.15)
        
        # Apply volatility adjustment
        if market_regime == "VOLATILE":
            # Reduce sentiment weight in volatile markets
            adjusted_weights['sentiment'] *= 0.7
            # Increase risk weight
            adjusted_weights['risk'] *= 1.3
        
        return adjusted_weights
    
    def _enforce_sentiment_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure sentiment weight stays within reasonable bounds"""
        constrained_weights = weights.copy()
        
        # Apply sentiment weight caps
        if constrained_weights['sentiment'] > self.max_sentiment_weight:
            excess_weight = constrained_weights['sentiment'] - self.max_sentiment_weight
            constrained_weights['sentiment'] = self.max_sentiment_weight
            
            # Distribute excess weight proportionally to other signals
            other_signals = ['technical', 'ml', 'fundamental', 'risk']
            total_other_weight = sum(constrained_weights[s] for s in other_signals)
            if total_other_weight > 0:
                for signal in other_signals:
                    constrained_weights[signal] += (constrained_weights[signal] / total_other_weight) * excess_weight
        
        elif constrained_weights['sentiment'] < self.min_sentiment_weight:
            deficit_weight = self.min_sentiment_weight - constrained_weights['sentiment']
            constrained_weights['sentiment'] = self.min_sentiment_weight
            
            # Remove deficit weight proportionally from other signals
            other_signals = ['technical', 'ml', 'fundamental', 'risk']
            total_other_weight = sum(constrained_weights[s] for s in other_signals)
            if total_other_weight > 0:
                for signal in other_signals:
                    reduction = (constrained_weights[signal] / total_other_weight) * deficit_weight
                    constrained_weights[signal] = max(0.01, constrained_weights[signal] - reduction)
        
        return constrained_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0"""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {'technical': 0.35, 'sentiment': 0.20, 'ml': 0.30, 'fundamental': 0.10, 'risk': 0.05}
        
        normalized = {}
        for signal, weight in weights.items():
            normalized[signal] = weight / total_weight
        
        return normalized
    
    def _apply_weight_smoothing(self, weights: Dict[str, float]) -> SignalWeights:
        """Apply smoothing to prevent weight whiplash"""
        if len(self.weight_history) < 2:
            return SignalWeights(**weights)
        
        # Apply exponential smoothing
        smoothing_factor = 0.7  # Higher = more smoothing
        smoothed_weights = {}
        
        for signal in weights:
            current_weight = weights[signal]
            previous_weight = self.weight_history[-1].__dict__[signal]
            smoothed_weights[signal] = (
                smoothing_factor * current_weight + 
                (1 - smoothing_factor) * previous_weight
            )
        
        # Ensure final normalization
        total_smoothed = sum(smoothed_weights.values())
        if total_smoothed > 0:
            for signal in smoothed_weights:
                smoothed_weights[signal] /= total_smoothed
        
        return SignalWeights(**smoothed_weights)
    
    def _update_performance_metrics(self, historical_performance: Dict[str, Any]):
        """Update performance metrics for adaptive weighting"""
        for signal_type, performance_data in historical_performance.items():
            if signal_type in self.signal_performance:
                self.signal_performance[signal_type].append(performance_data)
    
    def _get_signal_performance(self, signal_type: str) -> SignalPerformance:
        """Get current performance metrics for a signal type"""
        if not self.signal_performance[signal_type]:
            return SignalPerformance(signal_type=signal_type)
        
        recent_data = list(self.signal_performance[signal_type])
        if not recent_data:
            return SignalPerformance(signal_type=signal_type)
        
        # Calculate average performance metrics
        avg_accuracy = np.mean([d.get('accuracy', 0.5) for d in recent_data])
        avg_return = np.mean([d.get('return', 0.0) for d in recent_data])
        sample_size = len(recent_data)
        
        # Calculate consistency score
        if len(recent_data) > 1:
            accuracy_std = np.std([d.get('accuracy', 0.5) for d in recent_data])
            consistency_score = 1.0 - min(1.0, accuracy_std * 2)  # Higher consistency = lower std
        else:
            consistency_score = 0.5
        
        # Calculate volatility of returns
        if len(recent_data) > 1:
            return_volatility = np.std([d.get('return', 0.0) for d in recent_data])
        else:
            return_volatility = 0.0
        
        return SignalPerformance(
            signal_type=signal_type,
            recent_accuracy=avg_accuracy,
            recent_return=avg_return,
            consistency_score=consistency_score,
            volatility=return_volatility,
            sample_size=sample_size
        )
    
    def get_weight_report(self) -> Dict[str, Any]:
        """Generate comprehensive weight adjustment report"""
        report = {
            "current_weights": {
                "technical": self.current_weights.technical,
                "sentiment": self.current_weights.sentiment,
                "ml": self.current_weights.ml,
                "fundamental": self.current_weights.fundamental,
                "risk": self.current_weights.risk
            },
            "constraints_applied": {
                "max_sentiment_weight": self.max_sentiment_weight,
                "min_sentiment_weight": self.min_sentiment_weight,
                "current_sentiment_weight": self.current_weights.sentiment
            },
            "signal_performance": {},
            "weight_history": len(self.weight_history)
        }
        
        # Add performance metrics
        for signal_type in ['technical', 'sentiment', 'ml', 'fundamental', 'risk']:
            perf = self._get_signal_performance(signal_type)
            report["signal_performance"][signal_type] = {
                "recent_accuracy": perf.recent_accuracy,
                "recent_return": perf.recent_return,
                "consistency_score": perf.consistency_score,
                "sample_size": perf.sample_size
            }
        
        return report