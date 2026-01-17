#!/usr/bin/env python3
"""
Model Explainability and Decision Tracking
============================================

Provides comprehensive decision explainability, model interpretation,
and decision auditing for quantitative trading systems.

Features:
- SHAP and permutation feature importance
- Decision path visualization
- Model prediction confidence tracking
- P&L attribution to model components
- Decision audit trail
- Backtesting with explanation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelDecision:
    """Individual model decision with full context"""
    decision_id: str  # Unique identifier
    timestamp: datetime
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    prediction_value: float
    
    # Explanation components
    top_features: Dict[str, float]  # Feature name -> contribution
    feature_importance: Dict[str, float]
    model_contributions: Dict[str, float]  # Model name -> prediction/weight
    ensemble_method: str
    
    # Context
    market_regime: Optional[str] = None
    sentiment_score: Optional[float] = None
    technical_score: Optional[float] = None
    volume_score: Optional[float] = None
    
    # Outcome tracking
    actual_outcome: Optional[float] = None  # Actual price change
    pnl: Optional[float] = None
    decision_accuracy: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str, indent=2)


@dataclass
class DecisionExplanation:
    """Comprehensive explanation of a decision"""
    decision_id: str
    explanation_text: str
    reason_breakdown: List[str]  # Key reasons
    confidence_factors: List[str]  # Factors affecting confidence
    risk_factors: List[str]  # Potential risks
    supporting_data: Dict[str, Any]  # Relevant metrics


class ModelExplainabilityEngine:
    """
    Provides comprehensive model explainability and decision tracking
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize explainability engine
        
        Args:
            history_size: Maximum decisions to keep in memory
        """
        self.decision_history = deque(maxlen=history_size)
        self.daily_pnl_attribution = {}  # Day -> {model: pnl}
        self.feature_importance_history = deque(maxlen=100)
        self.model_performance_history = {}  # Model -> [accuracies]
        
        logger.info("Model Explainability Engine initialized")
    
    def record_decision(self, decision: ModelDecision):
        """Record a model decision"""
        self.decision_history.append(decision)
        logger.debug(f"Recorded decision {decision.decision_id}: {decision.signal_type} ({decision.confidence:.2%})")
    
    def record_outcome(self, decision_id: str, actual_outcome: float, pnl: float):
        """Record outcome for a decision"""
        # Find and update decision
        for decision in self.decision_history:
            if decision.decision_id == decision_id:
                decision.actual_outcome = actual_outcome
                decision.pnl = pnl
                
                # Calculate accuracy
                if decision.signal_type == "BUY":
                    decision.decision_accuracy = 1.0 if actual_outcome > 0 else 0.0
                elif decision.signal_type == "SELL":
                    decision.decision_accuracy = 1.0 if actual_outcome < 0 else 0.0
                else:
                    decision.decision_accuracy = 0.5
                
                logger.info(f"Updated decision {decision_id}: outcome={actual_outcome:.4f}, pnl={pnl:.2f}")
                break
    
    def explain_decision(self, decision: ModelDecision) -> DecisionExplanation:
        """Generate comprehensive decision explanation"""
        # Build explanation text
        explanation_parts = []
        
        # Main signal
        if decision.confidence > 0.8:
            strength = "STRONG"
        elif decision.confidence > 0.6:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        explanation_parts.append(f"{strength} {decision.signal_type} signal with {decision.confidence:.2%} confidence")
        
        # Top features
        if decision.top_features:
            top_features_str = ", ".join([f"{name} ({contrib:.3f})" 
                                         for name, contrib in list(decision.top_features.items())[:3]])
            explanation_parts.append(f"Top features: {top_features_str}")
        
        # Market regime
        if decision.market_regime:
            explanation_parts.append(f"Market regime: {decision.market_regime}")
        
        # Sentiment alignment
        if decision.sentiment_score is not None:
            sentiment_direction = "bullish" if decision.sentiment_score > 0 else "bearish"
            explanation_parts.append(f"Sentiment: {sentiment_direction} ({decision.sentiment_score:.3f})")
        
        # Model consensus
        if decision.model_contributions:
            consensus_count = sum(1 for v in decision.model_contributions.values() if v > 0)
            total_models = len(decision.model_contributions)
            explanation_parts.append(f"Model consensus: {consensus_count}/{total_models} models agree")
        
        explanation_text = "; ".join(explanation_parts)
        
        # Build reason breakdown
        reasons = []
        
        # Technical reasons
        if decision.technical_score:
            if decision.technical_score > 0.6:
                reasons.append(f"Strong technical setup (score: {decision.technical_score:.2f})")
            elif decision.technical_score > 0.4:
                reasons.append(f"Moderate technical signal (score: {decision.technical_score:.2f})")
        
        # Volume reasons
        if decision.volume_score:
            if decision.volume_score > 0.6:
                reasons.append(f"High volume confirmation (score: {decision.volume_score:.2f})")
            elif decision.volume_score > 0.4:
                reasons.append(f"Volume support (score: {decision.volume_score:.2f})")
        
        # Sentiment reasons
        if decision.sentiment_score:
            if abs(decision.sentiment_score) > 0.6:
                reasons.append(f"Strong sentiment alignment ({decision.sentiment_score:+.2f})")
        
        # Feature reasons
        if decision.top_features:
            reasons.append(f"Feature analysis supports signal")
        
        # Confidence factors
        confidence_factors = []
        
        if decision.confidence > 0.8:
            confidence_factors.append(f"High model confidence ({decision.confidence:.2%})")
        
        if decision.market_regime == "TRENDING":
            confidence_factors.append("Trend-following models performing well")
        
        if decision.sentiment_score and abs(decision.sentiment_score) > 0.5:
            confidence_factors.append("Strong sentiment consensus")
        
        # Risk factors
        risk_factors = []
        
        if decision.confidence < 0.6:
            risk_factors.append("Low model confidence - higher risk")
        
        if decision.market_regime == "VOLATILE":
            risk_factors.append("High market volatility - use tighter stops")
        
        if decision.sentiment_score is not None and decision.sentiment_score == 0:
            risk_factors.append("Neutral sentiment - mixed signals")
        
        # Build supporting data
        supporting_data = {
            'prediction_value': float(decision.prediction_value),
            'top_features': decision.top_features,
            'model_count': len(decision.model_contributions),
            'ensemble_method': decision.ensemble_method
        }
        
        return DecisionExplanation(
            decision_id=decision.decision_id,
            explanation_text=explanation_text,
            reason_breakdown=reasons,
            confidence_factors=confidence_factors,
            risk_factors=risk_factors,
            supporting_data=supporting_data
        )
    
    def attribute_pnl_to_components(self, decision: ModelDecision) -> Dict[str, float]:
        """
        Attribute P&L to different components (models, features, sentiment)
        
        Args:
            decision: Model decision with outcome
        
        Returns:
            Dict of component -> attributed_pnl
        """
        if decision.pnl is None or decision.actual_outcome is None:
            return {}
        
        attribution = {}
        total_contribution = sum(abs(v) for v in decision.model_contributions.values())
        
        # Attribute based on model contributions
        for model_name, contribution in decision.model_contributions.items():
            weight = (abs(contribution) / total_contribution) if total_contribution > 0 else 0
            attribution[f"model_{model_name}"] = decision.pnl * weight
        
        # Attribute to features
        for feature_name, importance in list(decision.feature_importance.items())[:5]:
            total_importance = sum(abs(v) for v in decision.feature_importance.values())
            weight = (abs(importance) / total_importance) if total_importance > 0 else 0
            attribution[f"feature_{feature_name}"] = decision.pnl * weight
        
        # Attribute to sentiment if strong
        if decision.sentiment_score and abs(decision.sentiment_score) > 0.5:
            attribution['sentiment'] = decision.pnl * 0.15
        
        return attribution
    
    def calculate_model_accuracy(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate model accuracy from decision history
        
        Args:
            symbol: Optional specific symbol filter
        
        Returns:
            Dict of model -> accuracy
        """
        relevant_decisions = [d for d in self.decision_history 
                            if d.decision_accuracy is not None and 
                            (symbol is None or d.symbol == symbol)]
        
        if not relevant_decisions:
            return {}
        
        model_accuracies = {}
        
        for decision in relevant_decisions:
            for model_name in decision.model_contributions.keys():
                if model_name not in model_accuracies:
                    model_accuracies[model_name] = []
                model_accuracies[model_name].append(decision.decision_accuracy)
        
        return {model: np.mean(accuracies) for model, accuracies in model_accuracies.items()}
    
    def calculate_signal_type_accuracy(self, signal_type: str = "BUY") -> Dict[str, Any]:
        """
        Calculate accuracy for specific signal type
        
        Args:
            signal_type: BUY, SELL, or HOLD
        
        Returns:
            Accuracy metrics
        """
        relevant_decisions = [d for d in self.decision_history 
                            if d.signal_type == signal_type and d.decision_accuracy is not None]
        
        if not relevant_decisions:
            return {}
        
        accuracies = [d.decision_accuracy for d in relevant_decisions]
        confidences = [d.confidence for d in relevant_decisions]
        pnls = [d.pnl for d in relevant_decisions if d.pnl is not None]
        
        return {
            'count': len(relevant_decisions),
            'accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'avg_confidence': np.mean(confidences),
            'win_rate': sum(1 for acc in accuracies if acc > 0.5) / len(accuracies) if accuracies else 0,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'total_pnl': sum(pnls) if pnls else 0,
            'sharpe_ratio': self._calculate_sharpe(pnls) if pnls else 0
        }
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized
        sharpe = (mean_return - risk_free_rate/252) / std_return * np.sqrt(252)
        return sharpe
    
    def get_decision_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get summary of recent decisions"""
        recent_decisions = list(self.decision_history)[-last_n:]
        
        if not recent_decisions:
            return {}
        
        signal_counts = {}
        total_pnl = 0.0
        win_count = 0
        
        for decision in recent_decisions:
            # Count signals
            signal_type = decision.signal_type
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            # P&L tracking
            if decision.pnl is not None:
                total_pnl += decision.pnl
                if decision.pnl > 0:
                    win_count += 1
        
        return {
            'decision_count': len(recent_decisions),
            'signal_breakdown': signal_counts,
            'total_pnl': total_pnl,
            'win_count': win_count,
            'win_rate': win_count / len(recent_decisions) if recent_decisions else 0,
            'avg_confidence': np.mean([d.confidence for d in recent_decisions]),
            'date_range': {
                'start': recent_decisions[0].timestamp.isoformat(),
                'end': recent_decisions[-1].timestamp.isoformat()
            }
        }
    
    def export_decision_audit_trail(self, start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> List[Dict]:
        """Export decision audit trail for compliance"""
        audit_trail = []
        
        filtered_decisions = [d for d in self.decision_history 
                            if (start_date is None or d.timestamp >= start_date) and
                               (end_date is None or d.timestamp <= end_date)]
        
        for decision in filtered_decisions:
            audit_entry = decision.to_dict()
            audit_entry['explanation'] = self.explain_decision(decision).__dict__
            
            # Add PnL attribution
            if decision.pnl is not None:
                audit_entry['pnl_attribution'] = self.attribute_pnl_to_components(decision)
            
            audit_trail.append(audit_entry)
        
        return audit_trail
    
    def get_feature_importance_trend(self, last_n: int = 50) -> Dict[str, List[float]]:
        """Get trending feature importance over time"""
        recent_decisions = list(self.decision_history)[-last_n:]
        
        feature_importance_trend = {}
        
        for decision in recent_decisions:
            for feature, importance in decision.feature_importance.items():
                if feature not in feature_importance_trend:
                    feature_importance_trend[feature] = []
                feature_importance_trend[feature].append(importance)
        
        # Return top features
        avg_importances = {feat: np.mean(vals) for feat, vals in feature_importance_trend.items()}
        top_features = sorted(avg_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        return {feat: feature_importance_trend[feat] for feat, _ in top_features}
    
    def generate_decision_report(self, symbol: str, last_n: int = 20) -> str:
        """Generate human-readable decision report"""
        decisions = [d for d in self.decision_history if d.symbol == symbol][-last_n:]
        
        if not decisions:
            return f"No decisions found for {symbol}"
        
        report = f"\n{'='*60}\nDECISION REPORT FOR {symbol}\n{'='*60}\n"
        
        # Summary stats
        summary = self.get_decision_summary(len(decisions))
        report += f"\nSummary (Last {len(decisions)} decisions):\n"
        report += f"  Total P&L: {summary.get('total_pnl', 0):.2f}\n"
        report += f"  Win Rate: {summary.get('win_rate', 0):.2%}\n"
        report += f"  Avg Confidence: {summary.get('avg_confidence', 0):.2%}\n"
        
        # Signal breakdown
        report += f"\nSignal Breakdown:\n"
        for signal_type, count in summary.get('signal_breakdown', {}).items():
            accuracy = self.calculate_signal_type_accuracy(signal_type)
            report += f"  {signal_type}: {count} signals, "
            if accuracy:
                report += f"{accuracy.get('accuracy', 0):.2%} accuracy\n"
            else:
                report += "no outcomes yet\n"
        
        # Recent decisions
        report += f"\nRecent Decisions:\n"
        for decision in decisions[-5:]:
            explanation = self.explain_decision(decision)
            report += f"\n  [{decision.timestamp.strftime('%Y-%m-%d %H:%M')}] {explanation.explanation_text}\n"
            report += f"    Confidence: {decision.confidence:.2%}\n"
            if decision.pnl is not None:
                report += f"    P&L: {decision.pnl:+.2f}\n"
        
        report += f"\n{'='*60}\n"
        return report


# Singleton instance
_explainability_engine = None

def get_explainability_engine() -> ModelExplainabilityEngine:
    """Get or create singleton explainability engine"""
    global _explainability_engine
    if _explainability_engine is None:
        _explainability_engine = ModelExplainabilityEngine()
    return _explainability_engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = get_explainability_engine()
    
    # Create sample decision
    decision = ModelDecision(
        decision_id=hashlib.md5(str(datetime.now()).encode()).hexdigest(),
        timestamp=datetime.now(),
        symbol="RELIANCE.NS",
        signal_type="BUY",
        confidence=0.85,
        prediction_value=100.5,
        top_features={'rsi_10': 0.3, 'macd_20': 0.25, 'volume_signal': 0.2},
        feature_importance={'rsi_10': 0.4, 'macd_20': 0.3, 'volume_signal': 0.3},
        model_contributions={'xgb': 0.9, 'lgb': 0.8, 'ensemble': 0.85},
        ensemble_method='weighted',
        market_regime='TRENDING',
        sentiment_score=0.65,
        technical_score=0.8,
        volume_score=0.7
    )
    
    # Record decision
    engine.record_decision(decision)
    
    # Generate explanation
    explanation = engine.explain_decision(decision)
    print("Decision Explanation:")
    print(explanation.explanation_text)
    print(f"\nReasons: {explanation.reason_breakdown}")
    print(f"Confidence Factors: {explanation.confidence_factors}")
    print(f"Risk Factors: {explanation.risk_factors}")
    
    # Record outcome
    engine.record_outcome(decision.decision_id, 0.05, 50.0)
    
    # Get summary
    summary = engine.get_decision_summary()
    print(f"\nDecision Summary: {summary}")
