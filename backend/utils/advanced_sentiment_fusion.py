#!/usr/bin/env python3
"""
Advanced Sentiment Fusion Engine
=================================

Provides multi-source sentiment aggregation with credibility weighting,
sentiment trend analysis, real-time monitoring, and decision integration
for quantitative trading systems.

Sources:
- Financial News (NewsAPI, RSS feeds)
- Social Media (Twitter, Reddit sentiment aggregates)
- Market Indicators (Fear & Greed Index, VIX)
- Analyst Sentiment
- Earnings Call Transcripts
- Insider Trading Signals
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class SentimentDirection(Enum):
    """Sentiment direction classification"""
    VERY_BULLISH = 4
    BULLISH = 3
    NEUTRAL = 2
    BEARISH = 1
    VERY_BEARISH = 0


@dataclass
class SentimentSignal:
    """Individual sentiment signal"""
    source: str  # news, social, market_indicator, analyst, insider
    direction: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    weight: float  # Credibility weight
    timestamp: datetime
    content: str = ""
    source_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentScore:
    """Aggregated sentiment score"""
    overall: float  # -1.0 to 1.0
    bullish_probability: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    strength: float  # Magnitude of sentiment
    direction: SentimentDirection
    timestamp: datetime
    source_breakdown: Dict[str, float] = field(default_factory=dict)
    signal_count: int = 0
    consensus_level: float = 0.0  # Agreement among sources


class AdvancedSentimentFusionEngine:
    """
    Advanced sentiment fusion with multi-source aggregation and credibility weighting
    """
    
    def __init__(self, lookback_days: int = 30):
        """
        Initialize sentiment fusion engine
        
        Args:
            lookback_days: Days to maintain sentiment history
        """
        self.lookback_days = lookback_days
        self.sentiment_history = deque(maxlen=1000)
        self.signal_history = deque(maxlen=5000)
        
        # Source credibility weights (can be updated based on historical accuracy)
        self.source_weights = {
            'finbert_news': 0.25,  # FinBERT trained on financial data
            'vader_social': 0.15,  # VADER sentiment
            'market_fear_greed': 0.20,  # VIX/Fear Index
            'analyst_consensus': 0.20,  # Analyst ratings
            'insider_trading': 0.12,  # Insider sentiment proxy
            'earnings_transcript': 0.08  # Earnings call sentiment
        }
        
        # Initialize source credibility tracker
        self.source_accuracy = {source: 0.8 for source in self.source_weights}  # Base accuracy
        
        # Sentiment trend analyzer
        self.sentiment_trends = {}
        self.volatility_clusters = []
        
        logger.info("Advanced Sentiment Fusion Engine initialized")
    
    def add_sentiment_signal(self, signal: SentimentSignal):
        """Add individual sentiment signal to aggregation pool"""
        # Update weight based on source credibility
        signal.weight = self.source_weights.get(signal.source, 0.1)
        signal.confidence *= self.source_accuracy.get(signal.source, 0.8)
        
        self.signal_history.append(signal)
        logger.debug(f"Added sentiment signal from {signal.source}: {signal.direction:.3f}")
    
    def aggregate_sentiment(self, lookback_minutes: int = 60) -> SentimentScore:
        """
        Aggregate sentiment signals with credibility weighting
        
        Args:
            lookback_minutes: Minutes to look back for signal aggregation
        
        Returns:
            Aggregated sentiment score
        """
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_signals = [s for s in self.signal_history if s.timestamp > cutoff_time]
        
        if not recent_signals:
            return self._create_neutral_sentiment()
        
        # Separate signals by source for breakdown
        source_signals = {}
        for signal in recent_signals:
            if signal.source not in source_signals:
                source_signals[signal.source] = []
            source_signals[signal.source].append(signal)
        
        # Calculate weighted sentiment for each source
        source_sentiments = {}
        for source, signals in source_signals.items():
            weighted_sum = sum(s.direction * s.weight * s.confidence for s in signals)
            total_weight = sum(s.weight * s.confidence for s in signals)
            
            if total_weight > 0:
                source_sentiments[source] = weighted_sum / total_weight
            else:
                source_sentiments[source] = 0.0
        
        # Calculate overall weighted sentiment
        total_weighted_sum = sum(source_sentiments.values() * self.source_weights.get(source, 0.1) 
                                for source, _ in source_sentiments.items())
        total_weight = sum(self.source_weights.get(source, 0.1) 
                          for source in source_sentiments)
        
        overall_sentiment = total_weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_sentiment = np.clip(overall_sentiment, -1.0, 1.0)
        
        # Calculate consensus level (agreement among sources)
        sentiments = list(source_sentiments.values())
        if len(sentiments) > 1:
            sentiment_std = np.std(sentiments)
            consensus = 1.0 - np.clip(sentiment_std, 0.0, 1.0)
        else:
            consensus = 0.5
        
        # Determine direction
        if overall_sentiment > 0.3:
            direction = SentimentDirection.BULLISH if overall_sentiment < 0.7 else SentimentDirection.VERY_BULLISH
        elif overall_sentiment < -0.3:
            direction = SentimentDirection.BEARISH if overall_sentiment > -0.7 else SentimentDirection.VERY_BEARISH
        else:
            direction = SentimentDirection.NEUTRAL
        
        # Calculate bullish probability
        bullish_prob = (overall_sentiment + 1.0) / 2.0
        
        # Calculate confidence
        signal_confidence = np.mean([s.confidence for s in recent_signals]) if recent_signals else 0.5
        
        score = SentimentScore(
            overall=overall_sentiment,
            bullish_probability=bullish_prob,
            confidence=signal_confidence,
            strength=abs(overall_sentiment),
            direction=direction,
            timestamp=datetime.now(),
            source_breakdown=source_sentiments,
            signal_count=len(recent_signals),
            consensus_level=consensus
        )
        
        self.sentiment_history.append(score)
        logger.info(f"Aggregated sentiment: {direction.name} ({overall_sentiment:.3f}), "
                   f"Confidence: {signal_confidence:.2%}, Consensus: {consensus:.2%}")
        
        return score
    
    def calculate_sentiment_trend(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Calculate sentiment trend and changes
        
        Args:
            lookback_hours: Hours to analyze for trend
        
        Returns:
            Dict with trend analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_sentiments = [s for s in self.sentiment_history if s.timestamp > cutoff_time]
        
        if len(recent_sentiments) < 2:
            return self._neutral_trend()
        
        sentiments = [s.overall for s in recent_sentiments]
        
        # Calculate trend
        trend_up = sentiments[-1] > sentiments[0]
        trend_strength = abs(sentiments[-1] - sentiments[0])
        
        # Calculate acceleration
        if len(sentiments) >= 4:
            recent_change = sentiments[-1] - sentiments[-2]
            previous_change = sentiments[-3] - sentiments[-4]
            acceleration = recent_change - previous_change
        else:
            acceleration = 0.0
        
        # Volatility clustering detection
        sentiment_std = np.std(sentiments)
        
        return {
            'direction': 'IMPROVING' if trend_up else 'DECLINING',
            'strength': float(trend_strength),
            'acceleration': float(acceleration),
            'volatility': float(sentiment_std),
            'current_sentiment': float(sentiments[-1]),
            'previous_sentiment': float(sentiments[0]) if sentiments else 0.0,
            'samples': len(recent_sentiments)
        }
    
    def identify_sentiment_divergences(self, price_series: pd.Series) -> List[Dict[str, Any]]:
        """
        Identify divergences between sentiment and price movement
        Bullish/Bearish divergences can signal reversals
        
        Args:
            price_series: Price data for analysis
        
        Returns:
            List of divergence signals
        """
        if len(self.sentiment_history) < 10 or len(price_series) < 10:
            return []
        
        divergences = []
        
        recent_sentiments = list(self.sentiment_history)[-10:]
        recent_prices = price_series.tail(10).values
        
        # Check sentiment trend
        sentiment_trend = recent_sentiments[-1].overall > recent_sentiments[0].overall
        
        # Check price trend
        price_trend = recent_prices[-1] > recent_prices[0]
        
        # Divergence detection
        if sentiment_trend and not price_trend:
            divergences.append({
                'type': 'BULLISH_DIVERGENCE',
                'strength': abs(recent_sentiments[-1].overall - recent_sentiments[0].overall),
                'confidence': np.mean([s.confidence for s in recent_sentiments]),
                'message': 'Sentiment improving but price declining - potential reversal signal',
                'timestamp': datetime.now()
            })
        
        elif not sentiment_trend and price_trend:
            divergences.append({
                'type': 'BEARISH_DIVERGENCE',
                'strength': abs(recent_sentiments[-1].overall - recent_sentiments[0].overall),
                'confidence': np.mean([s.confidence for s in recent_sentiments]),
                'message': 'Sentiment declining but price increasing - potential reversal signal',
                'timestamp': datetime.now()
            })
        
        return divergences
    
    def calculate_sentiment_strength(self, sentiment: SentimentScore) -> float:
        """
        Calculate combined sentiment strength considering multiple factors
        
        Args:
            sentiment: Aggregated sentiment score
        
        Returns:
            Normalized strength score (0-1)
        """
        # Base strength from sentiment magnitude
        magnitude_strength = sentiment.strength
        
        # Confidence boost
        confidence_boost = sentiment.confidence * 0.3
        
        # Consensus boost
        consensus_boost = sentiment.consensus_level * 0.2
        
        # Signal count normalization (more signals = higher strength, but with diminishing returns)
        signal_strength = min(1.0, np.log(sentiment.signal_count + 1) / np.log(100))
        signal_boost = signal_strength * 0.2
        
        total_strength = magnitude_strength + confidence_boost + consensus_boost + signal_boost
        return np.clip(total_strength, 0.0, 1.0)
    
    def generate_sentiment_alerts(self, sentiment: SentimentScore, 
                                 prev_sentiment: Optional[SentimentScore] = None) -> List[Dict[str, Any]]:
        """
        Generate actionable sentiment alerts for trading
        
        Args:
            sentiment: Current sentiment score
            prev_sentiment: Previous sentiment score
        
        Returns:
            List of alert signals
        """
        alerts = []
        
        # Strong sentiment reversal
        if prev_sentiment:
            direction_change = sentiment.direction != prev_sentiment.direction
            if direction_change:
                strength = abs(sentiment.overall - prev_sentiment.overall)
                alerts.append({
                    'type': 'SENTIMENT_REVERSAL',
                    'severity': 'HIGH' if strength > 0.5 else 'MEDIUM',
                    'message': f'Sentiment reversed: {prev_sentiment.direction.name} → {sentiment.direction.name}',
                    'change_magnitude': float(strength),
                    'confidence': float(sentiment.confidence),
                    'timestamp': datetime.now()
                })
        
        # Extreme sentiment
        if sentiment.overall > 0.7:
            alerts.append({
                'type': 'EXTREME_BULLISH',
                'severity': 'MEDIUM',
                'message': 'Very strong bullish sentiment detected',
                'sentiment_level': float(sentiment.overall),
                'consensus': float(sentiment.consensus_level),
                'timestamp': datetime.now()
            })
        
        elif sentiment.overall < -0.7:
            alerts.append({
                'type': 'EXTREME_BEARISH',
                'severity': 'MEDIUM',
                'message': 'Very strong bearish sentiment detected',
                'sentiment_level': float(sentiment.overall),
                'consensus': float(sentiment.consensus_level),
                'timestamp': datetime.now()
            })
        
        # Low consensus (conflicting signals)
        if sentiment.consensus_level < 0.3:
            alerts.append({
                'type': 'LOW_CONSENSUS',
                'severity': 'LOW',
                'message': 'Low consensus among sentiment sources - mixed signals',
                'consensus_level': float(sentiment.consensus_level),
                'sources_count': len(sentiment.source_breakdown),
                'timestamp': datetime.now()
            })
        
        return alerts
    
    def update_source_credibility(self, source: str, actual_direction: float, predicted_direction: float):
        """
        Update source credibility based on prediction accuracy
        
        Args:
            source: Source name
            actual_direction: Actual sentiment direction that materialized
            predicted_direction: Predicted sentiment direction
        """
        if source not in self.source_accuracy:
            self.source_accuracy[source] = 0.5
        
        # Calculate accuracy (1.0 if correct direction, 0.0 if opposite)
        if (actual_direction > 0 and predicted_direction > 0) or (actual_direction < 0 and predicted_direction < 0):
            accuracy = 1.0
        elif abs(actual_direction) < 0.1 or abs(predicted_direction) < 0.1:
            accuracy = 0.7  # Partial credit for neutral near-correct
        else:
            accuracy = 0.0
        
        # Update exponential moving average
        alpha = 0.3
        self.source_accuracy[source] = (alpha * accuracy) + ((1 - alpha) * self.source_accuracy[source])
        
        logger.info(f"Updated {source} credibility: {self.source_accuracy[source]:.3f}")
    
    def _create_neutral_sentiment(self) -> SentimentScore:
        """Create neutral sentiment score"""
        return SentimentScore(
            overall=0.0,
            bullish_probability=0.5,
            confidence=0.3,
            strength=0.0,
            direction=SentimentDirection.NEUTRAL,
            timestamp=datetime.now(),
            consensus_level=0.0,
            signal_count=0
        )
    
    def _neutral_trend(self) -> Dict[str, Any]:
        """Create neutral trend analysis"""
        return {
            'direction': 'STABLE',
            'strength': 0.0,
            'acceleration': 0.0,
            'volatility': 0.0,
            'current_sentiment': 0.0,
            'previous_sentiment': 0.0,
            'samples': 0
        }
    
    def export_sentiment_metrics(self) -> Dict[str, Any]:
        """Export sentiment metrics for analysis"""
        recent_sentiments = list(self.sentiment_history)[-100:]
        
        if not recent_sentiments:
            return {}
        
        sentiments = [s.overall for s in recent_sentiments]
        
        return {
            'count': len(recent_sentiments),
            'mean_sentiment': float(np.mean(sentiments)),
            'std_sentiment': float(np.std(sentiments)),
            'min_sentiment': float(np.min(sentiments)),
            'max_sentiment': float(np.max(sentiments)),
            'bullish_percentage': float(sum(1 for s in recent_sentiments if s.overall > 0) / len(recent_sentiments) * 100),
            'bearish_percentage': float(sum(1 for s in recent_sentiments if s.overall < 0) / len(recent_sentiments) * 100),
            'neutral_percentage': float(sum(1 for s in recent_sentiments if abs(s.overall) <= 0.1) / len(recent_sentiments) * 100),
            'mean_confidence': float(np.mean([s.confidence for s in recent_sentiments])),
            'mean_consensus': float(np.mean([s.consensus_level for s in recent_sentiments])),
            'average_signals_per_score': float(np.mean([s.signal_count for s in recent_sentiments]))
        }


# Singleton instance
_fusion_engine = None

def get_sentiment_fusion_engine() -> AdvancedSentimentFusionEngine:
    """Get or create singleton fusion engine instance"""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = AdvancedSentimentFusionEngine()
    return _fusion_engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = get_sentiment_fusion_engine()
    
    # Simulate sentiment signals
    signals = [
        SentimentSignal('finbert_news', 0.8, 0.9, 0.25, datetime.now(), "Positive news"),
        SentimentSignal('vader_social', 0.6, 0.7, 0.15, datetime.now(), "Positive social"),
        SentimentSignal('market_fear_greed', 0.5, 0.8, 0.20, datetime.now(), "Moderate fear"),
    ]
    
    for signal in signals:
        engine.add_sentiment_signal(signal)
    
    # Aggregate sentiment
    agg_sentiment = engine.aggregate_sentiment()
    print(f"Aggregated Sentiment: {agg_sentiment.direction.name} ({agg_sentiment.overall:.3f})")
    print(f"Confidence: {agg_sentiment.confidence:.2%}, Consensus: {agg_sentiment.consensus_level:.2%}")
    
    # Trend analysis
    trend = engine.calculate_sentiment_trend()
    print(f"Sentiment Trend: {trend['direction']}, Strength: {trend['strength']:.3f}")
    
    # Export metrics
    metrics = engine.export_sentiment_metrics()
    print(f"Sentiment Metrics: {metrics}")
