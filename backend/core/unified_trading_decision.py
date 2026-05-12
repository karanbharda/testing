"""
Unified Trading Decision System
Automatically selects optimal trade direction (Long vs Short) based on market conditions
Industry-level implementation with dual-angle analysis
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction enumeration"""
    LONG = "long"      # Buy first → Sell later (traditional)
    SHORT = "short"    # Sell first → Buy later (reverse)
    NEUTRAL = "neutral"  # No clear bias, wait for better setup


@dataclass
class UnifiedDecision:
    """Unified trading decision output"""
    direction: TradeDirection
    action: str  # "buy", "sell", or "hold"
    quantity: int
    confidence: float
    long_score: float  # Confidence score for long position
    short_score: float  # Confidence score for short position
    reasoning: str
    metadata: Dict


class UnifiedTradingDecisionSystem:
    """
    Industry-Level Trading Direction Selector
    
    Features:
    - Dual-angle analysis (both long and short)
    - Automatic direction selection based on confidence
    - Bias detection to avoid counter-trend trades
    - Seamless integration with existing buy/sell logic
    
    Decision Flow:
    1. Evaluate long bias score
    2. Evaluate short bias score
    3. Compare scores and select direction
    4. Execute appropriate logic (buy or sell)
    5. Monitor and exit when targets hit
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Configuration
        self.enable_auto_detection = config.get("auto_detect_trade_direction", True)
        self.long_threshold = config.get("long_bias_threshold", 0.65)
        self.short_threshold = config.get("short_bias_threshold", 0.65)
        self.neutral_zone_min = config.get("neutral_zone_min", 0.35)
        self.neutral_zone_max = config.get("neutral_zone_max", 0.65)
        
        logger.info("Unified Trading Decision System initialized")
        logger.info(f"   Auto-Detection: {self.enable_auto_detection}")
        logger.info(f"   Long Threshold: {self.long_threshold}")
        logger.info(f"   Short Threshold: {self.short_threshold}")
    
    def evaluate_trade_direction(
        self,
        ticker: str,
        current_price: float,
        analysis_data: Dict,
        market_context: Dict,
        portfolio_holdings: Dict = None
    ) -> UnifiedDecision:
        """
        Main entry point - evaluates both long and short bias,
        then recommends optimal trade direction
        
        Returns:
            UnifiedDecision with recommended direction and action
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"UNIFIED TRADING DECISION: {ticker}")
        logger.info(f"{'='*80}")
        logger.info(f"Current Price: ₹{current_price:.2f}")
        
        # Step 1: Evaluate Long Bias
        logger.info(f"\n📈 Evaluating LONG Bias...")
        long_confidence = self._evaluate_long_bias(analysis_data, market_context)
        
        # Step 2: Evaluate Short Bias
        logger.info(f"\n📉 Evaluating SHORT Bias...")
        short_confidence = self._evaluate_short_bias(analysis_data, market_context)
        
        # Step 3: Compare and Select Direction
        logger.info(f"\n{'='*80}")
        logger.info(f"DIRECTION SELECTION")
        logger.info(f"{'='*80}")
        logger.info(f"Long Score:  {long_confidence:.3f}")
        logger.info(f"Short Score: {short_confidence:.3f}")
        
        direction, selected_confidence, reasoning = self._select_direction(
            long_confidence, short_confidence
        )
        
        logger.info(f"\n🎯 SELECTED DIRECTION: {direction.value.upper()}")
        logger.info(f"   Confidence: {selected_confidence:.3f}")
        logger.info(f"   Reasoning: {reasoning}")
        
        # Step 4: Determine Action
        if portfolio_holdings and ticker in portfolio_holdings:
            holding = portfolio_holdings[ticker]
            qty = holding.get("qty", 0)
            
            if qty > 0:
                # We have a long position - should we sell?
                action = "sell" if direction == TradeDirection.SHORT else "hold"
            elif qty < 0:
                # We have a short position - should we cover?
                action = "buy" if direction == TradeDirection.LONG else "hold"
            else:
                # No position - take new position based on direction
                if direction == TradeDirection.LONG:
                    action = "buy"
                elif direction == TradeDirection.SHORT:
                    action = "sell"
                else:
                    action = "hold"
        else:
            # No existing position
            if direction == TradeDirection.LONG:
                action = "buy"
            elif direction == TradeDirection.SHORT:
                action = "sell"
            else:
                action = "hold"
        
        logger.info(f"\n💡 RECOMMENDED ACTION: {action.upper()}")
        
        return UnifiedDecision(
            direction=direction,
            action=action,
            quantity=0,  # Will be filled by execution logic
            confidence=selected_confidence,
            long_score=long_confidence,
            short_score=short_confidence,
            reasoning=reasoning,
            metadata={
                "ticker": ticker,
                "current_price": current_price
            }
        )
    
    def _evaluate_long_bias(self, analysis_data: Dict, market_context: Dict) -> float:
        """
        Evaluate bullish/long bias score (0.0 to 1.0)
        
        This is a simplified version - in production, this would call
        the actual ProfessionalBuyLogic evaluation
        """
        from core.professional_sell_logic import MarketTrend
        
        bias_score = 0.0
        max_score = 10.0
        
        technical = analysis_data.get("technical_indicators", {})
        sentiment = analysis_data.get("sentiment_analysis", {})
        ml = analysis_data.get("ml_analysis", {})
        
        # Factor 1: Market Trend (max 2.5 points)
        trend = market_context.get("trend", MarketTrend.SIDEWAYS)
        trend_strength = market_context.get("trend_strength", 0.5)
        
        if trend == MarketTrend.STRONG_UPTREND:
            trend_points = 2.5 * trend_strength
        elif trend == MarketTrend.UPTREND:
            trend_points = 2.0 * trend_strength
        elif trend == MarketTrend.SIDEWAYS:
            trend_points = 1.0
        elif trend == MarketTrend.DOWNTREND:
            trend_points = 0.3 * (1 - trend_strength)
        else:
            trend_points = 0.1 * (1 - trend_strength)
        
        bias_score += trend_points
        
        # Factor 2: Technical Strength (max 2.5 points)
        rsi = technical.get("rsi", 50)
        macd = technical.get("macd", 0)
        macd_signal = technical.get("macd_signal", 0)
        
        technical_points = 0.0
        if rsi > 50 and rsi < 70:
            technical_points += 1.0
        if macd > macd_signal:
            technical_points += 1.0
        if technical.get("current_price", 0) > technical.get("sma_20", 0):
            technical_points += 0.5
        
        bias_score += min(technical_points, 2.5)
        
        # Factor 3: Positive Sentiment (max 2.0 points)
        sentiment_score = sentiment.get("overall_sentiment", 0)
        sentiment_points = 0.0
        if sentiment_score > 0.3:
            sentiment_points = 1.5
        elif sentiment_score > 0.1:
            sentiment_points = 0.7
        
        bias_score += min(sentiment_points, 2.0)
        
        # Factor 4: ML Bullish Prediction (max 2.0 points)
        ml_prediction = ml.get("prediction_direction", 0)
        ml_confidence = ml.get("confidence", 0.5)
        rl_rec = ml.get("rl_recommendation", "HOLD")
        
        ml_points = 0.0
        if ml_prediction > 0.03:
            ml_points += 1.5 * ml_confidence
        elif ml_prediction > 0.01:
            ml_points += 0.7 * ml_confidence
        
        if rl_rec == "BUY":
            ml_points += 0.5 * ml.get("rl_confidence", 0.5)
        
        bias_score += min(ml_points, 2.0)
        
        # Factor 5: Market Conditions (max 1.0 point)
        market_stress = market_context.get("market_stress", 0.5)
        if market_stress < 0.3:
            bias_score += 0.7
        elif market_stress < 0.5:
            bias_score += 0.4
        
        final_confidence = min(bias_score / max_score, 1.0)
        
        logger.info(f"   Long Bias Score: {final_confidence:.3f}")
        return final_confidence
    
    def _evaluate_short_bias(self, analysis_data: Dict, market_context: Dict) -> float:
        """
        Evaluate bearish/short bias score (0.0 to 1.0)
        
        Simplified version - delegates to ProfessionalShortSellLogic in production
        """
        from core.professional_sell_logic import MarketTrend
        
        bias_score = 0.0
        max_score = 10.0
        
        technical = analysis_data.get("technical_indicators", {})
        sentiment = analysis_data.get("sentiment", {})
        ml = analysis_data.get("ml_analysis", {})
        
        # Factor 1: Market Trend (max 2.5 points)
        trend = market_context.get("trend", MarketTrend.SIDEWAYS)
        trend_strength = market_context.get("trend_strength", 0.5)
        
        if trend == MarketTrend.STRONG_DOWNTREND:
            trend_points = 2.5 * trend_strength
        elif trend == MarketTrend.DOWNTREND:
            trend_points = 2.0 * trend_strength
        elif trend == MarketTrend.SIDEWAYS:
            trend_points = 1.0
        elif trend == MarketTrend.UPTREND:
            trend_points = 0.3 * (1 - trend_strength)
        else:
            trend_points = 0.1 * (1 - trend_strength)
        
        bias_score += trend_points
        
        # Factor 2: Technical Weakness (max 2.5 points)
        rsi = technical.get("rsi", 50)
        macd = technical.get("macd", 0)
        macd_signal = technical.get("macd_signal", 0)
        
        technical_points = 0.0
        if rsi < 30:
            technical_points += 1.0
        elif rsi < 40:
            technical_points += 0.5
        
        if macd < macd_signal and macd < 0:
            technical_points += 1.0
        
        if technical.get("current_price", 0) < technical.get("sma_20", 0):
            technical_points += 0.5
        
        bias_score += min(technical_points, 2.5)
        
        # Factor 3: Negative Sentiment (max 2.0 points)
        sentiment_score = sentiment.get("overall_sentiment", 0)
        sentiment_points = 0.0
        if sentiment_score < -0.3:
            sentiment_points = 1.5
        elif sentiment_score < -0.1:
            sentiment_points = 0.7
        
        bias_score += min(sentiment_points, 2.0)
        
        # Factor 4: ML Bearish Prediction (max 2.0 points)
        ml_prediction = ml.get("prediction_direction", 0)
        ml_confidence = ml.get("confidence", 0.5)
        rl_rec = ml.get("rl_recommendation", "HOLD")
        
        ml_points = 0.0
        if ml_prediction < -0.03:
            ml_points += 1.5 * ml_confidence
        elif ml_prediction < -0.01:
            ml_points += 0.7 * ml_confidence
        
        if rl_rec == "SELL":
            ml_points += 0.5 * ml.get("rl_confidence", 0.5)
        
        bias_score += min(ml_points, 2.0)
        
        # Factor 5: Market Stress (max 1.0 point)
        market_stress = market_context.get("market_stress", 0.5)
        if market_stress > 0.7:
            bias_score += 0.7
        elif market_stress > 0.5:
            bias_score += 0.4
        
        final_confidence = min(bias_score / max_score, 1.0)
        
        logger.info(f"   Short Bias Score: {final_confidence:.3f}")
        return final_confidence
    
    def _select_direction(
        self,
        long_confidence: float,
        short_confidence: float
    ) -> Tuple[TradeDirection, float, str]:
        """
        Select trade direction based on confidence scores
        
        Returns:
            (TradeDirection, confidence, reasoning)
        """
        # Check for clear long bias
        if long_confidence >= self.long_threshold and long_confidence > short_confidence:
            return (
                TradeDirection.LONG,
                long_confidence,
                f"Strong long bias detected ({long_confidence:.3f} >= {self.long_threshold}). "
                f"Market conditions favor buying."
            )
        
        # Check for clear short bias
        if short_confidence >= self.short_threshold and short_confidence > long_confidence:
            return (
                TradeDirection.SHORT,
                short_confidence,
                f"Strong short bias detected ({short_confidence:.3f} >= {self.short_threshold}). "
                f"Market conditions favor selling."
            )
        
        # Neutral zone - no clear direction
        if (self.neutral_zone_min <= long_confidence <= self.neutral_zone_max and
            self.neutral_zone_min <= short_confidence <= self.neutral_zone_max):
            return (
                TradeDirection.NEUTRAL,
                max(long_confidence, short_confidence),
                f"Neutral market conditions. Long: {long_confidence:.3f}, Short: {short_confidence:.3f}. "
                f"Wait for clearer signals."
            )
        
        # Weak signals - prefer long if slightly bullish
        if long_confidence > short_confidence:
            return (
                TradeDirection.LONG,
                long_confidence,
                f"Weak long bias ({long_confidence:.3f} > {short_confidence:.3f}). "
                f"Proceed with caution."
            )
        
        # Weak signals - prefer short if slightly bearish
        return (
            TradeDirection.SHORT,
            short_confidence,
            f"Weak short bias ({short_confidence:.3f} > {long_confidence:.3f}). "
            f"Proceed with caution."
        )


# Convenience function for easy integration
def get_unified_trading_decision(
    ticker: str,
    current_price: float,
    analysis_data: Dict,
    market_context: Dict,
    portfolio_holdings: Dict = None,
    config: Dict = None
) -> UnifiedDecision:
    """
    Get unified trading decision with auto-detection
    
    Args:
        ticker: Stock symbol
        current_price: Current market price
        analysis_data: Technical, sentiment, ML analysis
        market_context: Market trend and conditions
        portfolio_holdings: Current positions
        config: Configuration dictionary
    
    Returns:
        UnifiedDecision with recommended action
    """
    if config is None:
        config = {
            "auto_detect_trade_direction": True,
            "long_bias_threshold": 0.65,
            "short_bias_threshold": 0.65
        }
    
    system = UnifiedTradingDecisionSystem(config)
    return system.evaluate_trade_direction(
        ticker=ticker,
        current_price=current_price,
        analysis_data=analysis_data,
        market_context=market_context,
        portfolio_holdings=portfolio_holdings
    )
