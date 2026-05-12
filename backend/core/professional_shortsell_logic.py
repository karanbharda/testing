"""
Professional Short-Sell Logic Framework for Intraday Trading
Implements institutional-grade short-selling logic with proper risk management,
reverse entry confirmation, and dynamic intraday position management.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ShortSellReason(Enum):
    """Professional short-sell reasons for audit trail"""
    TECHNICAL_BREAKDOWN = "technical_breakdown"
    MOMENTUM_REVERSAL = "momentum_reversal"
    SENTIMENT_DETERIORATION = "sentiment_deterioration"
    ML_BEARISH_PREDICTION = "ml_bearish_prediction"
    MARKET_DOWNTREND = "market_downtrend"
    OVERVALUED_STOCK = "overvalued_stock"
    VOLUME_SURGE = "volume_surge"


@dataclass
class ShortSellSignal:
    """Individual short-sell signal with strength and reasoning"""
    name: str
    strength: float  # 0.0 to 1.0
    weight: float    # Signal importance weight
    triggered: bool
    reasoning: str
    confidence: float
    category: str = ""  # Category: Technical, Momentum, Sentiment, ML, Market, Value


@dataclass
class ShortPosition:
    """Short position-specific metrics"""
    entry_price: float  # Price at which short was initiated
    current_price: float  # Current market price
    quantity: int  # Number of shares shorted
    unrealized_pnl: float  # (Entry - Current) * Qty
    unrealized_pnl_pct: float  # Percentage P&L
    lowest_price_since_entry: float  # Lowest price after shorting
    highest_price_since_entry: float  # Highest price after shorting
    volatility: float
    time_remaining_minutes: int  # Time left in trading day
    db_stop_loss: Optional[float] = None  # Stop-loss from database (buyback level)
    db_target_price: Optional[float] = None  # Target from database (profit booking)


@dataclass
class ShortDecision:
    """Professional short-sell decision output"""
    should_short: bool
    short_quantity: int
    short_percentage: float  # 0.0 to 1.0
    reason: ShortSellReason
    confidence: float
    urgency: float  # 0.0 to 1.0
    signals_triggered: List[ShortSellSignal]
    stop_loss_price: float  # Buyback stop-loss (above entry)
    take_profit_price: float  # Profit booking target (below entry)
    reasoning: str


class ProfessionalShortSellLogic:
    """
    Professional-grade short-sell logic for intraday trading
    Features:
    - Reverse logic of buy (sell high first, buy back low)
    - Minimum 2-3 signal confirmation
    - Dynamic stop-losses (above entry for shorts)
    - Intraday time-aware position management
    - Market context awareness (bearish bias)
    - Position-based sizing
    - AUTO-DETECTION: Automatically chooses long vs short based on signals
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Professional thresholds - SYMMETRIC with buy logic but reversed
        self.min_signals_required = config.get("min_short_signals", 2)
        self.max_signals_required = config.get("max_short_signals", 5)
        self.min_confidence_threshold = config.get("min_short_confidence", 0.55)
        self.min_weighted_score = config.get("min_weighted_short_score", 0.12)
        
        # Stop-loss configuration (REVERSED for shorts)
        self.base_stop_loss_pct = config.get("short_stop_loss_pct", 0.05)  # 5% above entry
        self.trailing_stop_pct = config.get("short_trailing_stop_pct", 0.0325)  # 3.25% trailing
        self.profit_protection_threshold = config.get("short_profit_protection", 0.06)  # 6% profit lock
        
        # Emergency loss threshold (8-10% loss triggers immediate buyback)
        self.emergency_loss_threshold = config.get("short_emergency_loss", 0.10)
        
        # Intraday time management
        self.enable_time_decay = config.get("enable_intraday_time_decay", True)
        self.last_hour_force_exit = config.get("force_exit_last_hour", False)
        self.minutes_before_close = config.get("minutes_before_mandatory_exit", 15)
        
        # Position sizing - Granular thresholds
        self.conservative_entry_threshold = config.get("conservative_short_threshold", 0.15)
        self.partial_entry_threshold = config.get("partial_short_threshold", 0.30)
        self.aggressive_entry_threshold = config.get("aggressive_short_threshold", 0.50)
        self.full_entry_threshold = config.get("full_short_threshold", 0.70)
        self.emergency_exit_threshold = config.get("emergency_short_exit", 0.90)
        
        # Market context filters
        self.downtrend_short_multiplier = config.get("downtrend_short_multiplier", 1.15)  # More aggressive in downtrends
        self.uptrend_short_multiplier = config.get("uptrend_short_multiplier", 0.85)  # Less aggressive in uptrends
        
        # AUTO-DETECTION: Enable automatic long/short detection
        self.auto_detect_direction = config.get("auto_detect_trade_direction", True)
        self.short_bias_threshold = config.get("short_bias_threshold", 0.65)  # Confidence threshold for short bias
        
        # Define category weights for short signals
        self.category_weights = {
            "Technical": 0.25,      # 25% weight to technical breakdown
            "Momentum": 0.20,       # 20% weight to momentum reversal
            "Sentiment": 0.20,      # 20% weight to negative sentiment
            "ML": 0.20,            # 20% weight to ML bearish prediction
            "Market": 0.10,        # 10% weight to market structure
            "Value": 0.05          # 5% weight to overvaluation
        }
        
        logger.info("Professional Short-Sell Logic initialized with AUTO-DETECTION")
    
    def _load_dynamic_config(self):
        """Load dynamic configuration from live_config.json"""
        try:
            import json
            import os
            
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'live_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    live_config = json.load(f)
                
                # Update dynamic values
                self.base_stop_loss_pct = live_config.get("short_stop_loss_pct", self.base_stop_loss_pct)
                self.trailing_stop_pct = live_config.get("short_trailing_stop_pct", self.trailing_stop_pct)
                self.profit_protection_threshold = live_config.get("short_profit_protection_threshold", self.profit_protection_threshold)
                self.emergency_loss_threshold = live_config.get("short_emergency_loss_threshold", self.emergency_loss_threshold)
                
                logger.info(f"📊 Loaded dynamic config for short-sell - Stop Loss: {self.base_stop_loss_pct:.1%}")
            else:
                logger.warning("live_config.json not found for short-sell logic, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load dynamic config for short-sell: {e}")
    
    def refresh_dynamic_config(self):
        """Refresh dynamic configuration from live_config.json"""
        self._load_dynamic_config()
    
    def evaluate_short_decision(
        self,
        ticker: str,
        position_metrics: ShortPosition,
        market_context: 'MarketContext',
        technical_analysis: Dict,
        sentiment_analysis: Dict,
        ml_analysis: Dict
    ) -> ShortDecision:
        """
        Main entry point for professional short-sell evaluation
        REVERSE LOGIC: Sell high first, then buy back low
        
        AUTO-DETECTION: First checks if short-bias is strong enough,
        otherwise recommends holding (let long logic handle bullish cases)
        """
        logger.info(f"=== PROFESSIONAL SHORT-SELL EVALUATION: {ticker} ===")
        logger.info(f"Current Price: {position_metrics.current_price:.2f}")
        logger.info(f"Entry Price: {position_metrics.entry_price:.2f}")
        logger.info(f"Unrealized P&L: {position_metrics.unrealized_pnl:.2f} ({position_metrics.unrealized_pnl_pct:.2%})")
        logger.info(f"Time Remaining: {position_metrics.time_remaining_minutes} minutes")
        logger.info(f"Market Context: {market_context.trend.value} (strength: {market_context.trend_strength:.2f})")
        
        # AUTO-DETECTION: Check if we should even consider shorting
        if self.auto_detect_direction:
            short_bias_confidence = self._evaluate_short_bias(
                market_context, technical_analysis, sentiment_analysis, ml_analysis
            )
            
            logger.info(f"\n🔍 AUTO-DETECTION: Short Bias Confidence = {short_bias_confidence:.3f}")
            logger.info(f"   Threshold: {self.short_bias_threshold:.3f}")
            
            if short_bias_confidence < self.short_bias_threshold:
                logger.info(f"❌ Short bias too weak ({short_bias_confidence:.3f} < {self.short_bias_threshold:.3f})")
                logger.info(f"   Recommendation: Use LONG logic instead (buy → sell)")
                logger.info(f"   Reason: Market conditions favor bullish trades")
                
                # Return hold decision - let long logic handle this
                return ShortDecision(
                    should_short=False,
                    short_quantity=0,
                    short_percentage=0.0,
                    reason=ShortSellReason.TECHNICAL_BREAKDOWN,
                    confidence=short_bias_confidence,
                    urgency=0.0,
                    signals_triggered=[],
                    stop_loss_price=0.0,
                    take_profit_price=0.0,
                    reasoning=f"Short bias insufficient ({short_bias_confidence:.3f}). Market favors LONG positions."
                )
            else:
                logger.info(f"✅ Short bias confirmed ({short_bias_confidence:.3f} >= {self.short_bias_threshold:.3f})")
                logger.info(f"   Proceeding with short-sell evaluation")
        
        # STEP 1: Check stored stop-loss/target conditions (REVERSED for shorts)
        if position_metrics.db_stop_loss is not None and position_metrics.current_price >= position_metrics.db_stop_loss:
            # Stop-loss triggered (price went UP, causing loss on short)
            logger.info(f"⚠️ Stored stop-loss triggered: {position_metrics.current_price:.2f} >= {position_metrics.db_stop_loss:.2f}")
            return ShortDecision(
                should_short=True,  # Should buyback (cover short)
                short_quantity=position_metrics.quantity,
                short_percentage=1.0,
                reason=ShortSellReason.TECHNICAL_BREAKDOWN,
                confidence=1.0,
                urgency=1.0,
                signals_triggered=[],
                stop_loss_price=position_metrics.db_stop_loss,
                take_profit_price=position_metrics.db_target_price if position_metrics.db_target_price else 0.0,
                reasoning=f"Stored stop-loss triggered: {position_metrics.current_price:.2f} >= {position_metrics.db_stop_loss:.2f}"
            )
        
        if position_metrics.db_target_price is not None and position_metrics.current_price <= position_metrics.db_target_price:
            # Take-profit triggered (price went DOWN, profit on short)
            logger.info(f"✅ Stored take-profit triggered: {position_metrics.current_price:.2f} <= {position_metrics.db_target_price:.2f}")
            return ShortDecision(
                should_short=True,  # Should buyback (cover short)
                short_quantity=position_metrics.quantity,
                short_percentage=1.0,
                reason=ShortSellReason.TECHNICAL_BREAKDOWN,
                confidence=0.9,
                urgency=0.8,
                signals_triggered=[],
                stop_loss_price=position_metrics.db_stop_loss if position_metrics.db_stop_loss else 0.0,
                take_profit_price=position_metrics.db_target_price,
                reasoning=f"Stored take-profit triggered: {position_metrics.current_price:.2f} <= {position_metrics.db_target_price:.2f}"
            )
        
        # STEP 2: Generate signals from each category (BEARISH signals)
        technical_signals = self._generate_technical_signals(technical_analysis, position_metrics)
        momentum_signals = self._generate_momentum_signals(technical_analysis, position_metrics)
        sentiment_signals = self._generate_sentiment_signals(sentiment_analysis)
        ml_signals = self._generate_ml_signals(ml_analysis)
        market_signals = self._generate_market_signals(market_context)
        value_signals = self._generate_value_signals(technical_analysis, position_metrics)
        
        # Combine all signals
        all_signals = technical_signals + momentum_signals + sentiment_signals + ml_signals + market_signals + value_signals
        
        # Log signal generation
        logger.info(f"Generated {len(all_signals)} short-sell signals from 6 categories")
        triggered_signals = [s for s in all_signals if s.triggered]
        logger.info(f"{len(triggered_signals)} triggered bearish signals:")
        
        # Group by category
        category_signals = {}
        for signal in all_signals:
            category = signal.category or "Uncategorized"
            if category not in category_signals:
                category_signals[category] = []
            category_signals[category].append(signal)
        
        for category, signals_list in category_signals.items():
            triggered_in_category = [s for s in signals_list if s.triggered]
            logger.info(f"  {category}: {len(triggered_in_category)}/{len(signals_list)} signals triggered")
            for signal in triggered_in_category:
                logger.info(f"    - {signal.name}: Strength {signal.strength:.3f}, Confidence {signal.confidence:.3f}")
        
        # STEP 3: Calculate weighted signal score
        total_weight = sum(s.weight for s in triggered_signals)
        weighted_score = sum(s.strength * s.weight for s in triggered_signals)
        
        # Calculate confidence
        signal_confidences = [s.confidence for s in triggered_signals]
        avg_confidence = np.mean(signal_confidences) if signal_confidences else 0.0
        
        # Professional decision criteria
        signals_count = len(triggered_signals)
        meets_signal_threshold = self.min_signals_required <= signals_count <= self.max_signals_required
        meets_confidence_threshold = avg_confidence >= self.min_confidence_threshold
        meets_weighted_threshold = weighted_score >= self.min_weighted_score
        
        logger.info(f"Signal Analysis Summary:")
        logger.info(f"  Signal Count: {signals_count} (Required: {self.min_signals_required}-{self.max_signals_required}) - {'✅ PASS' if meets_signal_threshold else '❌ FAIL'}")
        logger.info(f"  Average Confidence: {avg_confidence:.3f} (Threshold: {self.min_confidence_threshold}) - {'✅ PASS' if meets_confidence_threshold else '❌ FAIL'}")
        logger.info(f"  Weighted Score: {weighted_score:.3f} (Threshold: {self.min_weighted_score}) - {'✅ PASS' if meets_weighted_threshold else '❌ FAIL'}")
        
        # Professional short-sell logic: All three conditions must be met
        should_short = meets_signal_threshold and meets_confidence_threshold and meets_weighted_threshold
        
        if should_short:
            logger.info("✅ ALL THRESHOLD CHECKS PASSED - Proceeding with short-sell decision")
            
            # Calculate stop-loss and take-profit levels (REVERSED for shorts)
            stop_levels = self._calculate_short_stops(position_metrics, market_context)
            
            base_decision = ShortDecision(
                should_short=True,
                short_quantity=position_metrics.quantity,
                short_percentage=0.0,  # Will be calculated
                reason=ShortSellReason.TECHNICAL_BREAKDOWN,
                confidence=avg_confidence,
                urgency=0.5,
                signals_triggered=triggered_signals,
                stop_loss_price=stop_levels["active_stop"],  # Above entry
                take_profit_price=stop_levels["take_profit"],  # Below entry
                reasoning=f"Professional short-sell confirmed: {signals_count} signals triggered, "
                         f"confidence {avg_confidence:.3f}, weighted score {weighted_score:.3f}"
            )
            
            # Apply market context filters
            final_decision = self._apply_market_context_filters(base_decision, market_context)
            
            # Determine position sizing
            final_decision = self._determine_position_sizing(final_decision, all_signals, position_metrics)
            
            logger.info(f"SHORT-SELL DECISION: {final_decision.should_short} | "
                       f"Qty: {final_decision.short_quantity} | "
                       f"Reason: {final_decision.reason.value} | "
                       f"Confidence: {final_decision.confidence:.3f}")
            
            return final_decision
        else:
            logger.info("❌ THRESHOLD CHECKS FAILED - Generating hold decision")
            rejection_reasons = []
            if not meets_signal_threshold:
                rejection_reasons.append(f"Triggered signals {signals_count} not in range [{self.min_signals_required}, {self.max_signals_required}]")
            if not meets_confidence_threshold:
                rejection_reasons.append(f"Confidence {avg_confidence:.3f} below threshold {self.min_confidence_threshold}")
            if not meets_weighted_threshold:
                rejection_reasons.append(f"Weighted score {weighted_score:.3f} below threshold {self.min_weighted_score}")
            
            detailed_reasoning = " | ".join(rejection_reasons)
            logger.info(f"Rejection Reasons: {detailed_reasoning}")
            
            return ShortDecision(
                should_short=False,
                short_quantity=0,
                short_percentage=0.0,
                reason=ShortSellReason.TECHNICAL_BREAKDOWN,
                confidence=avg_confidence,
                urgency=0.0,
                signals_triggered=all_signals,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                reasoning=detailed_reasoning
            )
    
    def _generate_technical_signals(self, technical: Dict, position: ShortPosition) -> List[ShortSellSignal]:
        """Generate technical breakdown sell signals (REVERSE of buy signals)"""
        signals = []
        
        # RSI Oversold (Strong bearish signal)
        rsi = technical.get("rsi", 50)
        if rsi < 30:
            strength = min((30 - rsi) / 20, 1.0)
            signals.append(ShortSellSignal(
                name="rsi_oversold",
                strength=strength,
                weight=0.08,
                triggered=True,
                reasoning=f"RSI oversold at {rsi:.1f}",
                confidence=0.8,
                category="Technical"
            ))
        
        # MACD Bearish Crossover
        macd = technical.get("macd", 0)
        macd_signal = technical.get("macd_signal", 0)
        if macd < macd_signal:
            strength = min(abs(macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 1, 1.0)
            signals.append(ShortSellSignal(
                name="macd_bearish",
                strength=strength,
                weight=0.07,
                triggered=True,
                reasoning="MACD bearish crossover",
                confidence=0.7,
                category="Technical"
            ))
        
        # Moving Average Breakdown
        sma_20 = technical.get("sma_20", position.current_price)
        sma_50 = technical.get("sma_50", position.current_price)
        if position.current_price < sma_20 and sma_20 < sma_50:
            strength = (sma_20 - position.current_price) / sma_20
            signals.append(ShortSellSignal(
                name="ma_breakdown",
                strength=min(strength, 1.0),
                weight=0.06,
                triggered=True,
                reasoning="Price below key moving averages",
                confidence=0.75,
                category="Technical"
            ))
        
        # Support Level Breakdown
        support = technical.get("support_level", 0)
        if support > 0 and position.current_price < support * 0.98:
            strength = (support - position.current_price) / support
            signals.append(ShortSellSignal(
                name="support_breakdown",
                strength=min(strength * 2, 1.0),
                weight=0.09,
                triggered=True,
                reasoning=f"Support breakdown at {support:.2f}",
                confidence=0.85,
                category="Technical"
            ))
        
        # ADX Strong Downtrend
        adx = technical.get("adx", 20)
        plus_di = technical.get("plus_di", 20)
        minus_di = technical.get("minus_di", 20)
        if adx > 25 and minus_di > plus_di:
            strength = min((adx - 25) / 25, 1.0)
            signals.append(ShortSellSignal(
                name="adx_downtrend",
                strength=strength,
                weight=0.07,
                triggered=True,
                reasoning=f"Strong downtrend with ADX {adx:.1f}, -DI {minus_di:.1f} > +DI {plus_di:.1f}",
                confidence=0.80,
                category="Technical"
            ))
        
        return signals
    
    def _generate_momentum_signals(self, technical: Dict, position: ShortPosition) -> List[ShortSellSignal]:
        """Generate momentum reversal signals"""
        signals = []
        
        # Negative ROC (Rate of Change)
        roc_10 = technical.get("roc_10", 0)
        roc_20 = technical.get("roc_20", 0)
        if roc_10 < -1.0 and roc_20 < -0.5:
            strength = min(abs(roc_10) / 5, 1.0)
            signals.append(ShortSellSignal(
                name="negative_momentum",
                strength=strength,
                weight=0.06,
                triggered=True,
                reasoning=f"Negative momentum: ROC(10) {roc_10:.2f}%, ROC(20) {roc_20:.2f}%",
                confidence=0.70,
                category="Momentum"
            ))
        
        # TRIX Bearish
        trix = technical.get("trix", 0)
        if trix < 0:
            strength = min(abs(trix) * 10, 1.0)
            signals.append(ShortSellSignal(
                name="trix_bearish",
                strength=strength,
                weight=0.05,
                triggered=True,
                reasoning=f"Bearish trend with TRIX {trix:.4f}",
                confidence=0.65,
                category="Momentum"
            ))
        
        # CMO Oversold
        cmo = technical.get("cmo_14", 0)
        if cmo < -50:
            strength = min(abs(cmo) / 50, 1.0)
            signals.append(ShortSellSignal(
                name="cmo_oversold",
                strength=strength,
                weight=0.05,
                triggered=True,
                reasoning=f"CMO oversold at {cmo:.1f}",
                confidence=0.65,
                category="Momentum"
            ))
        
        return signals
    
    def _generate_sentiment_signals(self, sentiment: Dict) -> List[ShortSellSignal]:
        """Generate negative sentiment signals"""
        signals = []
        
        # Negative sentiment
        sentiment_score = sentiment.get("overall_sentiment", 0)
        if sentiment_score < -0.2:
            strength = min(abs(sentiment_score) / 0.8, 1.0)
            signals.append(ShortSellSignal(
                name="negative_sentiment",
                strength=strength,
                weight=0.12,
                triggered=True,
                reasoning=f"Negative sentiment: {sentiment_score:.2f}",
                confidence=0.6,
                category="Sentiment"
            ))
        
        # News deterioration
        news_trend = sentiment.get("news_trend", 0)
        if news_trend < -0.3:
            strength = min(abs(news_trend) / 0.7, 1.0)
            signals.append(ShortSellSignal(
                name="news_deterioration",
                strength=strength,
                weight=0.08,
                triggered=True,
                reasoning="Deteriorating news sentiment",
                confidence=0.5,
                category="Sentiment"
            ))
        
        return signals
    
    def _generate_ml_signals(self, ml_analysis: Dict) -> List[ShortSellSignal]:
        """Generate ML bearish prediction signals"""
        signals = []
        
        # ML Bearish Prediction
        ml_prediction = ml_analysis.get("prediction_direction", 0)
        if ml_prediction < -0.01:
            strength = min(abs(ml_prediction) / 0.10, 1.0)
            signals.append(ShortSellSignal(
                name="ml_bearish",
                strength=strength,
                weight=0.10,
                triggered=True,
                reasoning=f"ML bearish prediction: {ml_prediction:.1%}",
                confidence=ml_analysis.get("confidence", 0.5),
                category="ML"
            ))
        
        # RL SELL Recommendation
        rl_recommendation = ml_analysis.get("rl_recommendation", "HOLD")
        if rl_recommendation == "SELL":
            rl_confidence = ml_analysis.get("rl_confidence", 0.5)
            signals.append(ShortSellSignal(
                name="rl_sell",
                strength=rl_confidence,
                weight=0.05,
                triggered=True,
                reasoning="RL algorithm recommends SELL",
                confidence=rl_confidence,
                category="ML"
            ))
        
        return signals
    
    def _generate_market_signals(self, market_context: 'MarketContext') -> List[ShortSellSignal]:
        """Generate market structure bearish signals"""
        signals = []
        
        # Market stress
        if market_context.market_stress > 0.6:
            signals.append(ShortSellSignal(
                name="market_stress",
                strength=market_context.market_stress,
                weight=0.06,
                triggered=True,
                reasoning=f"High market stress: {market_context.market_stress:.1%}",
                confidence=0.7,
                category="Market"
            ))
        
        # Sector underperformance
        if market_context.sector_performance < -0.02:
            strength = min(abs(market_context.sector_performance) / 0.05, 1.0)
            signals.append(ShortSellSignal(
                name="sector_weakness",
                strength=strength,
                weight=0.04,
                triggered=True,
                reasoning="Sector underperforming market",
                confidence=0.6,
                category="Market"
            ))
        
        return signals
    
    def _generate_value_signals(self, technical: Dict, position: ShortPosition) -> List[ShortSellSignal]:
        """Generate overvaluation signals"""
        signals = []
        
        # High P/E ratio
        pe_ratio = technical.get("pe_ratio", 0)
        sector_pe = technical.get("sector_pe", 20)
        if pe_ratio > sector_pe * 1.5 and pe_ratio > 30:
            strength = min((pe_ratio - sector_pe * 1.5) / (sector_pe * 1.5), 1.0)
            signals.append(ShortSellSignal(
                name="overvalued",
                strength=strength,
                weight=0.05,
                triggered=True,
                reasoning=f"Stock overvalued: P/E {pe_ratio:.1f} vs sector {sector_pe:.1f}",
                confidence=0.6,
                category="Value"
            ))
        
        return signals
    
    def _evaluate_short_bias(
        self,
        market_context: 'MarketContext',
        technical_analysis: Dict,
        sentiment_analysis: Dict,
        ml_analysis: Dict
    ) -> float:
        """
        AUTO-DETECTION: Evaluate overall market bias for short-selling
        
        Returns a confidence score (0.0 to 1.0) indicating how favorable
        conditions are for short-selling vs long-buying.
        
        HIGH SCORE (> 0.65): Market favors shorts (bearish)
        LOW SCORE (< 0.35): Market favors longs (bullish)
        MEDIUM (0.35-0.65): Neutral/mixed conditions
        
        Factors considered:
        1. Market trend direction
        2. Technical breakdown signals
        3. Sentiment deterioration
        4. ML bearish predictions
        5. Market stress levels
        """
        
        bias_score = 0.0
        max_score = 10.0  # Total possible points
        
        logger.info(f"\n📊 Evaluating Short Bias...")
        
        # Factor 1: Market Trend (max 2.5 points)
        from .professional_sell_logic import MarketTrend
        trend = market_context.trend
        trend_strength = market_context.trend_strength
        
        if trend == MarketTrend.STRONG_DOWNTREND:
            trend_points = 2.5 * trend_strength
            logger.info(f"   ✅ Strong Downtrend: +{trend_points:.2f} points")
        elif trend == MarketTrend.DOWNTREND:
            trend_points = 2.0 * trend_strength
            logger.info(f"   ✅ Downtrend: +{trend_points:.2f} points")
        elif trend == MarketTrend.SIDEWAYS:
            trend_points = 1.0  # Neutral
            logger.info(f"   ⚠️ Sideways: +{trend_points:.2f} points")
        elif trend == MarketTrend.UPTREND:
            trend_points = 0.3 * (1 - trend_strength)
            logger.info(f"   ❌ Uptrend: +{trend_points:.2f} points (penalized)")
        else:  # STRONG_UPTREND
            trend_points = 0.1 * (1 - trend_strength)
            logger.info(f"   ❌ Strong Uptrend: +{trend_points:.2f} points (heavily penalized)")
        
        bias_score += trend_points
        
        # Factor 2: Technical Breakdown (max 2.5 points)
        rsi = technical_analysis.get("rsi", 50)
        macd = technical_analysis.get("macd", 0)
        macd_signal = technical_analysis.get("macd_signal", 0)
        price_vs_ma = technical_analysis.get("current_price", 0) < technical_analysis.get("sma_20", 0)
        
        technical_points = 0.0
        if rsi < 30:  # Oversold
            technical_points += 1.0
            logger.info(f"   ✅ RSI Oversold ({rsi:.1f}): +1.0 point")
        elif rsi < 40:
            technical_points += 0.5
            logger.info(f"   ⚠️ RSI Weak ({rsi:.1f}): +0.5 points")
        
        if macd < macd_signal and macd < 0:  # Bearish MACD
            technical_points += 1.0
            logger.info(f"   ✅ MACD Bearish: +1.0 point")
        
        if price_vs_ma:  # Price below SMA20
            technical_points += 0.5
            logger.info(f"   ✅ Price Below MA: +0.5 points")
        
        bias_score += min(technical_points, 2.5)
        
        # Factor 3: Negative Sentiment (max 2.0 points)
        sentiment = sentiment_analysis.get("overall_sentiment", 0)
        news_trend = sentiment_analysis.get("news_trend", 0)
        
        sentiment_points = 0.0
        if sentiment < -0.3:
            sentiment_points += 1.5
            logger.info(f"   ✅ Negative Sentiment ({sentiment:.2f}): +1.5 points")
        elif sentiment < -0.1:
            sentiment_points += 0.7
            logger.info(f"   ⚠️ Weak Sentiment ({sentiment:.2f}): +0.7 points")
        
        if news_trend < -0.3:
            sentiment_points += 0.5
            logger.info(f"   ✅ Negative News Trend: +0.5 points")
        
        bias_score += min(sentiment_points, 2.0)
        
        # Factor 4: ML Bearish Prediction (max 2.0 points)
        ml_prediction = ml_analysis.get("prediction_direction", 0)
        rl_recommendation = ml_analysis.get("rl_recommendation", "HOLD")
        ml_confidence = ml_analysis.get("confidence", 0.5)
        
        ml_points = 0.0
        if ml_prediction < -0.03:  # Strong bearish prediction
            ml_points += 1.5 * ml_confidence
            logger.info(f"   ✅ ML Bearish ({ml_prediction:.1%}): +{ml_points:.2f} points")
        elif ml_prediction < -0.01:  # Moderate bearish
            ml_points += 0.7 * ml_confidence
            logger.info(f"   ⚠️ ML Weak Bearish ({ml_prediction:.1%}): +{ml_points:.2f} points")
        
        if rl_recommendation == "SELL":
            rl_conf = ml_analysis.get("rl_confidence", 0.5)
            ml_points += 0.5 * rl_conf
            logger.info(f"   ✅ RL SELL: +{0.5 * rl_conf:.2f} points")
        
        bias_score += min(ml_points, 2.0)
        
        # Factor 5: Market Stress (max 1.0 point)
        market_stress = market_context.market_stress
        sector_performance = market_context.sector_performance
        
        stress_points = 0.0
        if market_stress > 0.7:
            stress_points = 0.7
            logger.info(f"   ✅ High Market Stress ({market_stress:.2f}): +0.7 points")
        elif market_stress > 0.5:
            stress_points = 0.4
            logger.info(f"   ⚠️ Moderate Stress ({market_stress:.2f}): +0.4 points")
        
        if sector_performance < -0.02:
            stress_points += 0.3
            logger.info(f"   ✅ Sector Underperformance ({sector_performance:.1%}): +0.3 points")
        
        bias_score += min(stress_points, 1.0)
        
        # Calculate final confidence score
        final_confidence = bias_score / max_score
        final_confidence = min(max(final_confidence, 0.0), 1.0)  # Clamp to [0, 1]
        
        logger.info(f"\n📊 Short Bias Evaluation Complete:")
        logger.info(f"   Total Score: {bias_score:.2f} / {max_score}")
        logger.info(f"   Confidence: {final_confidence:.3f}")
        
        # Interpretation guide
        if final_confidence >= 0.65:
            logger.info(f"   🎯 INTERPRETATION: STRONG SHORT BIAS ✅")
        elif final_confidence >= 0.50:
            logger.info(f"   🎯 INTERPRETATION: MODERATE SHORT BIAS ⚠️")
        elif final_confidence >= 0.35:
            logger.info(f"   🎯 INTERPRETATION: NEUTRAL/MIXED ➖")
        else:
            logger.info(f"   🎯 INTERPRETATION: LONG BIAS (avoid shorts) ❌")
        
        return final_confidence
    
    def _calculate_short_stops(self, position: ShortPosition, market_context: 'MarketContext') -> Dict:
        """Calculate dynamic stop-loss levels for short positions (REVERSED)"""
        
        # Base stop-loss (ABOVE entry for shorts)
        base_stop = position.entry_price * (1 + self.base_stop_loss_pct)
        
        # Use stored stop-loss if available
        if position.db_stop_loss is not None:
            base_stop = max(base_stop, position.db_stop_loss)  # Use more conservative
        
        # Volatility-adjusted stop
        volatility_multiplier = 1 + (position.volatility / 0.02)
        calculated_volatility_stop = position.entry_price * (1 + self.base_stop_loss_pct * volatility_multiplier)
        volatility_stop = min(calculated_volatility_stop, base_stop)  # Use more conservative
        
        # Take-profit (BELOW entry for shorts)
        take_profit = position.entry_price * (1 - self.base_stop_loss_pct * 2)  # 2:1 reward-risk
        
        # Use stored target if available
        if position.db_target_price is not None:
            take_profit = min(take_profit, position.db_target_price)  # Use more conservative
        
        # Profit protection (lock in gains as price falls)
        profit_protection_stop = 0.0
        if position.unrealized_pnl_pct > self.profit_protection_threshold:
            # Lock in profits after threshold
            protected_gain = position.lowest_price_since_entry * (1 + self.profit_protection_threshold)
            profit_protection_stop = protected_gain
        
        # Use most conservative stop (lowest for shorts)
        active_stop = min(base_stop, volatility_stop)
        
        return {
            "base_stop": base_stop,
            "volatility_stop": volatility_stop,
            "profit_protection_stop": profit_protection_stop,
            "take_profit": take_profit,
            "active_stop": active_stop
        }
    
    def _apply_market_context_filters(self, decision: ShortDecision, market_context: 'MarketContext') -> ShortDecision:
        """Apply market context filters (more aggressive in downtrends)"""
        
        if not decision.should_short:
            return decision
        
        # More aggressive in downtrends
        if market_context.trend in [MarketTrend.DOWNTREND, MarketTrend.STRONG_DOWNTREND]:
            if decision.urgency < 0.5:
                decision.confidence *= self.downtrend_short_multiplier
                logger.info(f"Downtrend boost - confidence increased to {decision.confidence:.3f}")
        
        # Less aggressive in uptrends
        elif market_context.trend in [MarketTrend.UPTREND, MarketTrend.STRONG_UPTREND]:
            original_confidence = decision.confidence
            decision.confidence *= self.uptrend_short_multiplier
            logger.info(f"Uptrend moderation - confidence reduced from {original_confidence:.3f} to {decision.confidence:.3f}")
        
        return decision
    
    def _determine_position_sizing(
        self,
        decision: ShortDecision,
        signals: List[ShortSellSignal],
        position: ShortPosition
    ) -> ShortDecision:
        """Determine how much to short-sell"""
        
        if not decision.should_short:
            return decision
        
        logger.info(f"Determining position sizing for {position.quantity} shares")
        logger.info(f"Decision confidence: {decision.confidence:.3f}")
        logger.info(f"Decision urgency: {decision.urgency:.3f}")
        logger.info(f"Unrealized P&L: {position.unrealized_pnl_pct:.2%}")
        
        # Emergency exit (immediate buyback)
        if (decision.confidence >= self.emergency_exit_threshold or
            decision.urgency >= 0.95 or
            position.unrealized_pnl_pct < -0.10):  # 10% loss on short
            
            decision.short_quantity = position.quantity
            decision.short_percentage = 1.0
            decision.reasoning += " | EMERGENCY BUYBACK"
            logger.info(f"Emergency buyback - covering 100% ({position.quantity} shares)")
        
        # Full exit
        elif (decision.confidence >= self.full_entry_threshold or
              decision.urgency >= 0.85 or
              position.unrealized_pnl_pct < -0.08):
            
            decision.short_quantity = position.quantity
            decision.short_percentage = 1.0
            decision.reasoning += " | FULL BUYBACK"
            logger.info(f"Full buyback - covering 100% ({position.quantity} shares)")
        
        # Aggressive exit
        elif decision.confidence >= self.aggressive_entry_threshold:
            exit_percentage = min(decision.confidence * decision.urgency * 1.2, 0.9)
            decision.short_quantity = int(position.quantity * exit_percentage)
            decision.short_percentage = exit_percentage
            decision.reasoning += f" | AGGRESSIVE BUYBACK ({exit_percentage:.1%})"
            logger.info(f"Aggressive buyback - covering {exit_percentage:.1%} ({decision.short_quantity} shares)")
        
        # Partial exit
        elif decision.confidence >= self.partial_entry_threshold:
            exit_percentage = min(decision.confidence * decision.urgency, 0.75)
            decision.short_quantity = max(1, int(position.quantity * exit_percentage))
            decision.short_percentage = exit_percentage
            decision.reasoning += f" | PARTIAL BUYBACK ({exit_percentage:.1%})"
            logger.info(f"Partial buyback - covering {exit_percentage:.1%} ({decision.short_quantity} shares)")
        
        # Conservative exit
        elif decision.confidence >= self.conservative_entry_threshold:
            exit_percentage = min(decision.confidence * decision.urgency * 0.5, 0.5)
            decision.short_quantity = max(1, int(position.quantity * exit_percentage))
            decision.short_percentage = exit_percentage
            decision.reasoning += f" | CONSERVATIVE BUYBACK ({exit_percentage:.1%})"
            logger.info(f"Conservative buyback - covering {exit_percentage:.1%} ({decision.short_quantity} shares)")
        
        else:
            # Minimal exit
            decision.short_quantity = max(1, int(position.quantity * 0.1))
            decision.short_percentage = 0.1
            decision.reasoning += " | MINIMAL BUYBACK (10%)"
            logger.info(f"Minimal buyback - covering 10% ({decision.short_quantity} shares)")
        
        # Ensure we don't cover more than we have
        decision.short_quantity = min(decision.short_quantity, position.quantity)
        logger.info(f"Final buyback quantity: {decision.short_quantity} shares ({decision.short_percentage:.1%})")
        
        return decision


# Import MarketContext from sell logic
from .professional_sell_logic import MarketContext, MarketTrend
