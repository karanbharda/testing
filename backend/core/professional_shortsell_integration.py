"""
Professional Short-Sell Logic Integration
Integrates the professional short-sell logic with existing trading modules
for intraday short-selling (sell first, buy back later)
"""

import logging
import os
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from .professional_shortsell_logic import (
    ProfessionalShortSellLogic, ShortPosition, ShortDecision, ShortSellReason
)
from .market_context_analyzer import MarketContextAnalyzer
from .professional_sell_logic import MarketContext, MarketTrend

# PRODUCTION ENHANCEMENT: Import signal tracker
from utils.signal_tracker import get_signal_tracker

logger = logging.getLogger(__name__)


class ProfessionalShortSellIntegration:
    """
    Integration layer for professional short-sell logic
    Handles intraday short-selling: Sell high first, then buy back low
    
    AUTO-DETECTION FEATURE:
    - Automatically evaluates if market favors shorts or longs
    - Works seamlessly with long logic for optimal direction selection
    - No manual configuration needed - system decides based on conditions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize professional components
        self.shortsell_logic = ProfessionalShortSellLogic(config)
        
        # PRODUCTION ENHANCEMENT: Initialize signal tracker
        self.signal_tracker = get_signal_tracker()
        
        # Integration settings
        self.enable_professional_logic = config.get("enable_professional_shortsell_logic", True)
        self.fallback_to_legacy = config.get("fallback_to_legacy_shortsell", False)
        self.auto_detect_direction = config.get("auto_detect_trade_direction", True)
        
        logger.info("Professional Short-Sell Integration initialized with AUTO-DETECTION")
    
    def refresh_dynamic_config(self):
        """Refresh dynamic configuration from live_config.json"""
        self.shortsell_logic.refresh_dynamic_config()
    
    def evaluate_professional_shortsell(
        self,
        ticker: str,
        current_price: float,
        portfolio_holdings: Dict,
        analysis_data: Dict,
        price_history: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Main integration point for professional short-sell evaluation
        Returns a decision compatible with existing trading modules
        REVERSE LOGIC: Sell first at high price, then buy back at low price
        """
        # Check if short-sell is disabled by configuration
        enable_shortsell = str(os.getenv("ENABLE_SHORTSELL", "false")).lower() not in ("false", "0", "no", "off")
        if not enable_shortsell:
            logger.info(f"❌ SHORT-SELL DISABLED: Short-sell signals globally disabled by configuration (ENABLE_SHORTSELL=false) for {ticker}")
            return {
                "action": "hold",
                "ticker": ticker,
                "qty": 0,
                "price": current_price,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "success": True,
                "confidence_score": 0.0,
                "signals": 0,
                "reason": "shortsell_disabled",
                "professional_reasoning": "Short-sell functionality disabled by configuration"
            }
        
        try:
            logger.info(f"=== STARTING PROFESSIONAL SHORT-SELL EVALUATION FOR {ticker} ===")
            
            # Check if we have a short position
            if ticker not in portfolio_holdings:
                logger.info(f"No short position found for {ticker}, returning no position decision")
                return self._no_position_decision()
            
            # Build short position metrics
            position_metrics = self._build_short_position_metrics(
                ticker, current_price, portfolio_holdings, price_history
            )
            
            # Build market context
            market_context = self._build_market_context(analysis_data, price_history)
            
            # Extract analysis components
            technical_analysis = analysis_data.get("technical_indicators", {})
            sentiment_analysis = analysis_data.get("sentiment", {})
            ml_analysis = analysis_data.get("ml_analysis", {})
            
            # Get professional short-sell decision
            shortsell_decision = self.shortsell_logic.evaluate_short_decision(
                ticker=ticker,
                position_metrics=position_metrics,
                market_context=market_context,
                technical_analysis=technical_analysis,
                sentiment_analysis=sentiment_analysis,
                ml_analysis=ml_analysis
            )
            
            # Log detailed decision information
            logger.info(f"Professional Short-Sell Decision for {ticker}:")
            logger.info(f"  Should Buyback (Cover): {shortsell_decision.should_short}")
            logger.info(f"  Confidence: {shortsell_decision.confidence:.3f}")
            logger.info(f"  Buyback Percentage: {shortsell_decision.short_percentage:.3f}")
            logger.info(f"  Reason: {shortsell_decision.reason}")
            logger.info(f"  Signals Triggered: {len(shortsell_decision.signals_triggered)}")
            logger.info(f"  Reasoning: {shortsell_decision.reasoning}")
            
            # Convert to legacy format
            result = self._convert_to_legacy_format(shortsell_decision, position_metrics)
            
            # Log the final result
            if result['action'] == 'buy':  # Buyback to cover short
                logger.info(f"🔴 SHORT BUYBACK DECISION: {result['qty']} shares of {ticker}")
                logger.info(f"   Price: ₹{result['price']:.2f}")
                logger.info(f"   Stop Loss: ₹{result['stop_loss']:.2f}")
                logger.info(f"   Take Profit: ₹{result['take_profit']:.2f}")
                logger.info(f"   Confidence: {result['confidence_score']:.3f}")
                logger.info(f"   Reason: {result['reason']}")
                logger.info(f"   Professional Reasoning: {result['professional_reasoning']}")
            elif result['action'] == 'hold':
                logger.info(f"🟡 HOLD SHORT POSITION: {ticker}")
                logger.info(f"   Reason: {result['reason']}")
                logger.info(f"   Professional Reasoning: {result['professional_reasoning']}")
                logger.info(f"   Confidence: {result['confidence_score']:.3f}")
            else:
                logger.info(f"🔴 UNEXPECTED DECISION: {result['action']} for {ticker}")
            
            logger.info(f"=== END PROFESSIONAL SHORT-SELL EVALUATION FOR {ticker} ===")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in professional short-sell evaluation for {ticker}: {e}")
            logger.exception("Full traceback:")
            if self.fallback_to_legacy:
                logger.info("Falling back to legacy short-sell logic")
                return self._legacy_shortsell_decision(ticker, current_price, portfolio_holdings, analysis_data)
            else:
                return self._error_decision(str(e))
    
    def _build_short_position_metrics(
        self,
        ticker: str,
        current_price: float,
        portfolio_holdings: Dict,
        price_history: Optional[pd.DataFrame]
    ) -> ShortPosition:
        """Build short position metrics from portfolio data"""
        
        # Helper function to safely convert values to float
        def safe_float(value, default=0.0):
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return default
        
        holding = portfolio_holdings[ticker]
        entry_price = safe_float(holding.get("avg_price", current_price), current_price)
        quantity = safe_float(holding.get("qty", 0), 0)
        
        # Calculate P&L for short position (REVERSED: profit when price goes down)
        unrealized_pnl = (entry_price - current_price) * quantity  # Entry - Current for shorts
        unrealized_pnl_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0.0
        
        # Calculate days held (for intraday, this should be minutes)
        days_held = safe_float(holding.get("days_held", 1), 1)
        
        # Calculate highest/lowest prices since entry
        highest_price = current_price
        lowest_price = current_price
        volatility = 0.02  # Default volatility
        
        if price_history is not None and len(price_history) > 1:
            # Use actual price history if available
            recent_prices = price_history['Close'].tail(30)
            highest_price = recent_prices.max()
            lowest_price = recent_prices.min()
            
            # Calculate volatility
            returns = recent_prices.pct_change().dropna()
            if len(returns) > 1:
                volatility = returns.std() * (252 ** 0.5)
        
        # Fetch stop-loss and target price from database
        db_stop_loss, db_target_price = self._fetch_db_stop_loss_target(ticker)
        
        # Calculate time remaining in trading day (for intraday)
        time_remaining_minutes = self._calculate_time_remaining()
        
        logger.info(f"Short Position Metrics for {ticker}:")
        logger.info(f"  Entry Price (Short): {entry_price:.2f}")
        logger.info(f"  Current Price: {current_price:.2f}")
        logger.info(f"  DB Stop-Loss (Buyback): {db_stop_loss if db_stop_loss else 'Not set'}")
        logger.info(f"  DB Target Price (Profit): {db_target_price if db_target_price else 'Not set'}")
        logger.info(f"  Time Remaining: {time_remaining_minutes} minutes")
        
        return ShortPosition(
            entry_price=entry_price,
            current_price=current_price,
            quantity=quantity,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            lowest_price_since_entry=lowest_price,
            highest_price_since_entry=highest_price,
            volatility=volatility,
            time_remaining_minutes=time_remaining_minutes,
            db_stop_loss=db_stop_loss,
            db_target_price=db_target_price
        )
    
    def _fetch_db_stop_loss_target(self, ticker: str) -> Tuple[Optional[float], Optional[float]]:
        """Fetch stop-loss and target price from database for short position"""
        try:
            from ..db.database import DatabaseManager
            
            db_manager = DatabaseManager()
            session = db_manager.Session()
            
            try:
                # Query the most recent SELL trade for this ticker (short entry)
                from ..db.database import Trade
                latest_sell = session.query(Trade).filter(
                    Trade.ticker == ticker,
                    Trade.action == 'sell'
                ).order_by(Trade.timestamp.desc()).first()
                
                # Helper function to safely convert values to float
                def safe_float(value, default=None):
                    if isinstance(value, str):
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return default
                    elif isinstance(value, (int, float)):
                        return float(value) if value > 0 else default
                    else:
                        return default
                
                if latest_sell:
                    stop_loss = safe_float(latest_sell.stop_loss)  # Buyback stop-loss
                    target_price = safe_float(latest_sell.take_profit)  # Profit booking target
                    
                    if stop_loss or target_price:
                        logger.info(f"📊 Database values retrieved for short {ticker}: Stop-Loss={stop_loss}, Target={target_price}")
                    
                    return stop_loss, target_price
                else:
                    logger.debug(f"No sell trade (short entry) found in database for {ticker}")
                    return None, None
                    
            finally:
                session.close()
                
        except Exception as e:
            logger.warning(f"Failed to fetch database stop-loss/target for short {ticker}: {e}")
            return None, None
    
    def _calculate_time_remaining(self) -> int:
        """Calculate time remaining in trading day (minutes)"""
        try:
            from datetime import datetime
            
            # Indian market hours: 9:15 AM - 3:30 PM
            now = datetime.now()
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # If already past close, return 0
            if now >= market_close:
                return 0
            
            # Calculate minutes remaining
            time_diff = market_close - now
            return int(time_diff.total_seconds() / 60)
            
        except Exception as e:
            logger.error(f"Error calculating time remaining: {e}")
            return 0
    
    def _build_market_context(self, analysis_data: Dict, price_history: Optional[pd.DataFrame]):
        """Build market context from analysis data"""
        
        if price_history is not None and len(price_history) > 50:
            # Use professional market analysis
            return self.market_analyzer.analyze_market_context(price_history)
        else:
            # Fallback to basic context from analysis data
            technical = analysis_data.get("technical_indicators", {})
            
            # Helper function to safely convert values to float
            def safe_float(value, default=0.0):
                if isinstance(value, str):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    return default
            
            # Extract basic trend info
            sma_20 = safe_float(technical.get("sma_20", 0), 0)
            sma_50 = safe_float(technical.get("sma_50", 0), 0)
            current_price = safe_float(technical.get("current_price", 0), 0)
            
            # Simple trend detection
            if current_price < sma_20 < sma_50:
                trend = MarketTrend.DOWNTREND
                trend_strength = 0.6
            elif current_price > sma_20 > sma_50:
                trend = MarketTrend.UPTREND
                trend_strength = 0.6
            else:
                trend = MarketTrend.SIDEWAYS
                trend_strength = 0.3
            
            return MarketContext(
                trend=trend,
                trend_strength=trend_strength,
                volatility_regime="normal",
                market_stress=0.3,
                sector_performance=0.0,
                volume_profile=0.5
            )
    
    def _convert_to_legacy_format(self, shortsell_decision: ShortDecision, position_metrics: ShortPosition) -> Dict:
        """Convert professional short-sell decision to legacy format"""
        
        # PRODUCTION ENHANCEMENT: Track signals for continuous learning
        self._track_signals(shortsell_decision)
        
        if not shortsell_decision.should_short:
            # Use stored database values if available, otherwise use calculated values
            stop_loss = position_metrics.db_stop_loss if position_metrics.db_stop_loss is not None else shortsell_decision.stop_loss_price
            take_profit = position_metrics.db_target_price if position_metrics.db_target_price is not None else shortsell_decision.take_profit_price
            
            return {
                "action": "hold",
                "ticker": "",
                "qty": 0,
                "price": position_metrics.current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "success": True,
                "confidence_score": shortsell_decision.confidence,
                "signals": len(shortsell_decision.signals_triggered),
                "reason": "professional_hold",
                "professional_reasoning": shortsell_decision.reasoning
            }
        
        # Use stored database values if available, otherwise use calculated values
        stop_loss = position_metrics.db_stop_loss if position_metrics.db_stop_loss is not None else shortsell_decision.stop_loss_price
        take_profit = position_metrics.db_target_price if position_metrics.db_target_price is not None else shortsell_decision.take_profit_price
        
        return {
            "action": "buy",  # Buyback to cover short position
            "ticker": "",  # Will be set by calling module
            "qty": shortsell_decision.short_quantity,
            "price": position_metrics.current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "success": True,
            "confidence_score": shortsell_decision.confidence,
            "signals": len(shortsell_decision.signals_triggered),
            "reason": shortsell_decision.reason.value,
            "professional_reasoning": shortsell_decision.reasoning,
            "buyback_percentage": shortsell_decision.short_percentage,
            "urgency": shortsell_decision.urgency
        }
    
    def _no_position_decision(self) -> Dict:
        """Return decision when no short position exists"""
        return {
            "action": "hold",
            "ticker": "",
            "qty": 0,
            "price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "success": True,
            "confidence_score": 0.0,
            "signals": 0,
            "reason": "no_position",
            "professional_reasoning": "No short position to cover"
        }
    
    def _legacy_shortsell_decision(self, ticker: str, current_price: float, portfolio_holdings: Dict, analysis_data: Dict) -> Dict:
        """Fallback to legacy short-sell logic"""
        return {
            "action": "hold",
            "ticker": ticker,
            "qty": 0,
            "price": current_price,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "success": True,
            "confidence_score": 0.0,
            "signals": 0,
            "reason": "legacy_fallback",
            "professional_reasoning": "Using legacy short-sell logic"
        }
    
    def _track_signals(self, shortsell_decision: ShortDecision):
        """
        PRODUCTION ENHANCEMENT: Track signals for continuous learning
        """
        try:
            # Record each triggered signal
            for signal in shortsell_decision.signals_triggered:
                if signal.triggered:
                    signal_data = {
                        'symbol': '',  # Will be filled in by calling function
                        'signal_type': signal.category.lower() if signal.category else 'unknown',
                        'signal_name': signal.name,
                        'signal_strength': signal.strength,
                        'signal_confidence': signal.confidence,
                        'market_regime': 'unknown',
                        'liquidity_score': 0.5,
                        'volatility_regime': 'normal',
                        'additional_metrics': {
                            'reasoning': signal.reasoning,
                            'weight': signal.weight
                        }
                    }
                    
                    # Record the signal
                    self.signal_tracker.record_signal(signal_data)
            
            logger.debug(f"Tracked {len(shortsell_decision.signals_triggered)} short-sell signals for continuous learning")
            
        except Exception as e:
            logger.warning(f"Error tracking short-sell signals: {e}")
    
    def _error_decision(self, error_msg: str) -> Dict:
        """Return decision when error occurs"""
        return {
            "action": "hold",
            "ticker": "",
            "qty": 0,
            "price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "success": False,
            "confidence_score": 0.0,
            "signals": 0,
            "reason": "error",
            "professional_reasoning": f"Error: {error_msg}"
        }
