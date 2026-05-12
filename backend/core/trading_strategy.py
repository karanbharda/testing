"""
Unified Trading Strategy for BUY, SHORT, and SELL decisions
Supports both swing trading (CNC) and intraday trading (MIS/INTRADAY)
"""

import logging
import os
from typing import Dict, Optional
import pandas as pd

from .professional_buy_integration import ProfessionalBuyIntegration
from .professional_sell_integration import ProfessionalSellIntegration

logger = logging.getLogger(__name__)


class TradingStrategy:
    """
    Unified trading strategy that evaluates BUY, SHORT, and SELL signals
    
    For CNC (Delivery):
    - Only BUY and SELL operations
    
    For INTRADAY/MIS (Intraday):
    - BUY (go LONG), SHORT (go SHORT), SELL (close LONG), BUY-TO-COVER (close SHORT)
    """
    
    def __init__(self, config: Dict):
        """Initialize trading strategy with professional integrations"""
        self.config = config
        self.product_type = config.get("productType", "CNC")
        
        # Initialize professional integrations
        self.buy_integration = ProfessionalBuyIntegration(config)
        self.sell_integration = ProfessionalSellIntegration(config)
        
        # Enable SHORT selling for intraday trading
        self.enable_shortsell = config.get("enable_shortsell", True)
        self.min_short_signal_strength = config.get("min_short_signal_strength", -0.4)
        self.min_buy_signal_strength = config.get("min_buy_signal_strength", 0.6)
        
        logger.info(f"TradingStrategy initialized for {self.product_type} trading")
        logger.info(f"  Short selling enabled: {self.enable_shortsell}")
        logger.info(f"  Buy signal threshold: {self.min_buy_signal_strength}")
        logger.info(f"  Short signal threshold: {self.min_short_signal_strength}")
    
    def make_trade(self, analysis: Dict, portfolio_context: Optional[Dict] = None) -> Dict:
        """
        Make a unified trading decision based on analysis
        
        Args:
            analysis: Stock analysis data with technical, ML, sentiment signals
            portfolio_context: Current portfolio holdings and cash
            
        Returns:
            Trade decision dict with action, qty, price, etc.
        """
        try:
            if not analysis.get("success"):
                logger.warning(f"Analysis failed, cannot make trade decision")
                return {
                    "success": False,
                    "message": "Analysis not successful"
                }
            
            ticker = analysis.get("ticker")
            current_price = analysis.get("current_price", 0)
            
            if not ticker or current_price <= 0:
                logger.warning(f"Invalid ticker or price in analysis")
                return {
                    "success": False,
                    "message": "Invalid ticker or price"
                }
            
            logger.info(f"=== MAKING UNIFIED TRADING DECISION FOR {ticker} ===")
            
            # Evaluate buy signals
            buy_decision = self.buy_integration.evaluate_professional_buy(
                ticker=ticker,
                current_price=current_price,
                analysis_data=analysis,
                price_history=self._extract_price_history(analysis),
                portfolio_context=portfolio_context
            )
            
            buy_action = buy_decision.get("action", "hold")
            buy_confidence = buy_decision.get("confidence_score", 0)
            
            logger.info(f"Buy evaluation: action={buy_action}, confidence={buy_confidence:.2%}")
            
            # Get overall signal strength from analysis (-1 to +1)
            signal_strength = self._extract_signal_strength(analysis)
            logger.info(f"Overall signal strength: {signal_strength:.3f}")
            
            # Determine if we should consider SHORT
            is_intraday = self._is_intraday()
            can_short = self.enable_shortsell and is_intraday
            
            # Make trading decision based on signal strength
            if signal_strength >= self.min_buy_signal_strength and buy_action == "buy":
                # Strong BUY signal
                logger.info(f"✅ BUY SIGNAL CONFIRMED: strength={signal_strength:.3f}, confidence={buy_confidence:.2%}")
                return self._format_buy_decision(buy_decision)
            
            elif signal_strength <= self.min_short_signal_strength and can_short:
                # Strong SHORT signal (intraday only)
                logger.info(f"📉 SHORT SIGNAL DETECTED: strength={signal_strength:.3f}, can_short={can_short}")
                short_decision = self._evaluate_short_decision(
                    ticker, current_price, analysis, signal_strength
                )
                if short_decision["action"] == "short":
                    logger.info(f"✅ SHORT POSITION OPENED: {short_decision}")
                    return short_decision
            
            # Check if we should evaluate SELL
            if portfolio_context:
                holdings = portfolio_context.get("holdings", {})
                if ticker in holdings and holdings[ticker].get("quantity", 0) > 0:
                    # We have a LONG position - evaluate if we should sell
                    sell_decision = self.sell_integration.evaluate_professional_sell(
                        ticker=ticker,
                        current_price=current_price,
                        portfolio_holdings=holdings,
                        analysis_data=analysis,
                        price_history=self._extract_price_history(analysis)
                    )
                    
                    if sell_decision.get("action") == "sell":
                        logger.info(f"📊 SELL SIGNAL: {sell_decision}")
                        return self._format_sell_decision(sell_decision)
            
            # Default: HOLD
            logger.info(f"🟡 HOLD DECISION: No clear buy/sell/short signal (strength={signal_strength:.3f})")
            return {
                "success": True,
                "action": "hold",
                "ticker": ticker,
                "qty": 0,
                "price": current_price,
                "confidence_score": 0,
                "reason": "no_clear_signal",
                "message": f"No clear signal for {ticker}. Signal strength: {signal_strength:.3f}"
            }
        
        except Exception as e:
            logger.error(f"Error in make_trade: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error making trade decision: {str(e)}"
            }
    
    def _evaluate_short_decision(self, ticker: str, current_price: float, 
                                 analysis: Dict, signal_strength: float) -> Dict:
        """Evaluate SHORT selling opportunity for intraday trading"""
        try:
            # For short, we want bearish signals
            # Extract bearish indicators from analysis
            technical = analysis.get("technical_indicators", {})
            sentiment = analysis.get("sentiment_analysis", {})
            
            # Calculate short confidence (inverse of buying)
            short_confidence = min(abs(signal_strength), 1.0)
            
            logger.info(f"Evaluating SHORT for {ticker}: strength={signal_strength:.3f}, confidence={short_confidence:.2%}")
            
            # Check if SHORT signal is strong enough
            if short_confidence < abs(self.min_short_signal_strength):
                logger.info(f"Short signal too weak: {short_confidence:.2%} < {abs(self.min_short_signal_strength):.2%}")
                return {"action": "hold", "reason": "short_signal_too_weak"}
            
            # Calculate position size for SHORT
            position_size = self._calculate_short_position_size(
                current_price, short_confidence
            )
            
            # Get stop loss and take profit for SHORT
            stop_loss, take_profit = self._calculate_short_levels(
                current_price, analysis
            )
            
            return {
                "success": True,
                "action": "short",
                "ticker": ticker,
                "qty": position_size,
                "price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence_score": short_confidence,
                "signal_strength": signal_strength,
                "reason": "bearish_signal",
                "product_type": "MIS",
                "message": f"SHORT signal for {ticker} at Rs.{current_price:.2f} (confidence: {short_confidence:.2%})"
            }
        
        except Exception as e:
            logger.error(f"Error evaluating short decision: {e}")
            return {"action": "hold", "reason": "short_evaluation_error"}
    
    def _calculate_short_position_size(self, current_price: float, confidence: float) -> int:
        """Calculate position size for SHORT position"""
        # Position size scales with confidence
        # Conservative sizing: 1-10 shares based on confidence
        base_quantity = max(1, int(10 * confidence))
        return base_quantity
    
    def _calculate_short_levels(self, current_price: float, analysis: Dict) -> tuple:
        """Calculate stop-loss and take-profit levels for SHORT position"""
        # For short: SL above current price, TP below current price
        atr = analysis.get("technical_indicators", {}).get("atr", current_price * 0.02)
        
        # Stop loss at 1.5x ATR above
        stop_loss = current_price + (1.5 * atr)
        
        # Take profit at 2x ATR below  
        take_profit = current_price - (2 * atr)
        
        return stop_loss, take_profit
    
    def _is_intraday(self) -> bool:
        """Check if trading is intraday (MIS/INTRADAY) mode"""
        product_type = str(self.product_type).upper()
        return product_type in ["MIS", "INTRADAY", "MARGIN", "NRML"]
    
    def _extract_signal_strength(self, analysis: Dict) -> float:
        """Extract overall signal strength from analysis (-1.0 to +1.0)"""
        # Look for combined signal or calculate from components
        technical = analysis.get("technical_indicators", {})
        
        # Check for combination score
        if "combination_score" in technical:
            return technical["combination_score"]
        
        # Try RSI-based signal
        if "rsi" in technical:
            rsi = technical["rsi"]
            # Convert RSI (0-100) to signal (-1 to +1)
            # RSI > 70 = overbought (bearish = -), RSI < 30 = oversold (bullish = +)
            if rsi > 70:
                return -0.5  # Bearish
            elif rsi < 30:
                return 0.5   # Bullish
            else:
                return 0.0   # Neutral
        
        # Try MACD signal
        if "macd_signal" in technical:
            macd = technical.get("macd", 0)
            signal = technical.get("macd_signal", 0)
            if macd > signal:
                return 0.4   # Bullish
            else:
                return -0.4  # Bearish
        
        # Default neutral
        return 0.0
    
    def _extract_price_history(self, analysis: Dict) -> Optional[pd.DataFrame]:
        """Extract price history from analysis if available"""
        if "price_history" in analysis:
            return analysis["price_history"]
        return None
    
    def _format_buy_decision(self, buy_decision: Dict) -> Dict:
        """Format buy decision for output"""
        return {
            "success": True,
            "action": "buy",
            "ticker": buy_decision.get("ticker"),
            "qty": buy_decision.get("qty", 0),
            "price": buy_decision.get("price", 0),
            "stop_loss": buy_decision.get("stop_loss", 0),
            "take_profit": buy_decision.get("take_profit", 0),
            "confidence_score": buy_decision.get("confidence_score", 0),
            "reason": buy_decision.get("reason", "buy_signal"),
            "message": buy_decision.get("professional_reasoning", "BUY signal confirmed")
        }
    
    def _format_sell_decision(self, sell_decision: Dict) -> Dict:
        """Format sell decision for output"""
        return {
            "success": True,
            "action": "sell",
            "ticker": sell_decision.get("ticker"),
            "qty": sell_decision.get("qty", 0),
            "price": sell_decision.get("price", 0),
            "confidence_score": sell_decision.get("confidence_score", 0),
            "reason": sell_decision.get("reason", "sell_signal"),
            "message": sell_decision.get("professional_reasoning", "SELL signal confirmed")
        }
