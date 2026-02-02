#!/usr/bin/env python3
"""
Finance Reasoning Service
========================

Service layer for the LangGraph Finance Reasoning Engine integration.
Provides high-level API for market analysis and decision reasoning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from finance_reasoning.engine.langgraph_workflow import FinanceReasoningWorkflow, MarketContext

logger = logging.getLogger(__name__)

class FinanceReasoningService:
    """
    Service for finance reasoning using LangGraph workflow

    Features:
    - Market analysis and reasoning
    - Risk assessment
    - Trade decision explanation
    - Audit trail and monitoring
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.service_id = config.get("service_id", "finance_reasoning_service")

        # Initialize workflow
        self.workflow = FinanceReasoningWorkflow(config)

        # Performance tracking
        self.analyses_performed = 0
        self.average_confidence = 0.0
        self.error_count = 0

        logger.info(f"Finance Reasoning Service {self.service_id} initialized")

    async def analyze_market_conditions(self,
                                      symbol: str,
                                      current_price: float,
                                      technical_signals: Optional[Dict[str, Any]] = None,
                                      market_data: Optional[Dict[str, Any]] = None,
                                      portfolio_data: Optional[Dict[str, Any]] = None,
                                      risk_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze market conditions and provide reasoned assessment

        Args:
            symbol: Trading symbol
            current_price: Current market price
            technical_signals: Technical analysis signals
            market_data: General market data
            portfolio_data: Portfolio context
            risk_metrics: Risk assessment metrics

        Returns:
            Analysis result with reasoning and audit trail
        """
        try:
            # Create market context
            market_context = MarketContext(
                symbol=symbol,
                current_price=current_price,
                technical_signals=technical_signals or {},
                market_data=market_data or {},
                portfolio_data=portfolio_data,
                risk_metrics=risk_metrics
            )

            # Add risk flags if risk metrics provided
            if risk_metrics:
                market_context.stop_loss_hit = risk_metrics.get("stop_loss_hit", False)
                market_context.volatility_spike = risk_metrics.get("volatility_spike", False)
                market_context.drawdown_limit_hit = risk_metrics.get("drawdown_limit_hit", False)

            # Execute workflow
            result = await self.workflow.reason_about_market(market_context)

            # Update performance metrics
            self.analyses_performed += 1
            self.average_confidence = (
                (self.average_confidence * (self.analyses_performed - 1)) +
                result["analysis"]["confidence_score"]
            ) / self.analyses_performed

            if result["status"] in ["failed", "critical_failure"]:
                self.error_count += 1

            logger.info(f"Market analysis completed for {symbol}: {result['status']}")
            return result

        except Exception as e:
            logger.error(f"Market analysis failed for {symbol}: {e}")
            self.error_count += 1

            return {
                "session_id": f"error_{int(datetime.now().timestamp())}",
                "symbol": symbol,
                "analysis": {
                    "raw_triggers": 0,
                    "prioritized_triggers": 0,
                    "resolved_contracts": 0,
                    "confidence_score": 0.0,
                    "explanation": f"Analysis failed: {str(e)}"
                },
                "status": "service_error",
                "errors": [str(e)],
                "audit_trail": [{
                    "timestamp": datetime.now().isoformat(),
                    "action": "analyze_market_conditions",
                    "status": "error",
                    "details": str(e)
                }],
                "timestamp": datetime.now().isoformat()
            }

    async def assess_trade_risk(self,
                               symbol: str,
                               entry_price: float,
                               stop_loss: float,
                               target_price: float,
                               position_size: float,
                               portfolio_value: float) -> Dict[str, Any]:
        """
        Assess risk for a potential trade

        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            target_price: Target price
            position_size: Position size
            portfolio_value: Total portfolio value

        Returns:
            Risk assessment with reasoning
        """
        try:
            # Calculate risk metrics
            risk_amount = abs(entry_price - stop_loss) * position_size
            risk_percentage = risk_amount / portfolio_value
            reward_amount = abs(target_price - entry_price) * position_size
            reward_percentage = reward_amount / portfolio_value
            risk_reward_ratio = reward_percentage / risk_percentage if risk_percentage > 0 else 0

            risk_metrics = {
                "risk_amount": risk_amount,
                "risk_percentage": risk_percentage,
                "reward_amount": reward_amount,
                "reward_percentage": reward_percentage,
                "risk_reward_ratio": risk_reward_ratio,
                "stop_loss_hit": False,
                "volatility_spike": risk_percentage > 0.05,  # 5% risk threshold
                "drawdown_limit_hit": risk_percentage > 0.10  # 10% risk threshold
            }

            # Analyze with risk context
            result = await self.analyze_market_conditions(
                symbol=symbol,
                current_price=entry_price,
                risk_metrics=risk_metrics
            )

            # Enhance result with risk-specific insights
            result["risk_assessment"] = {
                "risk_amount": risk_amount,
                "risk_percentage": risk_percentage,
                "risk_reward_ratio": risk_reward_ratio,
                "risk_level": "HIGH" if risk_percentage > 0.05 else "MEDIUM" if risk_percentage > 0.02 else "LOW",
                "recommended": risk_reward_ratio >= 2.0 and risk_percentage <= 0.05
            }

            return result

        except Exception as e:
            logger.error(f"Risk assessment failed for {symbol}: {e}")
            return {
                "session_id": f"error_{int(datetime.now().timestamp())}",
                "symbol": symbol,
                "analysis": {
                    "raw_triggers": 0,
                    "prioritized_triggers": 0,
                    "resolved_contracts": 0,
                    "confidence_score": 0.0,
                    "explanation": f"Risk assessment failed: {str(e)}"
                },
                "status": "risk_assessment_error",
                "errors": [str(e)],
                "risk_assessment": {
                    "risk_level": "UNKNOWN",
                    "recommended": False
                },
                "audit_trail": [{
                    "timestamp": datetime.now().isoformat(),
                    "action": "assess_trade_risk",
                    "status": "error",
                    "details": str(e)
                }],
                "timestamp": datetime.now().isoformat()
            }

    async def explain_portfolio_decision(self,
                                       holdings: Dict[str, float],
                                       target_allocation: Dict[str, float],
                                       market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain portfolio rebalancing decisions

        Args:
            holdings: Current portfolio holdings {symbol: weight}
            target_allocation: Target allocation {symbol: weight}
            market_conditions: Current market conditions

        Returns:
            Portfolio decision explanation
        """
        try:
            # Calculate current vs target differences
            rebalancing_needed = {}
            total_deviation = 0.0

            for symbol in set(holdings.keys()) | set(target_allocation.keys()):
                current = holdings.get(symbol, 0.0)
                target = target_allocation.get(symbol, 0.0)
                deviation = abs(current - target)
                rebalancing_needed[symbol] = {
                    "current": current,
                    "target": target,
                    "deviation": deviation
                }
                total_deviation += deviation

            # Create portfolio context
            portfolio_data = {
                "holdings": holdings,
                "target_allocation": target_allocation,
                "rebalancing_needed": rebalancing_needed,
                "total_deviation": total_deviation,
                "needs_rebalancing": total_deviation > 0.05  # 5% threshold
            }

            # Use first symbol for analysis (or create a portfolio symbol)
            primary_symbol = list(holdings.keys())[0] if holdings else "PORTFOLIO"

            result = await self.analyze_market_conditions(
                symbol=primary_symbol,
                current_price=0.0,  # Not applicable for portfolio
                portfolio_data=portfolio_data,
                market_data=market_conditions
            )

            # Enhance with portfolio-specific insights
            result["portfolio_decision"] = {
                "needs_rebalancing": total_deviation > 0.05,
                "total_deviation": total_deviation,
                "rebalancing_actions": rebalancing_needed,
                "recommendation": "REBALANCE" if total_deviation > 0.05 else "HOLD"
            }

            return result

        except Exception as e:
            logger.error(f"Portfolio decision explanation failed: {e}")
            return {
                "session_id": f"error_{int(datetime.now().timestamp())}",
                "symbol": "PORTFOLIO",
                "analysis": {
                    "raw_triggers": 0,
                    "prioritized_triggers": 0,
                    "resolved_contracts": 0,
                    "confidence_score": 0.0,
                    "explanation": f"Portfolio analysis failed: {str(e)}"
                },
                "status": "portfolio_error",
                "errors": [str(e)],
                "portfolio_decision": {
                    "needs_rebalancing": False,
                    "recommendation": "UNKNOWN"
                },
                "audit_trail": [{
                    "timestamp": datetime.now().isoformat(),
                    "action": "explain_portfolio_decision",
                    "status": "error",
                    "details": str(e)
                }],
                "timestamp": datetime.now().isoformat()
            }

    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and performance metrics"""
        return {
            "service_id": self.service_id,
            "analyses_performed": self.analyses_performed,
            "average_confidence": round(self.average_confidence, 3),
            "error_count": self.error_count,
            "error_rate": round(self.error_count / max(self.analyses_performed, 1), 3),
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }