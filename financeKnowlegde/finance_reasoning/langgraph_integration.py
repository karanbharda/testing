#!/usr/bin/env python3
"""
LangGraph Finance Reasoning Integration
======================================

Integration point for the LangGraph Finance Reasoning Engine in the main application.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class LangGraphFinanceReasoningIntegration:
    """
    Integration class for LangGraph Finance Reasoning Engine

    Provides a clean API for the main application to use the finance reasoning capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integration_id = config.get("integration_id", "langgraph_finance_reasoning")

        # Initialize the finance reasoning service
        try:
            # Add project root to path for backend imports
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))

            from finance_reasoning import FinanceReasoningService
            self.service = FinanceReasoningService(config.get("finance_reasoning", {}))
            self.available = True
            logger.info(f"LangGraph Finance Reasoning Integration {self.integration_id} initialized")
        except ImportError as e:
            logger.warning(f"Finance Reasoning Service not available: {e}")
            self.service = None
            self.available = False

    async def analyze_market_conditions(self,
                                      symbol: str,
                                      current_price: float,
                                      technical_signals: Optional[Dict[str, Any]] = None,
                                      market_data: Optional[Dict[str, Any]] = None,
                                      portfolio_data: Optional[Dict[str, Any]] = None,
                                      risk_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze market conditions using LangGraph workflow

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
        if not self.available or not self.service:
            return {
                "status": "unavailable",
                "explanation": "Finance Reasoning Engine not available",
                "confidence_score": 0.0,
                "audit_trail": []
            }

        try:
            return await self.service.analyze_market_conditions(
                symbol=symbol,
                current_price=current_price,
                technical_signals=technical_signals,
                market_data=market_data,
                portfolio_data=portfolio_data,
                risk_metrics=risk_metrics
            )
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {
                "status": "error",
                "explanation": f"Analysis failed: {str(e)}",
                "confidence_score": 0.0,
                "audit_trail": []
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
        if not self.available or not self.service:
            return {
                "status": "unavailable",
                "risk_assessment": {"risk_level": "UNKNOWN", "recommended": False}
            }

        try:
            return await self.service.assess_trade_risk(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                position_size=position_size,
                portfolio_value=portfolio_value
            )
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {
                "status": "error",
                "risk_assessment": {"risk_level": "UNKNOWN", "recommended": False}
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
        if not self.available or not self.service:
            return {
                "status": "unavailable",
                "portfolio_decision": {"needs_rebalancing": False, "recommendation": "UNKNOWN"}
            }

        try:
            return await self.service.explain_portfolio_decision(
                holdings=holdings,
                target_allocation=target_allocation,
                market_conditions=market_conditions
            )
        except Exception as e:
            logger.error(f"Portfolio decision explanation failed: {e}")
            return {
                "status": "error",
                "portfolio_decision": {"needs_rebalancing": False, "recommendation": "UNKNOWN"}
            }

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        if not self.available or not self.service:
            return {
                "integration_id": self.integration_id,
                "status": "unavailable",
                "service_status": None
            }

        return {
            "integration_id": self.integration_id,
            "status": "active",
            "service_status": self.service.get_service_status()
        }

# Global integration instance
_langgraph_integration = None

def get_langgraph_finance_reasoning_integration(config: Dict[str, Any]) -> LangGraphFinanceReasoningIntegration:
    """
    Get or create the global LangGraph Finance Reasoning integration instance

    Args:
        config: Configuration dictionary

    Returns:
        LangGraphFinanceReasoningIntegration instance
    """
    global _langgraph_integration
    if _langgraph_integration is None:
        _langgraph_integration = LangGraphFinanceReasoningIntegration(config)
    return _langgraph_integration

# Convenience functions for easy access
async def analyze_market_conditions(*args, **kwargs):
    """Convenience function for market analysis"""
    integration = get_langgraph_finance_reasoning_integration({})
    return await integration.analyze_market_conditions(*args, **kwargs)

async def assess_trade_risk(*args, **kwargs):
    """Convenience function for risk assessment"""
    integration = get_langgraph_finance_reasoning_integration({})
    return await integration.assess_trade_risk(*args, **kwargs)

async def explain_portfolio_decision(*args, **kwargs):
    """Convenience function for portfolio decisions"""
    integration = get_langgraph_finance_reasoning_integration({})
    return await integration.explain_portfolio_decision(*args, **kwargs)