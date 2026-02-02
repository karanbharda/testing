"""
Finance Reasoning Engine
=======================

LangGraph-integrated financial reasoning system.

Architecture:
- LangGraph controls flow and state
- Finance Engine controls business logic
- LLM only narrates explanations
- Failures are contained
- State is explicit and auditable
"""

from .engine.langgraph_workflow import FinanceReasoningWorkflow, MarketContext
from .finance_reasoning_service import FinanceReasoningService
from .langgraph_integration import (
    LangGraphFinanceReasoningIntegration,
    get_langgraph_finance_reasoning_integration,
    analyze_market_conditions,
    assess_trade_risk,
    explain_portfolio_decision
)
from .engine.contract_loader import load_contracts, get_contract_by_id, get_contracts_by_domain
from .engine.rule_evaluation import evaluate_all_rules, apply_priority
from .engine.contract_resolver import resolve_contracts
from .engine.explanation_builder import build_explanation, compose_explanation

__all__ = [
    "FinanceReasoningWorkflow",
    "FinanceReasoningService",
    "LangGraphFinanceReasoningIntegration",
    "get_langgraph_finance_reasoning_integration",
    "analyze_market_conditions",
    "assess_trade_risk",
    "explain_portfolio_decision",
    "MarketContext",
    "load_contracts",
    "get_contract_by_id",
    "get_contracts_by_domain",
    "evaluate_all_rules",
    "apply_priority",
    "resolve_contracts",
    "build_explanation",
    "compose_explanation"
]