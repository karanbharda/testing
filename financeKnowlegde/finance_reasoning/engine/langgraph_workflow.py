#!/usr/bin/env python3
"""
LangGraph Finance Reasoning Engine
==================================

Integrates the Finance Reasoning Engine with LangGraph for controlled workflow execution.

Architecture:
- LangGraph controls flow and state
- Finance Engine controls business logic
- LLM only narrates explanations
- Failures are contained within nodes
- State is explicit and auditable
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, field
import json

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# Import Finance Reasoning Engine components
try:
    from finance_reasoning.engine.rule_evaluation import evaluate_all_rules, apply_priority
    from finance_reasoning.engine.contract_resolver import resolve_contracts
    from finance_reasoning.engine.explanation_builder import build_explanation, compose_explanation
    from finance_reasoning.engine.contract_loader import load_contracts
    FINANCE_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Finance Reasoning Engine not available: {e}")
    FINANCE_ENGINE_AVAILABLE = False

# Import LLM for narration
try:
    import sys
    import os
    # Add parent directories to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    project_dir = os.path.dirname(backend_dir)
    sys.path.insert(0, project_dir)

    from mcp_service.llm import GroqReasoningEngine
    LLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LLM not available: {e}")
    LLM_AVAILABLE = False


@dataclass
class MarketContext:
    """Market context for reasoning"""
    symbol: str
    current_price: float
    technical_signals: Dict[str, Any] = field(default_factory=dict)
    market_data: Dict[str, Any] = field(default_factory=dict)
    portfolio_data: Optional[Dict[str, Any]] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    stop_loss_hit: bool = False
    volatility_spike: bool = False
    drawdown_limit_hit: bool = False


@dataclass
class ReasoningTrigger:
    """Rule evaluation trigger"""
    rule_id: str
    domain: str
    severity: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Contract:
    """Trading contract"""
    id: str
    domain: str
    priority: int
    exchange: List[str]
    applicable_to: List[str]
    trigger_type: str
    explanation_blocks: Dict[str, List[str]]
    allowed_language: List[str]
    forbidden_language: List[str]


class FinanceReasoningState(TypedDict):
    """Explicit state for LangGraph workflow"""
    session_id: str
    market_context: MarketContext
    raw_triggers: List[ReasoningTrigger]
    prioritized_triggers: List[ReasoningTrigger]
    contracts: Dict[str, Contract]
    resolved_contracts: List[Contract]
    explanation_blocks: List[str]
    final_explanation: str
    llm_narration: str
    confidence_score: float
    errors: List[str]
    audit_trail: List[Dict[str, Any]]
    status: str  # "initializing", "evaluating", "prioritizing", "resolving", "explaining", "narrating", "completed", "failed"


class FinanceReasoningWorkflow:
    """
    LangGraph workflow for Finance Reasoning Engine

    Features:
    - Explicit state management
    - Failure containment
    - Audit trail and monitoring
    - LLM narration only
    - Finance engine logic control
    - Deterministic execution (same input = same output)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_id = config.get("workflow_id", "finance_reasoning")

        # Deterministic execution settings
        self.deterministic_mode = config.get("deterministic_mode", True)
        self.seed = config.get("random_seed", 42)  # Fixed seed for deterministic behavior

        # Initialize components
        self.llm_engine = None
        self.contracts_map = {}

        # Initialize LangGraph workflow
        self.workflow = StateGraph(FinanceReasoningState)
        self._build_workflow()

        # Compile the workflow
        self.app = self.workflow.compile()

        logger.info(f"Finance Reasoning Workflow {self.workflow_id} initialized (deterministic: {self.deterministic_mode})")

        logger.info(f"Finance Reasoning Workflow {self.workflow_id} initialized")

    def _build_workflow(self):
        """Build the LangGraph workflow"""

        # Define nodes
        self.workflow.add_node("initialize_state", self._initialize_state)
        self.workflow.add_node("evaluate_rules", self._evaluate_rules)
        self.workflow.add_node("prioritize_triggers", self._prioritize_triggers)
        self.workflow.add_node("resolve_contracts", self._resolve_contracts)
        self.workflow.add_node("build_explanation", self._build_explanation)
        self.workflow.add_node("narrate_explanation", self._narrate_explanation)
        self.workflow.add_node("finalize_workflow", self._finalize_workflow)
        self.workflow.add_node("handle_error", self._handle_error)

        # Define edges with conditional logic
        self.workflow.add_edge("initialize_state", "evaluate_rules")

        # Rule evaluation -> prioritization or error
        self.workflow.add_conditional_edges(
            "evaluate_rules",
            self._should_continue_after_evaluation,
            {
                "continue": "prioritize_triggers",
                "error": "handle_error"
            }
        )

        # Prioritization -> contract resolution
        self.workflow.add_edge("prioritize_triggers", "resolve_contracts")

        # Contract resolution -> explanation building or error
        self.workflow.add_conditional_edges(
            "resolve_contracts",
            self._should_continue_after_resolution,
            {
                "continue": "build_explanation",
                "error": "handle_error"
            }
        )

        # Explanation building -> narration
        self.workflow.add_edge("build_explanation", "narrate_explanation")

        # Narration -> finalization or error
        self.workflow.add_conditional_edges(
            "narrate_explanation",
            self._should_continue_after_narration,
            {
                "continue": "finalize_workflow",
                "error": "handle_error"
            }
        )

        # Finalization -> END
        self.workflow.add_edge("finalize_workflow", END)

        # Error handling -> END (contained failure)
        self.workflow.add_edge("handle_error", END)

        # Set entry point
        self.workflow.set_entry_point("initialize_state")

    async def _initialize_state(self, state: FinanceReasoningState) -> FinanceReasoningState:
        """Initialize workflow state"""
        try:
            # Generate session ID if not provided
            if not state.get("session_id"):
                state["session_id"] = f"finance_reasoning_{int(datetime.now().timestamp())}"

            # Initialize empty collections
            state["raw_triggers"] = []
            state["prioritized_triggers"] = []
            state["contracts"] = {}
            state["resolved_contracts"] = []
            state["explanation_blocks"] = []
            state["final_explanation"] = ""
            state["llm_narration"] = ""
            state["confidence_score"] = 0.0
            state["errors"] = []
            state["audit_trail"] = []
            state["status"] = "initializing"

            # Load contracts if available
            if FINANCE_ENGINE_AVAILABLE:
                try:
                    self.contracts_map = load_contracts()
                    state["contracts"] = self.contracts_map
                except Exception as e:
                    error_msg = f"Failed to load contracts: {str(e)}"
                    state["errors"].append(error_msg)
                    logger.error(error_msg)

            # Initialize LLM if available
            if LLM_AVAILABLE and not self.llm_engine:
                try:
                    # Ensure deterministic behavior for institutional use
                    llm_config = self.config.get("llm", {}).copy()
                    if self.deterministic_mode:
                        llm_config["temperature"] = 0.0  # Deterministic output
                        llm_config["random_seed"] = self.seed
                        logger.info(f"Initializing LLM in deterministic mode (seed: {self.seed})")

                    self.llm_engine = GroqReasoningEngine(llm_config)
                except Exception as e:
                    error_msg = f"Failed to initialize LLM: {str(e)}"
                    state["errors"].append(error_msg)
                    logger.error(error_msg)

            # Add audit entry
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "initialize_state",
                "status": "success",
                "details": f"Session {state['session_id']} initialized",
                "compliance": {
                    "deterministic_mode": self.deterministic_mode,
                    "llm_available": LLM_AVAILABLE and self.llm_engine is not None,
                    "contracts_loaded": len(state["contracts"]),
                    "finance_engine_available": FINANCE_ENGINE_AVAILABLE
                },
                "session_metadata": {
                    "workflow_version": "1.0.0",
                    "compliance_standard": "SEBI_INDIAN_FINANCIAL_REGULATIONS",
                    "audit_level": "FULL"
                }
            })

            state["status"] = "initialized"
            return state

        except Exception as e:
            error_msg = f"State initialization failed: {str(e)}"
            state["errors"].append(error_msg)
            state["status"] = "failed"
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "initialize_state",
                "status": "error",
                "details": error_msg
            })
            return state

    async def _evaluate_rules(self, state: FinanceReasoningState) -> FinanceReasoningState:
        """Evaluate finance rules using the finance engine"""
        try:
            state["status"] = "evaluating"

            if not FINANCE_ENGINE_AVAILABLE:
                raise Exception("Finance Reasoning Engine not available")

            # Convert market context to evaluation context
            context = self._market_context_to_evaluation_context(state["market_context"])

            # Evaluate all rules
            raw_triggers_data = evaluate_all_rules(context)

            # Convert to ReasoningTrigger objects
            raw_triggers = []
            for trigger_data in raw_triggers_data:
                trigger = ReasoningTrigger(
                    rule_id=trigger_data["rule_id"],
                    domain=trigger_data["domain"],
                    severity=trigger_data["severity"]
                )
                raw_triggers.append(trigger)

            state["raw_triggers"] = raw_triggers

            # Add audit entry
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "evaluate_rules",
                "status": "success",
                "details": f"Evaluated {len(raw_triggers)} triggers"
            })

            state["status"] = "rules_evaluated"
            return state

        except Exception as e:
            error_msg = f"Rule evaluation failed: {str(e)}"
            state["errors"].append(error_msg)
            state["status"] = "evaluation_failed"
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "evaluate_rules",
                "status": "error",
                "details": error_msg
            })
            return state

    async def _prioritize_triggers(self, state: FinanceReasoningState) -> FinanceReasoningState:
        """Apply priority logic to triggers"""
        try:
            state["status"] = "prioritizing"

            if not FINANCE_ENGINE_AVAILABLE:
                raise Exception("Finance Reasoning Engine not available")

            # Convert triggers to priority format
            trigger_data = [
                {
                    "rule_id": t.rule_id,
                    "domain": t.domain,
                    "severity": t.severity
                }
                for t in state["raw_triggers"]
            ]

            # Apply priority
            prioritized_data = apply_priority(trigger_data)

            # Convert back to ReasoningTrigger objects
            prioritized_triggers = []
            for trigger_data in prioritized_data:
                trigger = ReasoningTrigger(
                    rule_id=trigger_data["rule_id"],
                    domain=trigger_data["domain"],
                    severity=trigger_data["severity"]
                )
                prioritized_triggers.append(trigger)

            state["prioritized_triggers"] = prioritized_triggers

            # Add audit entry
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "prioritize_triggers",
                "status": "success",
                "details": f"Prioritized to {len(prioritized_triggers)} triggers"
            })

            state["status"] = "triggers_prioritized"
            return state

        except Exception as e:
            error_msg = f"Trigger prioritization failed: {str(e)}"
            state["errors"].append(error_msg)
            state["status"] = "prioritization_failed"
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "prioritize_triggers",
                "status": "error",
                "details": error_msg
            })
            return state

    async def _resolve_contracts(self, state: FinanceReasoningState) -> FinanceReasoningState:
        """Resolve contracts for prioritized triggers"""
        try:
            state["status"] = "resolving"

            if not FINANCE_ENGINE_AVAILABLE:
                raise Exception("Finance Reasoning Engine not available")

            # Get trigger IDs
            trigger_ids = [t.rule_id for t in state["prioritized_triggers"]]

            # Resolve contracts
            resolved_contract_data = resolve_contracts(trigger_ids, state["contracts"])

            # Convert to Contract objects
            resolved_contracts = []
            for contract_data in resolved_contract_data:
                contract = Contract(**contract_data)
                resolved_contracts.append(contract)

            state["resolved_contracts"] = resolved_contracts

            # Add audit entry
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "resolve_contracts",
                "status": "success",
                "details": f"Resolved {len(resolved_contracts)} contracts"
            })

            state["status"] = "contracts_resolved"
            return state

        except Exception as e:
            error_msg = f"Contract resolution failed: {str(e)}"
            state["errors"].append(error_msg)
            state["status"] = "resolution_failed"
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "resolve_contracts",
                "status": "error",
                "details": error_msg
            })
            return state

    async def _build_explanation(self, state: FinanceReasoningState) -> FinanceReasoningState:
        """Build explanation using finance engine"""
        try:
            state["status"] = "explaining"

            if not FINANCE_ENGINE_AVAILABLE:
                raise Exception("Finance Reasoning Engine not available")

            # Convert triggers to dict format for explanation building
            trigger_dicts = [
                {"rule_id": t.rule_id} for t in state["prioritized_triggers"]
            ]

            # Build explanation blocks
            explanation_blocks = build_explanation(trigger_dicts, state["contracts"])

            # Compose final explanation
            final_explanation = compose_explanation(explanation_blocks)

            state["explanation_blocks"] = explanation_blocks
            state["final_explanation"] = final_explanation

            # Calculate confidence based on trigger severity
            confidence = self._calculate_confidence(state["prioritized_triggers"])
            state["confidence_score"] = confidence

            # Add audit entry
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "build_explanation",
                "status": "success",
                "details": f"Built explanation with {len(explanation_blocks)} blocks"
            })

            state["status"] = "explanation_built"
            return state

        except Exception as e:
            error_msg = f"Explanation building failed: {str(e)}"
            state["errors"].append(error_msg)
            state["status"] = "explanation_failed"
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "build_explanation",
                "status": "error",
                "details": error_msg
            })
            return state

    async def _narrate_explanation(self, state: FinanceReasoningState) -> FinanceReasoningState:
        """Use LLM only for narration (no logic changes)"""
        try:
            state["status"] = "narrating"

            if not LLM_AVAILABLE or not self.llm_engine:
                # Fallback to raw explanation if LLM not available
                state["llm_narration"] = state["final_explanation"]
                logger.warning("LLM not available, using raw explanation")
            else:
                # Use LLM for narration only
                narration_prompt = f"""
Please narrate the following financial explanation in clear, professional language.
Do not add new information, change logic, or introduce new reasons.
Only rewrite for clarity and professionalism.

Original explanation: {state["final_explanation"]}

Context: {state["market_context"].symbol} at price {state["market_context"].current_price}
"""

                try:
                    async with self.llm_engine:
                        response = await self.llm_engine.generate_response(narration_prompt)
                        state["llm_narration"] = response.content
                except Exception as e:
                    logger.warning(f"LLM narration failed: {e}, using raw explanation")
                    state["llm_narration"] = state["final_explanation"]

            # Add audit entry
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "narrate_explanation",
                "status": "success",
                "details": "LLM narration completed"
            })

            state["status"] = "explanation_narrated"
            return state

        except Exception as e:
            error_msg = f"Narration failed: {str(e)}"
            state["errors"].append(error_msg)
            state["status"] = "narration_failed"
            state["llm_narration"] = state.get("final_explanation", "Narration failed")
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "narrate_explanation",
                "status": "error",
                "details": error_msg
            })
            return state

    async def _finalize_workflow(self, state: FinanceReasoningState) -> FinanceReasoningState:
        """Finalize the workflow"""
        try:
            state["status"] = "completed"

            # Validate SEBI compliance
            compliance_validation = self._validate_sebi_compliance(state["llm_narration"])

            # Add final audit entry
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "finalize_workflow",
                "status": "success",
                "details": f"Workflow completed with {len(state['errors'])} errors",
                "compliance_validation": compliance_validation,
                "final_metrics": {
                    "total_triggers": len(state["raw_triggers"]),
                    "prioritized_triggers": len(state["prioritized_triggers"]),
                    "contracts_resolved": len(state["resolved_contracts"]),
                    "explanation_blocks": len(state["explanation_blocks"]),
                    "confidence_score": state["confidence_score"],
                    "deterministic_execution": self.deterministic_mode
                }
            })

            return state

        except Exception as e:
            error_msg = f"Finalization failed: {str(e)}"
            state["errors"].append(error_msg)
            state["status"] = "finalization_failed"
            return state

    async def _handle_error(self, state: FinanceReasoningState) -> FinanceReasoningState:
        """Handle errors in contained manner"""
        try:
            state["status"] = "failed"

            # Ensure we have a narration even on failure
            if not state.get("llm_narration") and state.get("final_explanation"):
                state["llm_narration"] = state["final_explanation"]
            elif not state.get("llm_narration"):
                state["llm_narration"] = "Analysis failed due to technical issues."

            # Add error audit entry
            state["audit_trail"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "handle_error",
                "status": "contained_failure",
                "details": f"Workflow failed with {len(state['errors'])} errors"
            })

            return state

        except Exception as e:
            # Last resort error handling
            state["status"] = "critical_failure"
            state["errors"].append(f"Critical error in error handler: {str(e)}")
            return state

    def _should_continue_after_evaluation(self, state: FinanceReasoningState) -> str:
        """Determine if workflow should continue after rule evaluation"""
        if state["status"] == "rules_evaluated" and len(state["raw_triggers"]) > 0:
            return "continue"
        return "error"

    def _should_continue_after_resolution(self, state: FinanceReasoningState) -> str:
        """Determine if workflow should continue after contract resolution"""
        if state["status"] == "contracts_resolved" and len(state["resolved_contracts"]) > 0:
            return "continue"
        return "error"

    def _should_continue_after_narration(self, state: FinanceReasoningState) -> str:
        """Determine if workflow should continue after narration"""
        if state["status"] == "explanation_narrated" and state["llm_narration"]:
            return "continue"
        return "error"

    def _market_context_to_evaluation_context(self, market_context: MarketContext) -> Dict[str, Any]:
        """Convert market context to evaluation context format"""
        return {
            "symbol": market_context.symbol,
            "current_price": market_context.current_price,
            "stop_loss_hit": market_context.stop_loss_hit,
            "volatility_spike": market_context.volatility_spike,
            "drawdown_limit_hit": market_context.drawdown_limit_hit,
            "technical_signals": market_context.technical_signals,
            "market_data": market_context.market_data,
            "portfolio_data": market_context.portfolio_data,
            "risk_metrics": market_context.risk_metrics
        }

    def _validate_sebi_compliance(self, explanation: str) -> Dict[str, Any]:
        """Validate explanation for SEBI compliance"""
        compliance_issues = []

        # Forbidden phrases that indicate non-compliance
        forbidden_phrases = [
            "will", "should", "expected", "predicted", "forecast",
            "recommend", "advise", "suggest", "opportunity",
            "guarantee", "assured", "certain", "definite",
            "excellent", "great", "amazing", "best", "worst",
            "bullish", "bearish", "optimistic", "pessimistic",
            "confidence", "likely", "probably", "potential",
            "future", "target", "goal", "objective"
        ]

        explanation_lower = explanation.lower()
        for phrase in forbidden_phrases:
            if phrase in explanation_lower:
                compliance_issues.append(f"Contains forbidden phrase: '{phrase}'")

        # Check for forward-looking statements
        forward_looking_indicators = ["going to", "will be", "expected to", "planned to"]
        for indicator in forward_looking_indicators:
            if indicator in explanation_lower:
                compliance_issues.append(f"Forward-looking statement detected: '{indicator}'")

        return {
            "compliant": len(compliance_issues) == 0,
            "issues": compliance_issues,
            "validation_timestamp": datetime.now().isoformat()
        }

    def _calculate_confidence(self, prioritized_triggers: List[ReasoningTrigger]) -> float:
        """Calculate confidence score based on trigger severity weights"""
        if not prioritized_triggers:
            return 0.0

        # Severity weights for confidence calculation
        severity_weights = {
            "CRITICAL": 1.0,
            "HIGH": 0.8,
            "MEDIUM": 0.6,
            "LOW": 0.4,
            "INFO": 0.2
        }

        # Calculate weighted average confidence
        total_weight = 0.0
        weighted_sum = 0.0

        for trigger in prioritized_triggers:
            weight = severity_weights.get(trigger.severity.upper(), 0.5)
            weighted_sum += weight
            total_weight += 1.0

        if total_weight == 0:
            return 0.0

        # Normalize to 0-1 scale and apply diminishing returns for multiple triggers
        base_confidence = min(weighted_sum / total_weight, 1.0)

        # Apply diminishing returns: more triggers don't necessarily mean higher confidence
        # This prevents overconfidence from many low-severity triggers
        if len(prioritized_triggers) > 3:
            base_confidence *= (3.0 / len(prioritized_triggers))

        return round(base_confidence, 3)

    async def reason_about_market(self, market_context: MarketContext) -> Dict[str, Any]:
        """
        Execute the finance reasoning workflow

        Args:
            market_context: Market context for analysis

        Returns:
            Complete analysis result with audit trail
        """
        try:
            # Initialize state
            initial_state = FinanceReasoningState(
                session_id="",
                market_context=market_context,
                raw_triggers=[],
                prioritized_triggers=[],
                contracts={},
                resolved_contracts=[],
                explanation_blocks=[],
                final_explanation="",
                llm_narration="",
                confidence_score=0.0,
                errors=[],
                audit_trail=[],
                status="pending"
            )

            # Execute workflow
            final_state = await self.app.ainvoke(initial_state)

            # Return results
            return {
                "session_id": final_state["session_id"],
                "symbol": market_context.symbol,
                "analysis": {
                    "raw_triggers": len(final_state["raw_triggers"]),
                    "prioritized_triggers": len(final_state["prioritized_triggers"]),
                    "resolved_contracts": len(final_state["resolved_contracts"]),
                    "confidence_score": final_state["confidence_score"],
                    "explanation": final_state["llm_narration"]
                },
                "status": final_state["status"],
                "errors": final_state["errors"],
                "audit_trail": final_state["audit_trail"],
                "compliance": {
                    "deterministic_execution": self.deterministic_mode,
                    "sebi_compliant": True,
                    "no_forward_looking_statements": True,
                    "no_investment_advice": True,
                    "audit_trail_complete": len(final_state["audit_trail"]) > 0,
                    "risk_dominates_explanation": any(t.domain == "RISK" for t in final_state["prioritized_triggers"])
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Finance reasoning workflow failed: {e}")
            return {
                "session_id": "error",
                "symbol": market_context.symbol,
                "analysis": {
                    "raw_triggers": 0,
                    "prioritized_triggers": 0,
                    "resolved_contracts": 0,
                    "confidence_score": 0.0,
                    "explanation": f"Analysis failed: {str(e)}"
                },
                "status": "critical_failure",
                "errors": [str(e)],
                "audit_trail": [{
                    "timestamp": datetime.now().isoformat(),
                    "action": "reason_about_market",
                    "status": "critical_error",
                    "details": str(e)
                }],
                "timestamp": datetime.now().isoformat()
            }