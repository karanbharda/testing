from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum
import random
import math

from backend.hft.shadow_execution.fee_model import FeeModel
from backend.hft.models.trade_event import FeeBreakdown, TradeType, RiskStopReason, TradeArtifact
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.core.karma import KarmaLog
from backend.hft.core.state_machine import TradeStateMachine, TradeState
from backend.hft.config import default_config, ExecutionMode

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"

@dataclass(frozen=True)
class ShadowOrder:
    order_id: str
    timestamp: datetime
    symbol: str
    side: Side
    quantity: float
    limit_price: Optional[float]
    status: OrderStatus
    trade_type: TradeType = TradeType.EQUITY_INTRADAY

@dataclass(frozen=True)
class ShadowFill:
    fill_id: str
    order_id: str
    timestamp: datetime
    price: float
    quantity: float
    fee_breakdown: FeeBreakdown
    liquidity_flag: str # 'MAKER' or 'TAKER'

@dataclass
class ShadowPosition:
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees: float = 0.0

@dataclass
class SimulationAuditTrail:
    session_id: str
    start_time: datetime
    orders: List[ShadowOrder] = field(default_factory=list)
    fills: List[ShadowFill] = field(default_factory=list)
    trade_artifacts: List[TradeArtifact] = field(default_factory=list)

class ShadowSimulator:
    """
    Simulates order execution without broker connection.
    Tracks shadow fills, PnL, and enforces strict audit trails.
    """
    def __init__(self, risk_gate: RiskGate, karma_log: Optional[KarmaLog] = None):
        if default_config.system.execution_mode != ExecutionMode.SHADOW_ONLY:
            raise SystemExit("CRITICAL: ShadowSimulator started in LIVE mode. SHUTTING DOWN.")
            
        self.audit_trail = SimulationAuditTrail(
            session_id=datetime.now().isoformat(),
            start_time=datetime.now()
        )
        self.positions: Dict[str, ShadowPosition] = {}
        self.fee_model = FeeModel()
        self.risk_gate = risk_gate
        self.karma_log = karma_log
        self.current_regime = VolatilityRegime.NORMAL

    def set_regime(self, regime: VolatilityRegime):
        self.current_regime = regime

    def _validate_inputs(self, order: ShadowOrder, current_market_price: float, spread_bps: float) -> None:
        """
        Validates all inputs to prevent corrupt calculations.
        Raises ValueError with descriptive message if invalid.
        """
        # Validate price
        if not isinstance(current_market_price, (int, float)):
            raise ValueError(f"Invalid price type: {type(current_market_price).__name__}. Must be numeric.")
        if math.isnan(current_market_price):
            raise ValueError("Invalid price: NaN not allowed")
        if math.isinf(current_market_price):
            raise ValueError(f"Invalid price: Infinity not allowed (got {current_market_price})")
        if current_market_price <= 0:
            raise ValueError(f"Invalid price: must be > 0 (got {current_market_price})")
        
        # Validate quantity
        if not isinstance(order.quantity, (int, float)):
            raise ValueError(f"Invalid quantity type: {type(order.quantity).__name__}. Must be numeric.")
        if math.isnan(order.quantity):
            raise ValueError("Invalid quantity: NaN not allowed")
        if math.isinf(order.quantity):
            raise ValueError(f"Invalid quantity: Infinity not allowed (got {order.quantity})")
        if order.quantity <= 0:
            raise ValueError(f"Invalid quantity: must be > 0 (got {order.quantity})")
        
        # Validate spread
        if not isinstance(spread_bps, (int, float)):
            raise ValueError(f"Invalid spread type: {type(spread_bps).__name__}. Must be numeric.")
        if math.isnan(spread_bps):
            raise ValueError("Invalid spread: NaN not allowed")
        if math.isinf(spread_bps):
            raise ValueError(f"Invalid spread: Infinity not allowed (got {spread_bps})")
        if spread_bps < 0:
            raise ValueError(f"Invalid spread: must be >= 0 (got {spread_bps})")

    def place_order(self, order: ShadowOrder, current_market_price: float, spread_bps: float = 2.0, liquidity_snapshot: Dict = None) -> str:
        """
        Attempts to place a shadow order with full State Machine enforcement.
        """
        sm = TradeStateMachine()
        
        # TRANSITION: INIT -> SIGNALLED
        sm.transition(TradeState.SIGNALLED)
        
        # 0. INPUT VALIDATION (CRITICAL)
        try:
            self._validate_inputs(order, current_market_price, spread_bps)
        except ValueError as e:
            sm.transition(TradeState.REJECTED)
            if self.karma_log:
                self.karma_log.append("INPUT_VALIDATION_REJECT", {"order_id": order.order_id, "reason": str(e)})
            sm.transition(TradeState.LOGGED)
            return f"REJECTED: {str(e)}"
        
        # 1. Mode Guard (Redundant but critical)
        if default_config.system.execution_mode != ExecutionMode.SHADOW_ONLY:
            sm.transition(TradeState.REJECTED)
            sm.transition(TradeState.LOGGED)
            return "REJECTED_MODE_VIOLATION"

        # 2. Risk Check
        allowed, reason = self.risk_gate.check_risk(self.current_regime)
        if not allowed:
            sm.transition(TradeState.REJECTED)
            reason_str = reason.value if reason else "UNKNOWN"
            if self.karma_log:
                self.karma_log.append("RISK_REJECT", {"order_id": order.order_id, "reason": reason_str})
            sm.transition(TradeState.LOGGED)
            return f"REJECTED: {reason_str}"

        self.audit_trail.orders.append(order)
        if self.karma_log:
            self.karma_log.append("ORDER_PLACED", {"order_id": order.order_id, "symbol": order.symbol, "qty": order.quantity})

        # TRANSITION: SIGNALLED -> SUBMITTED
        sm.transition(TradeState.SUBMITTED)
        
        # 3. Liquidity & Slippage Model
        # Simulate partial fills chunks
        remaining_qty = order.quantity
        filled_qty = 0.0
        
        # Break into 1-3 chunks usually
        chunks = self._calculate_chunks(order.quantity, self.current_regime)
        
        avg_exec_price = 0.0
        
        for i, chunk_qty in enumerate(chunks):
            # TRANSITION: SUBMITTED -> FILLED_PARTIAL (or FULL if last)
            if i == len(chunks) - 1:
                sm.transition(TradeState.FILLED_FULL)
            else:
                 sm.transition(TradeState.FILLED_PARTIAL)
            
            # Drift price per chunk based on impact
            chunk_price = self._apply_liquidity_slippage(
                order.side, current_market_price, spread_bps, chunk_qty, i
            )
            
            self._fill_chunk(order, chunk_price, chunk_qty, sm)
            
            avg_exec_price = ((avg_exec_price * filled_qty) + (chunk_price * chunk_qty)) / (filled_qty + chunk_qty)
            filled_qty += chunk_qty
            
        # TRANSITION: FILLED -> EXITED (Assuming instant fill logic for shadow means we are done with the 'execution' phase)
        # In a real async system this differs, but here 'place_order' completes the lifecycle.
        sm.transition(TradeState.EXITED)
        
        # Log final artifact
        if self.karma_log:
             self.karma_log.append("ORDER_COMPLETE", {"order_id": order.order_id, "avg_price": avg_exec_price, "filled": filled_qty})
        
        sm.transition(TradeState.LOGGED)
        return "FILLED"

    def _calculate_chunks(self, total_qty: float, regime: VolatilityRegime) -> List[float]:
        """Break order into liquidity chunks."""
        if total_qty < 10: return [total_qty]
        
        # In high volatility, liquidity is fragmented -> more chunks
        num_chunks = 1
        if regime == VolatilityRegime.HIGH: num_chunks = random.randint(2, 4)
        elif regime == VolatilityRegime.EXTREME: num_chunks = random.randint(3, 6)
        
        chunks = []
        rem = total_qty
        for _ in range(num_chunks - 1):
            take = int(rem * random.uniform(0.3, 0.6))
            chunks.append(take)
            rem -= take
        chunks.append(rem)
        return chunks

    def _apply_liquidity_slippage(self, side: Side, base_price: float, spread_bps: float, chunk_qty: float, chunk_idx: int) -> float:
        """
        slippage = f(volume, spread, volatility)
        Later chunks get worse prices (implementation shortfall).
        """
        # Base impact: half spread
        impact = spread_bps / 2.0
        
        # Volatility Impact
        vol_factor = 1.0
        if self.current_regime == VolatilityRegime.HIGH: vol_factor = 2.0
        elif self.current_regime == VolatilityRegime.EXTREME: vol_factor = 5.0
        
        # Depth Impact: Later chunks eat more liquidity
        depth_penalty = (chunk_idx * 1.5) * vol_factor # bps
        
        total_slippage_bps = impact * vol_factor + depth_penalty + random.uniform(0, 1.0)
        
        factor = total_slippage_bps / 10000.0
        if side == Side.BUY:
            return base_price * (1 + factor)
        else:
            return base_price * (1 - factor)

    def _fill_chunk(self, order: ShadowOrder, price: float, qty: float, sm: TradeStateMachine):
        self.risk_gate.record_trade()
        fees = self.fee_model.calculate_fees(price, qty, order.side.value, order.trade_type)
        
        fill = ShadowFill(
            fill_id=f"F{len(self.audit_trail.fills):06d}",
            order_id=order.order_id,
            timestamp=datetime.now(),
            price=price,
            quantity=qty,
            fee_breakdown=fees,
            liquidity_flag="TAKER"
        )
        self.audit_trail.fills.append(fill)
        
        # Create Immutable Trade Artifact
        pnl_data = self._update_position(order.symbol, order.side, qty, price, fees)
        
        artifact = TradeArtifact(
            trade_id=fill.fill_id,
            instrument_type=order.trade_type,
            symbol=order.symbol,
            side=order.side.value,
            quantity=qty,
            entry_price=pnl_data['entry_price'], # Might be avg price of open position
            exit_price=price,
            gross_pnl=pnl_data['gross_pnl'],
            fees=fees,
            net_pnl=pnl_data['net_pnl']
        )
        
        self.audit_trail.trade_artifacts.append(artifact)
        
        if self.karma_log:
            self.karma_log.append("TRADE_ARTIFACT", {
                "trade_id": artifact.trade_id,
                "symbol": artifact.symbol,
                "net_pnl": artifact.net_pnl,
                "fees": artifact.fees.total_tax_and_fees,
                "state": sm.current_state.value
            })

    def _update_position(self, symbol: str, side: Side, qty: float, price: float, fees: FeeBreakdown) -> Dict:
        """
        Updates position and returns PnL context for the artifact.
        """
        pos = self.positions.get(symbol, ShadowPosition(symbol=symbol))
        
        pnl_context = {
            "entry_price": price, # Default if opening
            "gross_pnl": 0.0,
            "net_pnl": -fees.total_tax_and_fees # Initial debit
        }
        
        if side == Side.BUY:
            if pos.quantity < 0: # Closing Short
                # Covering
                cover_qty = min(abs(pos.quantity), qty)
                gross_pnl = (pos.average_price - price) * cover_qty
                
                # Check 0 division just in case
                pnl_context['entry_price'] = pos.average_price
                pnl_context['gross_pnl'] = gross_pnl
                pnl_context['net_pnl'] = gross_pnl - fees.total_tax_and_fees
                
                pos.realized_pnl += gross_pnl
                remaining = qty - cover_qty
                
                # Update pos
                pos.quantity += qty # Moves towards 0 or positive
                # If flipped to long, avg price needs reset?
                # Simple logic for now: weighted avg not perfect on flip, but okay for shadow
                if pos.quantity > 0: pos.average_price = price 
                
            else: # Adding Long
                new_qty = pos.quantity + qty
                pos.average_price = ((pos.quantity * pos.average_price) + (qty * price)) / new_qty if new_qty > 0 else price
                pos.quantity = new_qty
                pnl_context['entry_price'] = price
                pnl_context['gross_pnl'] = 0.0
                pnl_context['net_pnl'] = -fees.total_tax_and_fees
                
        else: # SELL
            if pos.quantity > 0: # Closing Long
                sell_qty = min(pos.quantity, qty)
                gross_pnl = (price - pos.average_price) * sell_qty
                pnl_context['entry_price'] = pos.average_price
                pnl_context['gross_pnl'] = gross_pnl
                pnl_context['net_pnl'] = gross_pnl - fees.total_tax_and_fees
                
                pos.realized_pnl += gross_pnl
                
                pos.quantity -= qty
                if pos.quantity < 0: pos.average_price = price 
            else: # Shorting
                new_qty = pos.quantity - qty
                # Weighted avg for shorts
                # Ensure we handle division by zero
                abs_new = abs(new_qty)
                pos.average_price = ((abs(pos.quantity) * pos.average_price) + (qty * price)) / abs_new if abs_new > 0 else price
                pos.quantity = new_qty
                pnl_context['entry_price'] = price
                pnl_context['gross_pnl'] = 0.0
                pnl_context['net_pnl'] = -fees.total_tax_and_fees

        # Record Loss logic if needed
        if pnl_context['net_pnl'] < 0 and abs(pnl_context['gross_pnl']) > 0:
             # Only record loss on closing trades
             self.risk_gate.record_loss(abs(pnl_context['net_pnl']))

        pos.total_fees += fees.total_tax_and_fees
        self.positions[symbol] = pos
        return pnl_context
