from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum
import random

from backend.hft.shadow_execution.fee_model import FeeModel
from backend.hft.models.trade_event import FeeBreakdown, TradeType, RiskStopReason
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.reporting.karma import KarmaLogger

class Side(Enum):
    BUY = "BUY"
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
    """
    Full audit trail of the shadow session.
    """
    session_id: str
    start_time: datetime
    orders: List[ShadowOrder] = field(default_factory=list)
    fills: List[ShadowFill] = field(default_factory=list)
    pnl_snapshot: Dict[datetime, float] = field(default_factory=dict)

class ShadowSimulator:
    """
    Simulates order execution without broker connection.
    Tracks shadow fills and PnL.
    """
    def __init__(self, risk_gate: RiskGate, karma_logger: Optional[KarmaLogger] = None):
        self.audit_trail = SimulationAuditTrail(
            session_id=datetime.now().isoformat(),
            start_time=datetime.now()
        )
        self.positions: Dict[str, ShadowPosition] = {}
        self.fee_model = FeeModel()
        self.risk_gate = risk_gate
        self.karma_logger = karma_logger
        self.current_regime = VolatilityRegime.NORMAL

    def set_regime(self, regime: VolatilityRegime):
        self.current_regime = regime

    def place_order(self, order: ShadowOrder, current_market_price: float, spread_bps: float = 2.0) -> str:
        """
        Attempts to place a shadow order.
        Returns "OK" or Rejection Reason.
        """
        # 1. Risk Check
        allowed, reason = self.risk_gate.check_risk(self.current_regime)
        if not allowed:
            # reason is of type RiskStopReason
            reason_str = reason.value if reason else "UNKNOWN"
            if self.karma_logger:
                self.karma_logger.log_limit_check({"event": "REJECTED", "order_id": order.order_id, "reason": reason_str})
            return f"REJECTED: {reason_str}"

        self.audit_trail.orders.append(order)
        if self.karma_logger:
            self.karma_logger.log("orders", {"event": "PLACED", "order": str(order)})
        
        # 2. Simulate Execution
        
        # Slippage Model (Spread Aware)
        # Base slippage is half the spread (crossing the spread) + regime impact
        execution_price = self._apply_slippage(order, current_market_price, spread_bps)
        
        # Partial Fill Model
        fill_qty = self._calculate_fill_qty(order.quantity)
        
        if fill_qty > 0:
            self._fill_order(order, execution_price, fill_qty)
            return "FILLED" if fill_qty == order.quantity else "PARTIALLY_FILLED"
            
        return "OPEN" 

    def _apply_slippage(self, order: ShadowOrder, market_price: float, spread_bps: float) -> float:
        """
        Applies slippage.
        Slippage = (Spread / 2) + Unexplained Volatility Impact.
        """
        regime_impact = 0.0
        if self.current_regime == VolatilityRegime.HIGH:
            regime_impact = 2.0
        elif self.current_regime == VolatilityRegime.EXTREME:
            regime_impact = 10.0
            
        # Total slippage in BPS = Half Spread + Regime Impact + Random Jitter
        jitter = random.uniform(0, 1.0) # 0 to 1 bps jitter
        total_slippage_bps = (spread_bps / 2.0) + regime_impact + jitter
        
        slippage_factor = total_slippage_bps / 10000.0
        
        if order.side == Side.BUY:
            return market_price * (1 + slippage_factor)
        else:
            return market_price * (1 - slippage_factor)

    def _calculate_fill_qty(self, order_qty: float) -> float:
        """
        Simulates partial fills.
        """
        # In EXTREME regime, fill probability drops
        fill_prob = 0.9
        if self.current_regime == VolatilityRegime.EXTREME:
            fill_prob = 0.5
            
        if random.random() < fill_prob:
            return order_qty
        
        # Partial fill
        return int(order_qty * random.uniform(0.1, 0.9))

    def _fill_order(self, order: ShadowOrder, price: float, qty: float):
        self.risk_gate.record_trade()
        
        # Calculate Fees
        fees = self.fee_model.calculate_fees(price, qty, order.side.value, order.trade_type)
        
        # Create Fill
        fill = ShadowFill(
            fill_id=f"FILL_{len(self.audit_trail.fills)}",
            order_id=order.order_id,
            timestamp=datetime.now(),
            price=price,
            quantity=qty,
            fee_breakdown=fees,
            liquidity_flag="TAKER" 
        )
        self.audit_trail.fills.append(fill)
        
        if self.karma_logger:
            self.karma_logger.log_trade({
                "fill_id": fill.fill_id, 
                "price": price, 
                "qty": qty, 
                "fees": fees.total_tax_and_fees,
                "breakdown": str(fees)
            })
        
        # Update Position & PnL
        self._update_position(order.symbol, order.side, qty, price, fees)

    def _update_position(self, symbol: str, side: Side, qty: float, price: float, fees: FeeBreakdown):
        pos = self.positions.get(symbol, ShadowPosition(symbol=symbol))
        
        if side == Side.BUY:
            new_qty = pos.quantity + qty
            # Average Price calculation
            pos.average_price = ((pos.quantity * pos.average_price) + (qty * price)) / new_qty if new_qty > 0 else 0.0
            pos.quantity = new_qty
        else: # SELL
            # Realized PnL Calculation (FIFO/Average Cost)
            pnl = (price - pos.average_price) * qty
            pos.realized_pnl += pnl
            pos.quantity -= qty
            
            # Net PnL = Gross PnL - Fees (Fees are always positive cost)
            # Checking Loss for Risk Gate
            net_pnl_trade = pnl - fees.total_tax_and_fees
            if net_pnl_trade < 0:
                self.risk_gate.record_loss(abs(net_pnl_trade))

        pos.total_fees += fees.total_tax_and_fees
        self.positions[symbol] = pos


