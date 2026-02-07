from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum
import random

from backend.hft.shadow_execution.fee_model import FeeModel, FeeBreakdown
from backend.hft.risk.throttling import RiskMonitor, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.reporting.karma import KarmaLogger

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
    def __init__(self, risk_config: RiskConfig, karma_logger: Optional[KarmaLogger] = None):
        self.audit_trail = SimulationAuditTrail(
            session_id=datetime.now().isoformat(),
            start_time=datetime.now()
        )
        self.positions: Dict[str, ShadowPosition] = {}
        self.fee_model = FeeModel()
        self.risk_monitor = RiskMonitor(risk_config)
        self.karma_logger = karma_logger
        self.current_regime = VolatilityRegime.NORMAL

    def set_regime(self, regime: VolatilityRegime):
        self.current_regime = regime

    def place_order(self, order: ShadowOrder, current_market_price: float) -> str:
        """
        Attempts to place a shadow order.
        Returns "OK" or Rejection Reason.
        """
        # 1. Risk Check
        allowed, reason = self.risk_monitor.check_risk(self.current_regime)
        if not allowed:
            if self.karma_logger:
                self.karma_logger.log_limit_check({"event": "REJECTED", "order_id": order.order_id, "reason": reason})
            return f"REJECTED: {reason}"

        self.audit_trail.orders.append(order)
        if self.karma_logger:
            self.karma_logger.log("orders", {"event": "PLACED", "order": str(order)})
        
        # 2. Simulate Execution (Immediate simplified simulation for now)
        # In a real event loop, this would go into an order book. 
        # Here we simulate immediate fill possibilities.
        
        # Slippage Model
        execution_price = self._apply_slippage(order, current_market_price)
        
        # Partial Fill Model
        fill_qty = self._calculate_fill_qty(order.quantity)
        
        if fill_qty > 0:
            self._fill_order(order, execution_price, fill_qty)
            return "FILLED" if fill_qty == order.quantity else "PARTIALLY_FILLED"
            
        return "OPEN" # Placed but not filled (limit order far away)

    def _apply_slippage(self, order: ShadowOrder, market_price: float) -> float:
        """
        Applies slippage based on regime.
        """
        slippage_bps = 0.0
        if self.current_regime == VolatilityRegime.LOW:
            slippage_bps = 1.0 # 1 bps
        elif self.current_regime == VolatilityRegime.HIGH:
            slippage_bps = 5.0
        elif self.current_regime == VolatilityRegime.EXTREME:
            slippage_bps = 15.0
        else: # Normal
            slippage_bps = 2.0
            
        # Random jitter
        actual_slippage = slippage_bps * (0.5 + random.random()) # 0.5x to 1.5x of base
        
        slippage_factor = actual_slippage / 10000.0
        
        if order.side == Side.BUY:
            return market_price * (1 + slippage_factor)
        else:
            return market_price * (1 - slippage_factor)

    def _calculate_fill_qty(self, order_qty: float) -> float:
        """
        Simulates partial fills.
        """
        # Simple probabilistic model: 90% chance of full fill, 10% partial
        if random.random() < 0.9:
            return order_qty
        
        return int(order_qty * 0.5) # Fill half

    def _fill_order(self, order: ShadowOrder, price: float, qty: float):
        self.risk_monitor.record_trade()
        
        # Calculate Fees
        fees = self.fee_model.calculate_fees(price, qty, order.side.value)
        
        # Create Fill
        fill = ShadowFill(
            fill_id=f"FILL_{len(self.audit_trail.fills)}",
            order_id=order.order_id,
            timestamp=datetime.now(),
            price=price,
            quantity=qty,
            fee_breakdown=fees,
            liquidity_flag="TAKER" # Assuming aggressive for now
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
        
        # Simple Average Price implementation
        if side == Side.BUY:
            new_qty = pos.quantity + qty
            pos.average_price = ((pos.quantity * pos.average_price) + (qty * price)) / new_qty if new_qty > 0 else 0.0
            pos.quantity = new_qty
        else: # SELL
            # Realized PnL Calculation (FIFO/Average Cost)
            # Using Average Cost for simplicity
            pnl = (price - pos.average_price) * qty
            pos.realized_pnl += pnl
            pos.quantity -= qty
            
            # Check for Loss to update Risk Monitor
            # Note: PnL here is Gross. Net PnL = Gross - Fees.
            # Risk Monitor typically tracks Net PnL or Gross? Let's track Gross Loss for now, or Net.
            # Let's say we track Net Loss.
            net_pnl = pnl - fees.total_tax_and_fees
            if net_pnl < 0:
                self.risk_monitor.record_loss(abs(net_pnl))

        pos.total_fees += fees.total_tax_and_fees
        self.positions[symbol] = pos

    def cancel_order(self, order_id: str) -> None:
        pass

