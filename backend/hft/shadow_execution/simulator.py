from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

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
    commission: float
    liquidity_flag: str # 'MAKER' or 'TAKER'

@dataclass
class ShadowPosition:
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

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
    def __init__(self):
        self.audit_trail = SimulationAuditTrail(
            session_id=datetime.now().isoformat(),
            start_time=datetime.now()
        )
        self.positions: Dict[str, ShadowPosition] = {}

    def place_order(self, order: ShadowOrder) -> None:
        pass

    def cancel_order(self, order_id: str) -> None:
        pass
