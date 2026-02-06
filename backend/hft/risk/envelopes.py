from dataclasses import dataclass
from datetime import datetime
from typing import Dict

@dataclass(frozen=True)
class RiskConstraints:
    max_trades_per_minute: int
    max_loss_per_minute: float
    max_position_size: float
    max_drawdown_session: float

@dataclass
class RiskState:
    """
    Tracks current usage of risk limits.
    """
    trades_last_minute: int = 0
    loss_last_minute: float = 0.0
    current_position_size: float = 0.0
    current_drawdown: float = 0.0
    last_reset_time: datetime = datetime.min

class RiskGate:
    """
    Risk envelopes that ONLY constrain behavior.
    Does NOT approve trades, only blocks them if limits are exceeded.
    """
    def __init__(self, constraints: RiskConstraints):
        self.constraints = constraints
        self.state = RiskState()

    def check_trade(self, symbol: str, quantity: float, side: str) -> bool:
        """
        Returns True if trade is allowed (within envelope), False otherwise.
        Expected PnL impact should be pre-calculated or estimated.
        """
        pass
