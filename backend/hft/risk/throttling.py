from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Deque, Tuple, Optional
from collections import deque
import time

from backend.hft.risk.limits import RiskConfig
from backend.hft.models.trade_event import RiskStopReason

class VolatilityRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass(frozen=True)
class ThrottlingConfig:
    """
    Configuration for activity scaling based on regime.
    """
    base_rate_limit: int  # Orders per second
    regime_multipliers: Dict[VolatilityRegime, float] = field(default_factory=lambda: {
        VolatilityRegime.LOW: 1.2,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.HIGH: 0.5,
        VolatilityRegime.EXTREME: 0.0
    })

class RegimeThrottler:
    """
    Regime-Aware Throttling Logic.
    """
    def __init__(self, config: Optional[ThrottlingConfig] = None):
        if config is None:
            self.config = ThrottlingConfig(base_rate_limit=5)
        else:
            self.config = config

    def get_effective_limit(self, regime: VolatilityRegime) -> float:
        """
        Returns the effective rate limit (orders/sec) for the given regime.
        """
        return self.config.base_rate_limit * self.config.regime_multipliers.get(regime, 1.0)

class RiskGate:
    """
    Deterministic Risk Gate.
    Enforces hard limits on trades and losses per minute.
    Returns structured Stop Reasons.
    """
    def __init__(self, config: RiskConfig):
        self.config = config
        self.trade_timestamps: Deque[float] = deque()
        self.loss_timestamps: Deque[Tuple[float, float]] = deque() # (timestamp, loss_amount)
        self.is_halted: bool = False
        self.halt_reason: RiskStopReason = RiskStopReason.UNKNOWN

    def record_trade(self):
        """Call this when a trade is executed."""
        self.trade_timestamps.append(time.time())

    def record_loss(self, loss_amount: float):
        """Call this when a trade results in a Realized Loss."""
        if loss_amount > 0:
            self.loss_timestamps.append((time.time(), loss_amount))

    def check_risk(self, regime: VolatilityRegime = VolatilityRegime.NORMAL) -> Tuple[bool, Optional[RiskStopReason]]:
        """
        Checks if we can trade.
        Returns (Allowed: bool, Reason: Optional[RiskStopReason])
        """
        if self.is_halted:
            return False, self.halt_reason

        now = time.time()
        
        # 1. Prune old records (older than 60s)
        while self.trade_timestamps and self.trade_timestamps[0] < now - 60:
            self.trade_timestamps.popleft()
            
        while self.loss_timestamps and self.loss_timestamps[0][0] < now - 60:
            self.loss_timestamps.popleft()

        # 2. Check Throttle (Regime)
        # If regime is EXTREME, we might just stop.
        if regime == VolatilityRegime.EXTREME:
             return False, RiskStopReason.REGIME_THROTTLE

        # 3. Check Max Trades / Min
        if len(self.trade_timestamps) >= self.config.max_trades_per_min:
            return False, RiskStopReason.MAX_TRADES_LIMIT

        # 4. Check Max Loss / Min
        recent_loss = sum(l[1] for l in self.loss_timestamps)
        if recent_loss >= self.config.max_loss_per_min:
            return False, RiskStopReason.MAX_LOSS_MINUTE

        return True, None
    
    def force_halt(self, reason: RiskStopReason):
        self.is_halted = True
        self.halt_reason = reason
