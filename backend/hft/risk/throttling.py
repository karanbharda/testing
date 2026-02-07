from dataclasses import dataclass
from enum import Enum

class VolatilityRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

from typing import Dict, Deque, Tuple
from collections import deque
import time
from backend.hft.risk.limits import RiskConfig

@dataclass(frozen=True)
class ThrottlingConfig:
    """
    Configuration for activity scaling based on regime.
    """
    base_rate_limit: int  # Orders per second
    regime_multipliers: Dict[VolatilityRegime, float]
    # e.g., {LOW: 1.2, NORMAL: 1.0, HIGH: 0.5, EXTREME: 0.0}

class RegimeThrottler:
    """
    Regime-Aware Throttling Logic.
    """
    def __init__(self, config: ThrottlingConfig):
        self.config = config

    def get_effective_limit(self, regime: VolatilityRegime) -> float:
        """
        Returns the effective rate limit (orders/sec) for the given regime.
        """
        return self.config.base_rate_limit * self.config.regime_multipliers.get(regime, 1.0)

class RiskMonitor:
    """
    Stateful Risk Monitor.
    Tracks:
    - Trades per minute (Sliding Window)
    - PnL per minute (Approximation or Snapshot) - To simplify, we track realized loss in last window.
    """
    def __init__(self, config: RiskConfig):
        self.config = config
        self.trade_timestamps: Deque[float] = deque()
        self.loss_timestamps: Deque[Tuple[float, float]] = deque() # (timestamp, loss_amount)
        self.is_halted: bool = False
        self.halt_reason: str = ""

    def record_trade(self):
        """Call this when a trade is executed."""
        self.trade_timestamps.append(time.time())

    def record_loss(self, loss_amount: float):
        """Call this when a trade results in a Realized Loss."""
        if loss_amount > 0:
            self.loss_timestamps.append((time.time(), loss_amount))

    def check_risk(self, regime: VolatilityRegime = VolatilityRegime.NORMAL) -> Tuple[bool, str]:
        """
        Checks if we can trade.
        Returns (Allowed: bool, Reason: str)
        """
        if self.is_halted:
            return False, f"HALTED: {self.halt_reason}"

        now = time.time()
        
        # 1. Prune old records (older than 60s)
        while self.trade_timestamps and self.trade_timestamps[0] < now - 60:
            self.trade_timestamps.popleft()
            
        while self.loss_timestamps and self.loss_timestamps[0][0] < now - 60:
            self.loss_timestamps.popleft()

        # 2. Check Max Trades / Min
        if len(self.trade_timestamps) >= self.config.max_trades_per_min:
            return False, f"RATE_LIMIT: {len(self.trade_timestamps)} trades in last minute (Max {self.config.max_trades_per_min})"

        # 3. Check Max Loss / Min
        recent_loss = sum(l[1] for l in self.loss_timestamps)
        if recent_loss >= self.config.max_loss_per_min:
            return False, f"MAX_LOSS_LIMIT: Loss {recent_loss:.2f} in last minute (Max {self.config.max_loss_per_min})"

        return True, "OK"
    
    def force_halt(self, reason: str):
        self.is_halted = True
        self.halt_reason = reason
