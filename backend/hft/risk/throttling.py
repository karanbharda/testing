from dataclasses import dataclass
from enum import Enum

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
    regime_multipliers: Dict[VolatilityRegime, float]
    # e.g., {LOW: 1.2, NORMAL: 1.0, HIGH: 0.5, EXTREME: 0.0}

class RegimeThrottler:
    """
    Regime-Aware Throttling Logic.
    
    Analytical Logic:
    1. Input: Current Volatility Regime (V_regime).
    2. Lookup: Retrieve Multiplier (M) for V_regime.
    3. Calculate Effective Limit (L_eff):
       L_eff = Base_Limit * M
    4. Action:
       - If Market is Volatile (HIGH/EXTREME) -> M < 1.0 -> Reduces Activity.
       - If Market is Calm (LOW/NORMAL) -> M >= 1.0 -> Allows standard/high activity.
    """
    def __init__(self, config: ThrottlingConfig):
        self.config = config

    def get_effective_limit(self, regime: VolatilityRegime) -> float:
        """
        Returns the effective rate limit for the given regime.
        """
        return self.config.base_rate_limit * self.config.regime_multipliers.get(regime, 1.0)
