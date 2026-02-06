from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class RegimeType(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class MarketRegime:
    timestamp: datetime
    symbol: str
    regime: RegimeType
    confidence: float
    basis_explanation: str # e.g. "ADX > 25 (30.5) and Price > SMA(20)"

class RegimeDetector:
    """
    Deterministic regime classifier based on technical thresholds.
    No ML, purely rule-based.
    """
    def update(self, timestamp: datetime, price: float, volatility: float) -> MarketRegime:
        pass
