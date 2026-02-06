from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any

@dataclass(frozen=True)
class FeatureVector:
    """
    Read-only data contract between Intraday Features and downstream ML/RL models.
    Strictly contains features. NO signals, NO execution paths.
    """
    timestamp: datetime
    symbol: str
    
    # Core Features
    spread_bps: float
    obi_value: float
    volume_delta: float
    micro_momentum_bps: float
    volatility_garch: float
    
    # Regime / Context
    current_regime: str  # e.g., "TRENDING_UP", "RANGING"
    
    # Metadata (optional, specific feature versions)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "features": {
                "spread_bps": self.spread_bps,
                "obi_value": self.obi_value,
                "volume_delta": self.volume_delta,
                "micro_momentum_bps": self.micro_momentum_bps,
                "volatility_garch": self.volatility_garch,
                "current_regime": self.current_regime
            }
        }
