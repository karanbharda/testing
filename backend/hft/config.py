from dataclasses import dataclass, field
from .risk.throttling import VolatilityRegime

@dataclass
class RiskConfig:
    max_trades_per_minute: int = 5
    max_loss_per_minute: float = 1000.0
    max_position_size: float = 100000.0
    max_drawdown_session: float = 5000.0

@dataclass
class StrategyConfig:
    spread_ema_alpha: float = 0.1
    momentum_window_ticks: int = 10
    obi_threshold: float = 0.3
    regime_enabled: bool = True

@dataclass
class SystemConfig:
    tick_window_size: int = 1000
    shadow_mode: bool = True
    log_level: str = "INFO"

@dataclass
class HFTConfig:
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

# Default Instance
default_config = HFTConfig()
