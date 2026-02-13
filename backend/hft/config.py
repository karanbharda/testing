from dataclasses import dataclass, field
from enum import Enum
from .risk.throttling import VolatilityRegime
from .risk.limits import RiskConfig

class ExecutionMode(Enum):
    SHADOW_ONLY = "SHADOW_ONLY"
    LIVE = "LIVE"  # Explicitly different, requires override

@dataclass(frozen=True)
class FeeConfig:
    # 2024/2025 Rates
    BROKERAGE_PCT: float = 0.0003
    BROKERAGE_MAX: float = 20.0
    STT_INTRADAY_SELL_PCT: float = 0.00025
    STT_DELIVERY_PCT: float = 0.1
    EXCHANGE_TXN_NSE_PCT: float = 0.0000325
    SEBI_CHARGES_PCT: float = 0.000001
    STAMP_DUTY_INTRADAY_PCT: float = 0.00003
    STAMP_DUTY_DELIVERY_PCT: float = 0.00015
    GST_PCT: float = 0.18

@dataclass
class StrategyConfig:
    spread_ema_alpha: float = 0.1
    momentum_window_ticks: int = 10
    obi_threshold: float = 0.3
    regime_enabled: bool = True

@dataclass
class SystemConfig:
    tick_window_size: int = 1000
    execution_mode: ExecutionMode = ExecutionMode.SHADOW_ONLY
    log_level: str = "INFO"

@dataclass
class HFTConfig:
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)

# Default Instance
default_config = HFTConfig()
