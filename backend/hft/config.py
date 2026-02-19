from dataclasses import dataclass, field
from enum import Enum
from .risk.throttling import VolatilityRegime
from .risk.limits import RiskConfig


class ExecutionMode(Enum):
    SHADOW_ONLY = "SHADOW_ONLY"
    LIVE = "LIVE"   # Explicitly different, requires override


@dataclass(frozen=True)
class FeeConfig:
    """
    2024/2025 Regulatory Fee Rates — NSE/BSE/Zerodha Reference.
    All values are deterministic constants. Never sourced from runtime state.

    Sources:
      NSE Equity: https://nseindia.com/trade/securities/transactionCharges.htm
      NSE F&O:    NSE circular no. NSE/FAOP/49765
      Crypto TDS: Finance Act 2022, Section 115BBH / 194S
    """
    # ── Equity ────────────────────────────────────────────────────────────────
    BROKERAGE_PCT: float            = 0.0003        # 0.03% (discount broker cap: ₹20)
    BROKERAGE_MAX: float            = 20.0          # ₹20 per order
    STT_INTRADAY_SELL_PCT: float    = 0.00025       # 0.025% on sell-side turnover
    STT_DELIVERY_PCT: float         = 0.001         # 0.1% on both sides
    EXCHANGE_TXN_NSE_PCT: float     = 0.0000325     # ₹3.25 per ₹1Cr (capital mkt segment)
    SEBI_CHARGES_PCT: float         = 0.000001      # ₹10 per ₹1Cr
    STAMP_DUTY_INTRADAY_PCT: float  = 0.00003       # 0.003% on buy-side
    STAMP_DUTY_DELIVERY_PCT: float  = 0.00015       # 0.015% on buy-side
    GST_PCT: float                  = 0.18          # 18% on brokerage + exchange + SEBI

    # ── F&O — Futures (NSE NFO segment) ───────────────────────────────────────
    STT_FUTURES_SELL_PCT: float     = 0.0001        # 0.01% on sell turnover (notional)
    EXCHANGE_TXN_NFO_FUT_PCT: float = 0.000002      # ₹2/₹1Cr futures turnover
    STAMP_DUTY_FUT_PCT: float       = 0.00002       # 0.002% on buy premium

    # ── F&O — Options (NSE NFO segment) ───────────────────────────────────────
    STT_OPTIONS_SELL_PCT: float     = 0.001         # 0.1% on sell side premium
    EXCHANGE_TXN_NFO_OPT_PCT: float = 0.000053      # ₹53/₹1Cr options premium turnover
    STAMP_DUTY_OPT_PCT: float       = 0.00003       # 0.003% on buy premium

    # ── Crypto (Exchange-Agnostic Baseline) ───────────────────────────────────
    CRYPTO_TAKER_PCT: float         = 0.001         # 0.10% taker fee (standard tier)
    CRYPTO_MAKER_PCT: float         = 0.0004        # 0.04% maker fee
    CRYPTO_TDS_PCT: float           = 0.01          # 1% TDS on sell (Finance Act 2022 s.194S)

    # ── Slippage Model Constants ───────────────────────────────────────────────
    SLIPPAGE_K: float               = 0.1           # volume impact coefficient
    DEFAULT_ADV: float              = 10_000.0      # default avg daily volume (shares) if no snapshot


@dataclass
class StrategyConfig:
    spread_ema_alpha: float     = 0.1
    momentum_window_ticks: int  = 10
    obi_threshold: float        = 0.3
    regime_enabled: bool        = True


@dataclass
class SystemConfig:
    tick_window_size: int           = 1000
    execution_mode: ExecutionMode   = ExecutionMode.SHADOW_ONLY
    log_level: str                  = "INFO"


@dataclass
class HFTConfig:
    risk: RiskConfig         = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    system: SystemConfig     = field(default_factory=SystemConfig)
    fees: FeeConfig          = field(default_factory=FeeConfig)


# Singleton default config — used throughout the engine
default_config = HFTConfig()
