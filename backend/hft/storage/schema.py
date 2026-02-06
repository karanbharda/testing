from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

@dataclass(frozen=True)
class TickLog:
    """
    Immutable storage format for raw ticks.
    """
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    sequence_id: int

@dataclass(frozen=True)
class DerivedFeatureLog:
    """
    Immutable storage format for derived intraday features.
    Should match the FeatureVector content but optimized for persistence (e.g., Parquet/HDF5 compatible).
    """
    timestamp: datetime
    symbol: str
    spread: float
    obi: float
    volatility: float
    regime: str

@dataclass(frozen=True)
class ShadowOutcomeLog:
    """
    Immutable storage format for shadow trade outcomes.
    """
    timestamp: datetime
    session_id: str
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    exit_price: float
    pnl_realized: float
    fees_incurred: float
    slippage_incurred: float
