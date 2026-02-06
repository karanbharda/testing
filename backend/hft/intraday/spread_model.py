from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class SpreadMetrics:
    """
    Container for spread analysis metrics at a specific point in time.
    """
    timestamp: datetime  # Exchange timestamp (Event Time)
    symbol: str
    best_bid: float
    best_ask: float
    spread_absolute: float  # Ask - Bid
    spread_relative: float  # (Ask - Bid) / Midpoint
    spread_smoothed: float  # EMA/SMA of spread

class SpreadTracker:
    """
    Tracks and smoothes bid-ask spread over time.
    """
    def __init__(self, alpha: float = 0.1):
        """
        Initialize the spread tracker.
        
        Args:
           alpha (float): Smoothing factor for EMA (0 < alpha <= 1).
        """
        self.alpha = alpha
        self._last_smoothed_spread: Optional[float] = None

    def update(self, timestamp: datetime, best_bid: float, best_ask: float) -> SpreadMetrics:
        """
        Updates the spread model with new L1 data.
        
        Computation:
        1. Calculate raw spread (Ask - Bid).
        2. Calculate mid-price.
        3. Calculate relative spread.
        4. Update smoothed spread using EMA:
           Smoothed_t = alpha * Spread_t + (1 - alpha) * Smoothed_{t-1}
        5. Timestamp using the provided event timestamp.
        
        Returns:
            SpreadMetrics object with calculated values.
        """
        pass
