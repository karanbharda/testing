from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class SpreadMetrics:
    """Bid-ask spread metrics at a specific point in time."""
    timestamp: datetime
    symbol: str
    best_bid: float
    best_ask: float
    spread_absolute: float   # Ask − Bid
    spread_relative: float   # (Ask − Bid) / mid-price   [dimensionless]
    spread_smoothed: float   # EMA-smoothed spread (bps of mid-price)


class SpreadTracker:
    """
    Deterministic bid-ask spread tracker using Exponential Moving Average.

    EMA update rule:
        smoothed_t = α × spread_t + (1 − α) × smoothed_{t−1}
    First tick: smoothed = raw spread (cold-start).

    All outputs are deterministic for a given sequence of inputs.
    No randomness, no wall-clock dependency.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: EMA smoothing factor (0 < alpha ≤ 1).
                   Smaller → slower adaptation; larger → faster.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._smoothed_spread: Optional[float] = None

    def update(self, timestamp: datetime, best_bid: float, best_ask: float) -> SpreadMetrics:
        """
        Updates the spread model with new L1 quote data.

        Computation:
          1. spread_abs = ask − bid
          2. mid         = (ask + bid) / 2
          3. spread_rel  = spread_abs / mid
          4. smoothed_t  = α × spread_abs + (1−α) × smoothed_{t−1}
             (first tick: smoothed = spread_abs)
          5. All values expressed using event timestamp (not wall clock).

        Returns:
            SpreadMetrics with all computed fields.
        """
        if best_ask <= best_bid:
            raise ValueError(
                f"Invalid quote: ask ({best_ask}) must be > bid ({best_bid})"
            )

        spread_abs = best_ask - best_bid
        mid = (best_ask + best_bid) / 2.0
        spread_rel = spread_abs / mid  # dimensionless ratio

        # EMA smoothing
        if self._smoothed_spread is None:
            self._smoothed_spread = spread_abs   # cold-start: no history to average
        else:
            self._smoothed_spread = (
                self.alpha * spread_abs + (1.0 - self.alpha) * self._smoothed_spread
            )

        return SpreadMetrics(
            timestamp=timestamp,
            symbol="",                          # caller fills symbol if needed
            best_bid=best_bid,
            best_ask=best_ask,
            spread_absolute=spread_abs,
            spread_relative=spread_rel,
            spread_smoothed=self._smoothed_spread,
        )
