from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class VolumeDelta:
    """Volume delta state at a specific timestamp."""
    timestamp: datetime
    delta: float            # buy_volume − sell_volume for this tick
    cumulative_delta: float # running total across all ticks
    buy_volume: float
    sell_volume: float


class VolumeDeltaTracker:
    """
    Deterministic streaming volume delta tracker.

    Aggressor-side determination:
      1. If 'aggressor_side' is provided ('buy' or 'sell') → use it directly.
      2. Otherwise apply the Tick Rule:
           price > last_price  → buyer-aggressed (buy)
           price < last_price  → seller-aggressed (sell)
           price == last_price → repeat last side; if no history → neutral (0)

    The cumulative delta is the running sum of (buy_vol − sell_vol) across
    all ticks received since tracker construction.

    Fully deterministic: same ordered sequence of ticks → same outputs.
    No randomness, no wall-clock dependency.
    """

    def __init__(self):
        self._cumulative_delta: float = 0.0
        self._last_price: Optional[float] = None
        self._last_side: Optional[str] = None  # 'buy' or 'sell'

    def update(
        self,
        timestamp: datetime,
        price: float,
        volume: float,
        aggressor_side: Optional[str] = None,
    ) -> VolumeDelta:
        """
        Updates delta state with a new tick.

        Args:
            timestamp:      Event timestamp (not wall clock).
            price:          Last traded price.
            volume:         Volume of this tick (must be > 0).
            aggressor_side: 'buy', 'sell', or None (uses Tick Rule if None).

        Returns:
            VolumeDelta snapshot for this tick.
        """
        if volume <= 0:
            raise ValueError(f"volume must be > 0, got {volume}")

        # Determine aggressor side
        if aggressor_side is not None:
            side = aggressor_side.lower()
            if side not in ("buy", "sell"):
                raise ValueError(f"aggressor_side must be 'buy' or 'sell', got {aggressor_side!r}")
        else:
            # Tick Rule — deterministic
            if self._last_price is None:
                side = "neutral"
            elif price > self._last_price:
                side = "buy"
            elif price < self._last_price:
                side = "sell"
            else:
                side = self._last_side if self._last_side else "neutral"

        if side == "buy":
            buy_vol, sell_vol = volume, 0.0
        elif side == "sell":
            buy_vol, sell_vol = 0.0, volume
        else:  # neutral
            buy_vol, sell_vol = 0.0, 0.0

        tick_delta = buy_vol - sell_vol
        self._cumulative_delta += tick_delta
        self._last_price = price
        self._last_side = side if side != "neutral" else self._last_side

        return VolumeDelta(
            timestamp=timestamp,
            delta=tick_delta,
            cumulative_delta=self._cumulative_delta,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
        )
