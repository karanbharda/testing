from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Optional, Tuple
from collections import deque


@dataclass(frozen=True)
class MomentumEvent:
    timestamp: datetime
    symbol: str
    window_duration_ms: float       # elapsed time across the window (milliseconds)
    price_change_bps: float         # (p_last - p_first) / p_first × 10000
    velocity_bps_per_sec: float     # price_change_bps / window_duration_sec


class MicroMomentumTracker:
    """
    Measures intraday price velocity over a sliding N-tick window.

    Velocity = (price_change_bps) / (window_duration_seconds)

    Returns None until the window is fully populated (< window_ticks ticks seen).
    Once warm, slides forward tick-by-tick (oldest tick dropped automatically).

    Fully deterministic: same ordered tick stream → same outputs.
    No randomness, no wall-clock dependency.
    """

    def __init__(self, window_ticks: int = 10):
        if window_ticks < 2:
            raise ValueError(f"window_ticks must be >= 2, got {window_ticks}")
        self.window_ticks = window_ticks
        self._history: Deque[Tuple[datetime, float]] = deque(maxlen=window_ticks)

    def update(self, timestamp: datetime, price: float, symbol: str = "") -> Optional[MomentumEvent]:
        """
        Adds a new tick and computes momentum if window is full.

        Args:
            timestamp: Event timestamp (not wall clock).
            price:     Current price.
            symbol:    Instrument symbol (optional, for tagging output).

        Returns:
            MomentumEvent if window_ticks ticks have been observed, else None.
        """
        self._history.append((timestamp, price))

        if len(self._history) < self.window_ticks:
            return None     # warm-up: not enough history yet

        oldest_ts, oldest_price = self._history[0]
        latest_ts,  latest_price = self._history[-1]

        # Duration in milliseconds
        td = latest_ts - oldest_ts
        duration_ms = td.total_seconds() * 1000.0

        if duration_ms <= 0.0:
            # All ticks share the same timestamp — velocity undefined
            return MomentumEvent(
                timestamp=timestamp,
                symbol=symbol,
                window_duration_ms=0.0,
                price_change_bps=0.0,
                velocity_bps_per_sec=0.0,
            )

        price_change_bps = (latest_price - oldest_price) / oldest_price * 10_000.0
        velocity = price_change_bps / (duration_ms / 1_000.0)  # convert ms → sec

        return MomentumEvent(
            timestamp=timestamp,
            symbol=symbol,
            window_duration_ms=duration_ms,
            price_change_bps=price_change_bps,
            velocity_bps_per_sec=velocity,
        )
