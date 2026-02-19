import math
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Optional, Tuple
from collections import deque
from enum import Enum


class RegimeType(Enum):
    TRENDING_UP   = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING       = "RANGING"
    UNKNOWN       = "UNKNOWN"


@dataclass(frozen=True)
class MarketRegime:
    timestamp: datetime
    symbol: str
    regime: RegimeType
    confidence: float           # 0.0 – 1.0  (degree of deviation from SMA)
    basis_explanation: str      # e.g. "price=105.2 > SMA20=100.1 vel=+3.2bps/s"


class RegimeDetector:
    """
    Deterministic, rule-based market regime classifier.
    No ML. No randomness. Based purely on price relative to a rolling SMA
    and price velocity.

    Classification rules (all thresholds configurable via constructor):
      • TRENDING_UP:   price > SMA(window) AND velocity > trend_threshold_bps_s
      • TRENDING_DOWN: price < SMA(window) AND velocity < -trend_threshold_bps_s
      • RANGING:       otherwise

    Confidence = |price / SMA − 1| (normalised deviation from SMA, capped at 1.0).

    Returns UNKNOWN until the rolling window is fully populated.
    """

    def __init__(
        self,
        window_ticks: int  = 20,
        trend_threshold_bps_s: float = 1.0,   # min velocity (bps/s) to call a trend
    ):
        if window_ticks < 2:
            raise ValueError(f"window_ticks must be >= 2, got {window_ticks}")
        self.window_ticks = window_ticks
        self.trend_threshold = trend_threshold_bps_s
        self._price_window: Deque[Tuple[datetime, float]] = deque(maxlen=window_ticks)

    def update(self, timestamp: datetime, price: float, volatility: float = 0.0,
               symbol: str = "") -> MarketRegime:
        """
        Adds a new price observation and classifies the current regime.

        Args:
            timestamp:   Event timestamp (not wall clock).
            price:       Current price.
            volatility:  Optional annualised volatility (not used in rule logic,
                         reserved for future regime refinement).
            symbol:      Instrument label.

        Returns:
            MarketRegime with classification, confidence, and explanation.
        """
        self._price_window.append((timestamp, price))

        if len(self._price_window) < self.window_ticks:
            return MarketRegime(
                timestamp=timestamp,
                symbol=symbol,
                regime=RegimeType.UNKNOWN,
                confidence=0.0,
                basis_explanation=f"Warming up: {len(self._price_window)}/{self.window_ticks} ticks",
            )

        prices = [p for _, p in self._price_window]
        sma = sum(prices) / len(prices)

        # Price velocity over the full window
        oldest_ts, oldest_price = self._price_window[0]
        duration_s = (timestamp - oldest_ts).total_seconds()
        if duration_s > 0:
            velocity_bps_s = (price - oldest_price) / oldest_price * 10_000.0 / duration_s
        else:
            velocity_bps_s = 0.0

        # Confidence: normalised deviation from SMA
        deviation = abs(price / sma - 1.0) if sma != 0 else 0.0
        confidence = min(deviation * 100.0, 1.0)    # scale: 1% dev → conf~1.0

        # Classification
        if price > sma and velocity_bps_s > self.trend_threshold:
            regime = RegimeType.TRENDING_UP
            explanation = (
                f"price={price:.2f} > SMA{self.window_ticks}={sma:.2f}, "
                f"velocity=+{velocity_bps_s:.2f}bps/s"
            )
        elif price < sma and velocity_bps_s < -self.trend_threshold:
            regime = RegimeType.TRENDING_DOWN
            explanation = (
                f"price={price:.2f} < SMA{self.window_ticks}={sma:.2f}, "
                f"velocity={velocity_bps_s:.2f}bps/s"
            )
        else:
            regime = RegimeType.RANGING
            explanation = (
                f"price={price:.2f} ≈ SMA{self.window_ticks}={sma:.2f}, "
                f"velocity={velocity_bps_s:.2f}bps/s (below ±{self.trend_threshold})"
            )

        return MarketRegime(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            confidence=round(confidence, 4),
            basis_explanation=explanation,
        )
