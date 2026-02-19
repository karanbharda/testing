import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SlippageEstimate:
    """
    Deterministic slippage estimate for a prospective trade.

    price_impact_bps = k × √(trade_size / recent_volume) × 10000
    total_cost_bps   = half_spread_bps + price_impact_bps
    confidence_score = min(recent_volume / trade_size, 1.0)
                       (1.0 = deep liquidity; 0.0 = no liquidity data)
    """
    timestamp: datetime
    trade_size: float               # order size (shares/contracts)
    estimated_price_impact: float   # market impact in bps
    estimated_total_cost: float     # half-spread + impact, in bps
    confidence_score: float         # 0.0 – 1.0


class SlippageModel:
    """
    Analytically estimates trade cost using the Square-Root Market Impact model.

    Formula:
        price_impact_bps = k × √(trade_size / recent_volume) × 10000
        total_cost_bps   = (spread_bps / 2) + price_impact_bps

    Confidence:
        confidence_score = clamp(recent_volume / trade_size, 0.0, 1.0)
        If recent_volume ≤ 0 → use default ADV as fallback (conservative).

    This model is:
      • Deterministic: same inputs → same outputs
      • Monotone: larger orders → larger impact
      • Conservative: missing liquidity data triggers worst-case ADV fallback
      • Purely analytical: does NOT place orders
    """

    DEFAULT_ADV: float = 10_000.0   # shares — used when no recent_volume given

    def __init__(self, impact_factor: float = 0.1):
        """
        Args:
            impact_factor (k): Square-root market impact coefficient.
                               Typical empirical range: 0.05 – 0.20.
        """
        if impact_factor <= 0:
            raise ValueError(f"impact_factor must be > 0, got {impact_factor}")
        self.impact_factor = impact_factor

    def estimate(
        self,
        timestamp: datetime,
        spread_bps: float,
        recent_volume: float,
        trade_size: float,
    ) -> SlippageEstimate:
        """
        Estimates expected slippage cost for a given trade.

        Args:
            timestamp:      Event timestamp.
            spread_bps:     Current bid-ask spread in basis points.
            recent_volume:  Available liquidity proxy (e.g. 1-min cumulative volume).
                            If <= 0, falls back to DEFAULT_ADV (conservative).
            trade_size:     Order size (shares or contracts).

        Returns:
            SlippageEstimate with impact, total cost, and confidence.
        """
        if trade_size <= 0:
            raise ValueError(f"trade_size must be > 0, got {trade_size}")
        if spread_bps < 0:
            raise ValueError(f"spread_bps must be >= 0, got {spread_bps}")

        # Use fallback if no liquidity data
        effective_volume = recent_volume if recent_volume > 0 else self.DEFAULT_ADV

        # Square-Root Market Impact Model
        volume_ratio = trade_size / effective_volume
        price_impact_bps = self.impact_factor * math.sqrt(volume_ratio) * 10_000.0

        half_spread = spread_bps / 2.0
        total_cost_bps = half_spread + price_impact_bps

        # Confidence: how much liquidity relative to order size
        confidence = min(effective_volume / trade_size, 1.0)

        return SlippageEstimate(
            timestamp=timestamp,
            trade_size=trade_size,
            estimated_price_impact=round(price_impact_bps, 6),
            estimated_total_cost=round(total_cost_bps, 6),
            confidence_score=round(confidence, 4),
        )
