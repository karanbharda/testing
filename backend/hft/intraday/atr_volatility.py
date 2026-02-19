from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class AtrOutput:
    timestamp: datetime
    true_range: float
    atr_value: float


class IntradayATR:
    """
    Streaming Average True Range (ATR) using Wilder's EMA smoothing.

    True Range (TR):
        TR_t = max(H_t − L_t, |H_t − C_{t-1}|, |L_t − C_{t-1}|)

    ATR update rule (Wilder's EMA):
        ATR_t = (ATR_{t-1} × (period − 1) + TR_t) / period

    Cold-start (first 'period' candles):
        ATR = simple arithmetic mean of TR values.
        After that, Wilder's EMA runs indefinitely.

    Fully deterministic: identical candle sequence → identical ATR values.
    No randomness, no wall-clock dependency.
    """

    def __init__(self, period: int = 14):
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self.period = period
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None
        self._warmup_trs: List[float] = []       # accumulates until warm-up complete

    def update(self, candle: Candle) -> AtrOutput:
        """
        Updates ATR with a new completed candle.

        Returns:
            AtrOutput with the current true_range and atr_value.
            During warm-up (fewer than 'period' candles), atr_value
            is the running SMA of observed TRs so far.
        """
        # Compute True Range
        if self._prev_close is None:
            # First candle: TR = High − Low (no prior close)
            tr = candle.high - candle.low
        else:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - self._prev_close),
                abs(candle.low  - self._prev_close),
            )

        self._prev_close = candle.close

        # Warm-up phase: SMA of TRs
        if self._atr is None:
            self._warmup_trs.append(tr)
            current_atr = sum(self._warmup_trs) / len(self._warmup_trs)
            if len(self._warmup_trs) >= self.period:
                self._atr = current_atr   # transition to Wilder's EMA
        else:
            # Wilder's EMA (smoothing factor = 1/period)
            self._atr = (self._atr * (self.period - 1) + tr) / self.period
            current_atr = self._atr

        return AtrOutput(
            timestamp=candle.timestamp,
            true_range=tr,
            atr_value=current_atr,
        )
