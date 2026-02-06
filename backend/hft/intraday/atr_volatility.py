from dataclasses import dataclass
from datetime import datetime
from typing import Optional

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
    Calculates Average True Range (ATR) on streaming candle data.
    """
    def __init__(self, period: int = 14):
        self.period = period
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None

    def update(self, candle: Candle) -> AtrOutput:
        """
        Updates ATR with a new completed candle.
        """
        pass
