from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class VolumeDelta:
    """
    Represents the volume delta state at a specific timestamp.
    """
    timestamp: datetime
    delta: float          # Buy Vol - Sell Vol for this interval/tick
    cumulative_delta: float # Running total of delta
    buy_volume: float
    sell_volume: float

class VolumeDeltaTracker:
    """
    Tracks streaming volume delta using the Tick Rule or Aggressor side.
    """
    def __init__(self):
        self._cumulative_delta = 0.0
        self._last_price = None

    def update(self, timestamp: datetime, price: float, volume: float, aggressor_side: str = None) -> VolumeDelta:
        """
        Updates delta state.
        
        Args:
            aggressor_side: 'buy', 'sell', or None (uses Tick Rule if None)
        """
        pass
