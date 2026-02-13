from collections import deque
from typing import Optional, Deque, List
import time
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Tick:
    """
    Represents a single market tick (price update).
    """
    symbol: str
    price: float
    volume: int
    timestamp: float
    bid: float = 0.0
    ask: float = 0.0

class TickBuffer:
    """
    Hardened Tick Buffer with Fixed Size and Overflow Policy.
    Includes backpressure monitoring.
    """
    def __init__(self, max_size: int = 10000, drop_strategy: str = "DROP_OLDEST"):
        self.max_size = max_size
        self.drop_strategy = drop_strategy # "DROP_OLDEST" 
        self._buffer: Deque[Tick] = deque(maxlen=max_size) # deque handles DROP_OLDEST naturally
        self.dropped_ticks_count = 0
        self.peak_lag_ms = 0.0
        
    def add_tick(self, tick: Tick) -> bool:
        """
        Adds a tick to the buffer.
        """
        # Backpressure Check: In a real async system we'd check if consumer is keeping up.
        # Here we check if we are overwriting data too fast?
        # Deque with maxlen automatically drops oldest.
        
        if len(self._buffer) == self.max_size:
             self.dropped_ticks_count += 1
        
        # Monotonicity Check
        if self._buffer and tick.timestamp < self._buffer[-1].timestamp:
            # Out of order tick validation could go here
            pass 

        self._buffer.append(tick)
        return True

    def get_snapshot(self) -> List[Tick]:
        """Returns a list of current ticks."""
        return list(self._buffer)

    def get_latest(self) -> Optional[Tick]:
        """Returns the most recent tick, or None if empty."""
        return self._buffer[-1] if self._buffer else None
    
    def get_backpressure_metrics(self) -> dict:
        return {
            "buffer_usage_pct": (len(self._buffer) / self.max_size) * 100,
            "dropped_ticks": self.dropped_ticks_count,
            "is_full": len(self._buffer) == self.max_size
        }

    def size(self) -> int:
        return len(self._buffer)
    
    def clear(self):
        self._buffer.clear()
        self.dropped_ticks_count = 0
    
    def __len__(self) -> int:
        return len(self._buffer)
