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
    """
    def __init__(self, max_size: int = 10000, drop_strategy: str = "DROP_OLDEST"):
        self.max_size = max_size
        self.drop_strategy = drop_strategy # "DROP_OLDEST" or "DROP_NEWEST"
        self._buffer: Deque[Tick] = deque(maxlen=max_size if drop_strategy == "DROP_OLDEST" else None)
        self.dropped_ticks_count = 0

    def add_tick(self, tick: Tick) -> bool:
        """
        Adds a tick to the buffer.
        Returns True if added, False if dropped (for DROP_NEWEST).
        """
        if self.drop_strategy == "DROP_OLDEST":
            # deque with maxlen automatically drops oldest
            if len(self._buffer) == self.max_size:
                 self.dropped_ticks_count += 1
            
            # Monotonicity Check
            if self._buffer and tick.timestamp < self._buffer[-1].timestamp:
                # Log warning? For now just accept but note it's an anomaly if needed
                pass 

            self._buffer.append(tick)
            return True
        else: # DROP_NEWEST
            if len(self._buffer) >= self.max_size:
                self.dropped_ticks_count += 1
                return False # Reject new tick
            
            # Monotonicity Check
            if self._buffer and tick.timestamp < self._buffer[-1].timestamp:
                pass
                
            self._buffer.append(tick)
            return True

    def get_snapshot(self) -> List[Tick]:
        """Returns a list of current ticks."""
        return list(self._buffer)

    def get_latest(self) -> Optional[Tick]:
        """Returns the most recent tick, or None if empty."""
        return self._buffer[-1] if self._buffer else None

    def size(self) -> int:
        return len(self._buffer)
    
    def clear(self):
        self._buffer.clear()
    
    def __len__(self) -> int:
        return len(self._buffer)
