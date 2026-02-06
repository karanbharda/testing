from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass(frozen=True)
class Tick:
    """
    Represents a single market tick (price update).
    """
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float

class TickBuffer:
    """
    A simple in-memory buffer to store ticks in time order.
    Not linked to any broker.
    """
    def __init__(self):
        self._ticks: List[Tick] = []

    def add_tick(self, tick: Tick) -> None:
        """Stores a new tick in the buffer."""
        self._ticks.append(tick)

    def get_ticks(self) -> List[Tick]:
        """Returns a copy of all stored ticks."""
        return list(self._ticks)

    def clear(self) -> None:
        """Clears all stored ticks."""
        self._ticks.clear()

    def __len__(self) -> int:
        return len(self._ticks)
