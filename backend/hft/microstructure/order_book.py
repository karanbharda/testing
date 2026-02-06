from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

@dataclass(frozen=True)
class OrderBookSnapshot:
    """
    Represents a snapshot of the order book at a specific point in time.
    Bids and Asks are lists of (price, size) tuples.
    """
    timestamp: datetime
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
