from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class OrderBookImbalance:
    """
    Represents Order Book Imbalance (OBI) at a specific timestamp.
    """
    timestamp: datetime
    symbol: str
    bid_qty: float
    ask_qty: float
    obi_value: float  # (bid - ask) / (bid + ask)

    @property
    def interpretation(self) -> str:
        if self.obi_value > 0.3:
            return "Buy Pressure"
        elif self.obi_value < -0.3:
            return "Sell Pressure"
        return "Balanced"
