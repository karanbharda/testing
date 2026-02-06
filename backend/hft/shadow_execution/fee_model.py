from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

@dataclass(frozen=True)
class FeeBreakdown:
    brokerage: float
    exchange_txn_charge: float
    gst: float
    stt: float
    sebi_fees: float
    stamp_duty: float
    total_tax_and_fees: float

class FeeModel:
    """
    Calculates detailed fee structures for Indian Intraday Equity/FnO.
    """
    def calculate_fees(self, price: float, qty: float, side: str, is_intraday: bool = True) -> FeeBreakdown:
        """
        Computes the complete fee breakdown for a single leg of a trade.
        
        Args:
            side: 'BUY' or 'SELL'
        """
        pass
