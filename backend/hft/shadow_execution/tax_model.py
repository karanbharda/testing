from dataclasses import dataclass
from backend.hft.models.trade_event import TradeType

@dataclass
class TaxClassification:
    category: str
    description: str

class TaxModel:
    """
    Tax Awareness Layer (Explainable, Not Advisory).
    Maps trade types to income categories.
    """
    
    def classify_trade(self, trade_type: TradeType) -> TaxClassification:
        if trade_type == TradeType.EQUITY_INTRADAY:
            return TaxClassification(
                category="BUSINESS_INCOME",
                description="Intraday profits are treated as Speculative Business Income (taxed at slab rates)."
            )
        elif trade_type == TradeType.EQUITY_DELIVERY:
            return TaxClassification(
                category="STCG_OR_LTCG",
                description="Delivery trades attract Capital Gains tax (STCG 20% or LTCG 12.5% if > 1Yr)."
            )
        else:
            return TaxClassification(category="UNKNOWN", description="Unclassified trade type")
