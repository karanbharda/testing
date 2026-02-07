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
    tax_category: str = "BUSINESS_INCOME" # 'BUSINESS_INCOME' or 'CAPITAL_GAINS'

class FeeModel:
    """
    Calculates detailed fee structures for Indian Intraday Equity.
    Rates approx as of late 2024 (NSE/Zerodha standard).
    """
    
    # Constants
    BROKERAGE_PCT = 0.0003  # 0.03%
    BROKERAGE_MAX = 20.0    # Max Rs 20 per order
    
    STT_PCT_INTRADAY_SELL = 0.00025 # 0.025% (Sell only)
    STT_PCT_DELIVERY = 0.1          # 0.1% (Buy & Sell) - Not using for now, but good context
    
    EXCHANGE_TXN_NSE = 0.0000325    # 0.00325%
    
    SEBI_CHARGES_PCT = 0.000001     # Rs 10 per crore
    
    STAMP_DUTY_PCT_BUY = 0.00003    # 0.003% (Buy only)
    
    GST_PCT = 0.18                  # 18%

    def calculate_fees(self, price: float, qty: float, side: str, is_intraday: bool = True) -> FeeBreakdown:
        """
        Computes the complete fee breakdown for a single leg of a trade.
        
        Args:
            price: Execution price
            qty: Quantity
            side: 'BUY' or 'SELL'
            is_intraday: Check for Intraday rates (default True)
        """
        turnover = price * qty
        
        # 1. Brokerage: 0.03% or Rs 20, whichever is lower
        brokerage = min(turnover * self.BROKERAGE_PCT, self.BROKERAGE_MAX)
        
        # 2. STT: 
        # Intraday: 0.025% on Sell only
        # Delivery: 0.1% on Buy & Sell (Not implemented here properly for Delivery, assuming Intraday focus)
        stt = 0.0
        if is_intraday:
            if side == "SELL":
                stt = turnover * self.STT_PCT_INTRADAY_SELL
        else:
            # Placeholder for delivery logic if ever needed
            stt = turnover * 0.001 

        # 3. Exchange Txn Charges
        exchange_txn_charge = turnover * self.EXCHANGE_TXN_NSE
        
        # 4. SEBI Charges
        sebi_fees = turnover * self.SEBI_CHARGES_PCT
        
        # 5. Stamp Duty: 0.003% on Buy only
        stamp_duty = 0.0
        if side == "BUY":
            stamp_duty = turnover * self.STAMP_DUTY_PCT_BUY
            
        # 6. GST: 18% on (Brokerage + Exchange Txn + SEBI)
        gst = (brokerage + exchange_txn_charge + sebi_fees) * self.GST_PCT
        
        total_tax_and_fees = brokerage + stt + exchange_txn_charge + sebi_fees + stamp_duty + gst
        
        tax_category = "BUSINESS_INCOME" if is_intraday else "CAPITAL_GAINS"

        return FeeBreakdown(
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn_charge,
            gst=gst,
            stt=stt,
            sebi_fees=sebi_fees,
            stamp_duty=stamp_duty,
            total_tax_and_fees=total_tax_and_fees,
            tax_category=tax_category
        )
