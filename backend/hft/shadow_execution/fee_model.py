from dataclasses import dataclass
from enum import Enum
from backend.hft.models.trade_event import FeeBreakdown, TradeType

class FeeModel:
    """
    Calculates detailed fee structures for Indian Equity (Intraday & Delivery).
    Rates as of late 2024 (NSE/Zerodha standard).
    """
    
    # Constants
    BROKERAGE_PCT = 0.0003  # 0.03%
    BROKERAGE_MAX = 20.0    # Max Rs 20 per order
    
    # STT Rates
    STT_INTRADAY_SELL_PCT = 0.00025 # 0.025% (Sell only)
    STT_DELIVERY_PCT = 0.1          # 0.1% (Buy & Sell)
    
    # Exchange Txn Charges (NSE)
    EXCHANGE_TXN_NSE_PCT = 0.0000325    # 0.00325%
    
    # SEBI Charges
    SEBI_CHARGES_PCT = 0.000001     # Rs 10 per crore (0.0001%) -> actually it's 10/crore = 10/10,000,000 = 0.000001
    
    # Stamp Duty
    STAMP_DUTY_INTRADAY_PCT = 0.00003   # 0.003% (Buy only)
    STAMP_DUTY_DELIVERY_PCT = 0.00015   # 0.015% (Buy only)
    
    # GST
    GST_PCT = 0.18                  # 18%

    def calculate_fees(self, price: float, qty: float, side: str, trade_type: TradeType = TradeType.EQUITY_INTRADAY) -> FeeBreakdown:
        """
        Computes the complete fee breakdown for a single leg of a trade.
        
        Args:
            price: Execution price
            qty: Quantity
            side: 'BUY' or 'SELL' (String matching Side enum value)
            trade_type: TradeType enum
        """
        turnover = price * qty
        is_intraday = (trade_type == TradeType.EQUITY_INTRADAY)
        
        # 1. Brokerage: 0.03% or Rs 20, whichever is lower
        brokerage = min(turnover * self.BROKERAGE_PCT, self.BROKERAGE_MAX)
        
        # 2. STT
        stt = 0.0
        if is_intraday:
            if side == "SELL":
                stt = turnover * self.STT_INTRADAY_SELL_PCT
        else:
            # Delivery: STT on both Buy and Sell
            stt = turnover * self.STT_DELIVERY_PCT

        # 3. Exchange Txn Charges
        exchange_txn_charge = turnover * self.EXCHANGE_TXN_NSE_PCT
        
        # 4. SEBI Charges
        sebi_fees = turnover * self.SEBI_CHARGES_PCT
        
        # 5. Stamp Duty (Buy only)
        stamp_duty = 0.0
        if side == "BUY":
            if is_intraday:
                stamp_duty = turnover * self.STAMP_DUTY_INTRADAY_PCT
            else:
                stamp_duty = turnover * self.STAMP_DUTY_DELIVERY_PCT
            
        # 6. GST: 18% on (Brokerage + Exchange Txn + SEBI)
        gst = (brokerage + exchange_txn_charge + sebi_fees) * self.GST_PCT
        
        total_tax_and_fees = brokerage + stt + exchange_txn_charge + sebi_fees + stamp_duty + gst
        
        # Tax Classification tag
        tax_category = "BUSINESS_INCOME" if is_intraday else "STCG_OR_LTCG"

        return FeeBreakdown(
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn_charge,
            gst=gst,
            stt=stt,
            sebi_fees=sebi_fees,
            stamp_duty=stamp_duty,
            total_tax_and_fees=total_tax_and_fees,
            tax_category=tax_category,
            trade_type=trade_type.value
        )
