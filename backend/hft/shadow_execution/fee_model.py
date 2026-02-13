from dataclasses import dataclass
from enum import Enum
from backend.hft.models.trade_event import FeeBreakdown, TradeType
from backend.hft.config import default_config

class FeeModel:
    """
    Calculates detailed fee structures for Indian Equity (Intraday & Delivery).
    Rates strictly from 2024/2025 standard (NSE/Zerodha).
    Deterministic: Same inputs -> Same outputs.
    """
    
    def __init__(self):
        self.config = default_config.fees

    def calculate_fees(self, price: float, qty: float, side: str, trade_type: TradeType) -> FeeBreakdown:
        """
        Routing method to specific fee logic.
        """
        if trade_type == TradeType.EQUITY_INTRADAY:
            return self._calculate_intraday_fees(price, qty, side)
        elif trade_type == TradeType.EQUITY_DELIVERY:
            return self._calculate_delivery_fees(price, qty, side)
        else:
            raise ValueError(f"Unsupported TradeType: {trade_type}")

    def _calculate_intraday_fees(self, price: float, qty: float, side: str) -> FeeBreakdown:
        """
        Deterministic Intraday Fee Calculation.
        """
        turnover = price * qty
        
        # 1. Brokerage: Lower of (Turnover * PCT) or MAX
        brokerage = min(turnover * self.config.BROKERAGE_PCT, self.config.BROKERAGE_MAX)
        
        # 2. STT: Only on SELL
        stt = 0.0
        if side == "SELL":
            stt = turnover * self.config.STT_INTRADAY_SELL_PCT
            
        # 3. Exchange Txn Charges
        exchange_txn_charge = turnover * self.config.EXCHANGE_TXN_NSE_PCT
        
        # 4. SEBI Charges
        sebi_fees = turnover * self.config.SEBI_CHARGES_PCT
        
        # 5. Stamp Duty: Only on BUY
        stamp_duty = 0.0
        if side == "BUY":
            stamp_duty = turnover * self.config.STAMP_DUTY_INTRADAY_PCT
            
        # 6. GST: 18% on (Brokerage + Exchange + SEBI)
        gst = (brokerage + exchange_txn_charge + sebi_fees) * self.config.GST_PCT
        
        # Total
        total_fees = brokerage + stt + exchange_txn_charge + sebi_fees + stamp_duty + gst
        
        return FeeBreakdown(
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn_charge,
            gst=gst,
            stt=stt,
            sebi_fees=sebi_fees,
            stamp_duty=stamp_duty,
            total_tax_and_fees=total_fees,
            tax_category="BUSINESS_INCOME",
            trade_type="EQUITY_INTRADAY"
        )

    def _calculate_delivery_fees(self, price: float, qty: float, side: str) -> FeeBreakdown:
        """
        Deterministic Delivery Fee Calculation.
        """
        turnover = price * qty
        
        # 1. Brokerage: Usually 0 for delivery on many brokers, but keeping consistent with config if set.
        # Check if config allows 0 for delivery. Assuming standard flat logic or same as config.
        # For Zerodha equity delivery is free (0 brokerage). 
        # But let's stick to config params. If user wants 0, they set pct to 0. 
        # Standard Zerodha is 0. 
        # We will assume config value applies unless explicitly overridden here.
        # Let's assume standard discount broker model: Equity Delivery is 0 brokerage.
        brokerage = 0.0 
        
        # 2. STT: Both Buy and Sell
        stt = turnover * self.config.STT_DELIVERY_PCT
        
        # 3. Exchange Txn Charges
        exchange_txn_charge = turnover * self.config.EXCHANGE_TXN_NSE_PCT
        
        # 4. SEBI Charges
        sebi_fees = turnover * self.config.SEBI_CHARGES_PCT
        
        # 5. Stamp Duty: Only on BUY
        stamp_duty = 0.0
        if side == "BUY":
            stamp_duty = turnover * self.config.STAMP_DUTY_DELIVERY_PCT
            
        # 6. GST: 18% on (Brokerage + Exchange + SEBI)
        gst = (brokerage + exchange_txn_charge + sebi_fees) * self.config.GST_PCT
        
        total_fees = brokerage + stt + exchange_txn_charge + sebi_fees + stamp_duty + gst
        
        return FeeBreakdown(
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn_charge,
            gst=gst,
            stt=stt,
            sebi_fees=sebi_fees,
            stamp_duty=stamp_duty,
            total_tax_and_fees=total_fees,
            tax_category="STCG_OR_LTCG",
            trade_type="EQUITY_DELIVERY"
        )
