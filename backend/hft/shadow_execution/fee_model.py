from backend.hft.models.trade_event import FeeBreakdown, TradeType
from backend.hft.config import default_config


class FeeModel:
    """
    Deterministic Fee Calculator for all supported instrument classes.

    Rates: 2024/2025 NSE/BSE/Zerodha/Finance-Act standards.
    Reference: NSE circular FAOP/49765, Finance Act 2022 (Crypto TDS).

    Routing:
        EQUITY_INTRADAY  → _calculate_intraday_fees
        EQUITY_DELIVERY  → _calculate_delivery_fees
        FUTURES          → _calculate_futures_fees
        OPTIONS          → _calculate_options_fees
        CRYPTO_SPOT      → _calculate_crypto_fees (TAKER by default)
        CRYPTO_FUTURES   → _calculate_crypto_fees (TAKER by default)

    Contract:
        • Identical inputs → identical outputs (no randomness)
        • No runtime API calls
        • All rates sourced exclusively from FeeConfig
    """

    def __init__(self):
        self.cfg = default_config.fees

    # ──────────────────────────────────────────────────────────────────────────
    # Public router
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_fees(
        self,
        price: float,
        qty: float,
        side: str,
        trade_type: TradeType,
        liquidity_flag: str = "TAKER",   # 'MAKER' | 'TAKER' (used for crypto)
        lot_size: int = 1,               # contracts × lot_size = notional units (F&O)
    ) -> FeeBreakdown:
        """
        Routes to the appropriate fee calculation method based on TradeType.

        Args:
            price:          Execution price per unit (INR for equity; premium/unit for options).
            qty:            Number of units (shares / lots / coins).
            side:           'BUY' or 'SELL'.
            trade_type:     One of TradeType enum values.
            liquidity_flag: 'MAKER' or 'TAKER' (relevant for crypto).
            lot_size:       Contract lot size (for F&O turnover calculation).

        Returns:
            FeeBreakdown with all components set deterministically.
        """
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")

        dispatch = {
            TradeType.EQUITY_INTRADAY: lambda: self._calculate_intraday_fees(price, qty, side),
            TradeType.EQUITY_DELIVERY: lambda: self._calculate_delivery_fees(price, qty, side),
            TradeType.FUTURES:         lambda: self._calculate_futures_fees(price, qty, side, lot_size),
            TradeType.OPTIONS:         lambda: self._calculate_options_fees(price, qty, side, lot_size),
            TradeType.CRYPTO_SPOT:     lambda: self._calculate_crypto_fees(price, qty, side, liquidity_flag, "CRYPTO_SPOT"),
            TradeType.CRYPTO_FUTURES:  lambda: self._calculate_crypto_fees(price, qty, side, liquidity_flag, "CRYPTO_FUTURES"),
        }
        if trade_type not in dispatch:
            raise ValueError(f"Unsupported TradeType: {trade_type}")
        return dispatch[trade_type]()

    # ──────────────────────────────────────────────────────────────────────────
    # Equity Intraday
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_intraday_fees(self, price: float, qty: float, side: str) -> FeeBreakdown:
        """
        NSE Capital Market Segment — Intraday (MIS) trades.
        STT only on SELL. Stamp duty only on BUY.
        """
        turnover = price * qty

        brokerage          = min(turnover * self.cfg.BROKERAGE_PCT, self.cfg.BROKERAGE_MAX)
        stt                = turnover * self.cfg.STT_INTRADAY_SELL_PCT if side == "SELL" else 0.0
        exchange_txn       = turnover * self.cfg.EXCHANGE_TXN_NSE_PCT
        sebi_fees          = turnover * self.cfg.SEBI_CHARGES_PCT
        stamp_duty         = turnover * self.cfg.STAMP_DUTY_INTRADAY_PCT if side == "BUY" else 0.0
        gst                = (brokerage + exchange_txn + sebi_fees) * self.cfg.GST_PCT
        total              = brokerage + stt + exchange_txn + sebi_fees + stamp_duty + gst

        return FeeBreakdown(
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn,
            gst=gst,
            stt=stt,
            sebi_fees=sebi_fees,
            stamp_duty=stamp_duty,
            total_tax_and_fees=total,
            tax_category="BUSINESS_INCOME",
            trade_type="EQUITY_INTRADAY",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Equity Delivery
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_delivery_fees(self, price: float, qty: float, side: str) -> FeeBreakdown:
        """
        NSE Capital Market Segment — Delivery (CNC) trades.
        Zero brokerage (discount broker model). STT on both sides.
        """
        turnover = price * qty

        brokerage    = 0.0                                          # Zerodha / discount model
        stt          = turnover * self.cfg.STT_DELIVERY_PCT        # both sides
        exchange_txn = turnover * self.cfg.EXCHANGE_TXN_NSE_PCT
        sebi_fees    = turnover * self.cfg.SEBI_CHARGES_PCT
        stamp_duty   = turnover * self.cfg.STAMP_DUTY_DELIVERY_PCT if side == "BUY" else 0.0
        gst          = (brokerage + exchange_txn + sebi_fees) * self.cfg.GST_PCT
        total        = brokerage + stt + exchange_txn + sebi_fees + stamp_duty + gst

        return FeeBreakdown(
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn,
            gst=gst,
            stt=stt,
            sebi_fees=sebi_fees,
            stamp_duty=stamp_duty,
            total_tax_and_fees=total,
            tax_category="STCG_OR_LTCG",
            trade_type="EQUITY_DELIVERY",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # NSE F&O — Futures
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_futures_fees(
        self, price: float, qty: float, side: str, lot_size: int
    ) -> FeeBreakdown:
        """
        NSE NFO Segment — Futures contracts.
        Turnover = price × qty × lot_size  (notional contract value).
        STT on sell-side only. Stamp duty on buy-side only.
        """
        turnover = price * qty * lot_size

        brokerage    = min(turnover * self.cfg.BROKERAGE_PCT, self.cfg.BROKERAGE_MAX)
        stt          = turnover * self.cfg.STT_FUTURES_SELL_PCT if side == "SELL" else 0.0
        exchange_txn = turnover * self.cfg.EXCHANGE_TXN_NFO_FUT_PCT
        sebi_fees    = turnover * self.cfg.SEBI_CHARGES_PCT
        stamp_duty   = turnover * self.cfg.STAMP_DUTY_FUT_PCT if side == "BUY" else 0.0
        gst          = (brokerage + exchange_txn + sebi_fees) * self.cfg.GST_PCT
        total        = brokerage + stt + exchange_txn + sebi_fees + stamp_duty + gst

        return FeeBreakdown(
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn,
            gst=gst,
            stt=stt,
            sebi_fees=sebi_fees,
            stamp_duty=stamp_duty,
            total_tax_and_fees=total,
            tax_category="BUSINESS_INCOME",
            trade_type="FUTURES",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # NSE F&O — Options
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_options_fees(
        self, price: float, qty: float, side: str, lot_size: int
    ) -> FeeBreakdown:
        """
        NSE NFO Segment — Options contracts.
        Turnover = option premium × qty × lot_size  (premium-based, not notional).
        STT on sell premium only. Exchange charges on premium turnover.
        """
        turnover = price * qty * lot_size   # premium turnover

        brokerage    = min(turnover * self.cfg.BROKERAGE_PCT, self.cfg.BROKERAGE_MAX)
        stt          = turnover * self.cfg.STT_OPTIONS_SELL_PCT if side == "SELL" else 0.0
        exchange_txn = turnover * self.cfg.EXCHANGE_TXN_NFO_OPT_PCT
        sebi_fees    = turnover * self.cfg.SEBI_CHARGES_PCT
        stamp_duty   = turnover * self.cfg.STAMP_DUTY_OPT_PCT if side == "BUY" else 0.0
        gst          = (brokerage + exchange_txn + sebi_fees) * self.cfg.GST_PCT
        total        = brokerage + stt + exchange_txn + sebi_fees + stamp_duty + gst

        return FeeBreakdown(
            brokerage=brokerage,
            exchange_txn_charge=exchange_txn,
            gst=gst,
            stt=stt,
            sebi_fees=sebi_fees,
            stamp_duty=stamp_duty,
            total_tax_and_fees=total,
            tax_category="BUSINESS_INCOME",
            trade_type="OPTIONS",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Crypto (Spot + Futures)
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_crypto_fees(
        self, price: float, qty: float, side: str,
        liquidity_flag: str, trade_type_str: str
    ) -> FeeBreakdown:
        """
        Exchange-agnostic crypto fee model.
        Exchange fee: MAKER or TAKER rate depending on liquidity_flag.
        TDS (Finance Act 2022, s.194S): 1% on sell-side turnover.
        No STT, no stamp duty for crypto.
        GST applied on exchange fee.
        """
        turnover = price * qty

        # Exchange fee
        if liquidity_flag == "MAKER":
            exchange_fee = turnover * self.cfg.CRYPTO_MAKER_PCT
        else:  # TAKER (default conservative)
            exchange_fee = turnover * self.cfg.CRYPTO_TAKER_PCT

        # TDS: 1% on sell-side (withheld by exchange per Finance Act 2022)
        tds = turnover * self.cfg.CRYPTO_TDS_PCT if side == "SELL" else 0.0

        # GST on exchange fee
        gst = exchange_fee * self.cfg.GST_PCT

        # Re-use FeeBreakdown fields semantically:
        #   brokerage        → exchange trading fee
        #   stt              → TDS (largest crypto-specific levy)
        #   exchange_txn_charge, sebi_fees, stamp_duty → zero (not applicable)
        total = exchange_fee + tds + gst

        return FeeBreakdown(
            brokerage=exchange_fee,
            exchange_txn_charge=0.0,
            gst=gst,
            stt=tds,                    # TDS occupies the stt slot for crypto
            sebi_fees=0.0,
            stamp_duty=0.0,
            total_tax_and_fees=total,
            tax_category="SPECULATIVE_VIRTUAL_ASSET",
            trade_type=trade_type_str,
        )
