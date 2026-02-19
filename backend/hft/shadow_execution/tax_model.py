from dataclasses import dataclass
from backend.hft.models.trade_event import TradeType


@dataclass(frozen=True)
class TaxClassification:
    category: str           # e.g. 'BUSINESS_INCOME', 'STCG_OR_LTCG', 'SPECULATIVE_VIRTUAL_ASSET'
    description: str        # human-readable rationale


class TaxModel:
    """
    Deterministic Tax Awareness Layer.

    Maps trade types to Indian income tax categories and provides
    deterministic tax amount estimates based on the highest applicable slab.

    IMPORTANT LEGAL DISCLAIMER:
        This model provides illustrative estimates under the highest-bracket
        assumption for simulation purposes only. It does NOT constitute
        financial or tax advice. Actual tax liability depends on the taxpayer's
        total income, carry-forwards, and applicable deductions.

    Tax Classification (FY 2024-25):
        EQUITY_INTRADAY   → Speculative Business Income (Section 43(5))
                            Taxed at slab rate; assumed 30% (highest bracket)
        EQUITY_DELIVERY   → Capital Gains:
                            STCG  (<12 months): 20%    [Finance Act 2024]
                            LTCG  (≥12 months): 12.5%  [Finance Act 2024, >₹1.25L exempt]
        FUTURES           → Non-Speculative Business Income (Section 43(5) proviso)
                            Taxed at slab rate; assumed 30%
        OPTIONS           → Non-Speculative Business Income
                            Taxed at slab rate; assumed 30%
        CRYPTO_SPOT       → Virtual Digital Asset (Section 115BBH)
                            Flat 30%, no deductions, no loss set-off
        CRYPTO_FUTURES    → Virtual Digital Asset (Section 115BBH)
                            Flat 30%, no deductions, no loss set-off

    Negative PnL → zero tax in all cases (losses are not taxable).
    """

    # Deterministic tax rates (constants, never sourced from runtime state)
    _RATES = {
        TradeType.EQUITY_INTRADAY: 0.30,    # Speculative business income, highest slab
        TradeType.EQUITY_DELIVERY: 0.20,    # STCG default (conservative)
        TradeType.FUTURES:         0.30,    # Non-spec business income, highest slab
        TradeType.OPTIONS:         0.30,    # Non-spec business income, highest slab
        TradeType.CRYPTO_SPOT:     0.30,    # Section 115BBH flat rate
        TradeType.CRYPTO_FUTURES:  0.30,    # Section 115BBH flat rate
    }

    _DESCRIPTIONS = {
        TradeType.EQUITY_INTRADAY: (
            "BUSINESS_INCOME",
            "Intraday profits are Speculative Business Income (Section 43(5)). "
            "Taxed at normal slab rates; 30% assumed (highest bracket)."
        ),
        TradeType.EQUITY_DELIVERY: (
            "STCG_OR_LTCG",
            "Delivery profits are Capital Gains. STCG (<12 months): 20%; "
            "LTCG (≥12 months): 12.5% above ₹1.25L exemption. "
            "Model uses 20% (conservative, STCG assumption)."
        ),
        TradeType.FUTURES: (
            "BUSINESS_INCOME",
            "Futures P&L is Non-Speculative Business Income (Section 43(5) proviso). "
            "Taxed at slab rates; 30% assumed (highest bracket)."
        ),
        TradeType.OPTIONS: (
            "BUSINESS_INCOME",
            "Options P&L is Non-Speculative Business Income (Section 43(5) proviso). "
            "Taxed at slab rates; 30% assumed (highest bracket)."
        ),
        TradeType.CRYPTO_SPOT: (
            "SPECULATIVE_VIRTUAL_ASSET",
            "Crypto gains are Virtual Digital Assets (Section 115BBH). "
            "Flat 30% tax. No deductions. No loss set-off against other income."
        ),
        TradeType.CRYPTO_FUTURES: (
            "SPECULATIVE_VIRTUAL_ASSET",
            "Crypto futures gains are Virtual Digital Assets (Section 115BBH). "
            "Flat 30% tax. No deductions. No loss set-off against other income."
        ),
    }

    def classify_trade(self, trade_type: TradeType) -> TaxClassification:
        """
        Returns the tax category and description for the given trade type.
        Deterministic — same TradeType → same output always.
        """
        if trade_type not in self._DESCRIPTIONS:
            return TaxClassification(
                category="UNKNOWN",
                description=f"Unclassified trade type: {trade_type.value}"
            )
        category, description = self._DESCRIPTIONS[trade_type]
        return TaxClassification(category=category, description=description)

    def calculate_tax_amount(
        self,
        gross_pnl: float,
        trade_type: TradeType,
        holding_days: int = 0,
    ) -> float:
        """
        Returns a deterministic estimated tax liability on the given gross P&L.

        Args:
            gross_pnl:    Gross profit/loss before fees (INR).
            trade_type:   Instrument category.
            holding_days: For equity delivery only — adjusts rate to LTCG if ≥365 days.

        Returns:
            Estimated tax amount (INR). Always non-negative.
            Returns 0.0 for losses (negative pnl).

        Tax liability = max(0, gross_pnl) × tax_rate
        """
        if gross_pnl <= 0:
            return 0.0  # No tax on losses

        if trade_type not in self._RATES:
            return 0.0  # Unknown instrument — conservatively no tax

        rate = self._RATES[trade_type]

        # Delivery-specific LTCG adjustment
        if trade_type == TradeType.EQUITY_DELIVERY and holding_days >= 365:
            rate = 0.125    # 12.5% LTCG (Finance Act 2024)

        return round(gross_pnl * rate, 4)
