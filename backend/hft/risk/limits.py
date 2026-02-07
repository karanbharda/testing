from dataclasses import dataclass

@dataclass(frozen=True)
class RiskConfig:
    """
    Configuration for Risk Limits.
    """
    max_loss_per_min: float         # e.g., 5000.0 (INR)
    max_trades_per_min: int         # e.g., 10
    max_drawdown_session: float     # e.g., 20000.0 (INR) - Hard stop for the day
    max_order_qty: int              # e.g., 1000
