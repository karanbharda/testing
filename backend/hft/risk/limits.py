from dataclasses import dataclass

@dataclass(frozen=True)
class RiskConfig:
    """
    Configuration for Risk Limits.
    """
    max_loss_per_min: float = 1000.0        # e.g., 5000.0 (INR)
    max_trades_per_min: int = 5             # e.g., 10
    max_drawdown_session: float = 5000.0    # e.g., 20000.0 (INR) - Hard stop for the day
    max_order_qty: int = 100                # e.g., 1000
