import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class GarchOutput:
    timestamp: datetime
    returns_sq: float               # ε²_t (squared return shock this period)
    conditional_variance: float     # σ²_t
    annualized_volatility: float    # √(σ²_t × 252 × 390)  [intraday minute-scale]


class GarchVolatility:
    """
    Real-time GARCH(1,1) volatility estimator using pre-fitted parameters.

    Variance update rule:
        σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

    Where ε_t is the log-return at time t.

    Parameters (ω, α, β) must be provided at construction — they are
    treated as fixed, externally calibrated constants.

    Stationarity check: α + β < 1 is enforced at construction.

    Annualised volatility:
        σ_annual = √(conditional_variance × 252 × 390)
        where 252 = trading days/year and 390 = minutes/day (NSE session).

    Fully deterministic: identical return series → identical outputs.
    """

    TRADING_DAYS_PER_YEAR: int = 252
    MINUTES_PER_DAY: int       = 390   # NSE: 09:15–15:30

    def __init__(self, omega: float, alpha: float, beta: float):
        """
        Args:
            omega: Long-run variance constant (ω > 0).
            alpha: Coefficient on lagged squared return (α ≥ 0).
            beta:  Coefficient on lagged variance (β ≥ 0).

        Stationarity requires: α + β < 1.
        """
        if omega <= 0:
            raise ValueError(f"omega must be > 0, got {omega}")
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha and beta must be >= 0, got α={alpha}, β={beta}")
        if alpha + beta >= 1.0:
            raise ValueError(
                f"GARCH(1,1) non-stationary: α + β = {alpha + beta:.4f} must be < 1"
            )
        self.omega = omega
        self.alpha = alpha
        self.beta  = beta
        # Unconditional (long-run) variance as initial state
        self._variance: float = omega / max(1.0 - alpha - beta, 1e-10)
        self._prev_return_sq: float = 0.0

    def update(self, timestamp: datetime, current_return: float) -> GarchOutput:
        """
        Updates the conditional variance with the latest log-return.

        Args:
            timestamp:       Event timestamp (not wall clock).
            current_return:  Log-return for this interval (ε_t = ln(P_t/P_{t-1})).

        Returns:
            GarchOutput with updated variance and annualised volatility.
        """
        # GARCH(1,1) recursion:  σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1}
        new_variance = (
            self.omega
            + self.alpha * self._prev_return_sq
            + self.beta  * self._variance
        )

        # Annualise: sqrt(σ² × periods_per_year)
        periods_per_year = self.TRADING_DAYS_PER_YEAR * self.MINUTES_PER_DAY
        annualized_vol = math.sqrt(new_variance * periods_per_year)

        # Advance state
        self._prev_return_sq = current_return ** 2
        self._variance       = new_variance

        return GarchOutput(
            timestamp=timestamp,
            returns_sq=self._prev_return_sq,
            conditional_variance=new_variance,
            annualized_volatility=annualized_vol,
        )
