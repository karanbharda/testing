from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class KalmanState:
    timestamp: datetime
    observed_price: float
    smoother_price: float   # posterior estimate x̂_t
    error_variance: float   # posterior variance  P_t
    kalman_gain: float      # K_t (diagnostic)


class KalmanSmoother:
    """
    1-D Kalman Filter for streaming price smoothing.
    Assumes a Local Level (Random Walk + Noise) model:

        State equation:    x_t  = x_{t-1}  + w_t,    w_t ~ N(0, Q)
        Observation:       y_t  = x_t      + v_t,    v_t ~ N(0, R)

    Predict step:
        x̂_t|t-1 = x̂_{t-1}
        P_t|t-1  = P_{t-1} + Q

    Update step:
        K_t  = P_t|t-1 / (P_t|t-1 + R)
        x̂_t  = x̂_t|t-1 + K_t × (y_t − x̂_t|t-1)
        P_t  = (1 − K_t) × P_t|t-1

    Cold-start (first observation):
        x̂_0 = y_0  (initialise estimate to first observed price)
        P_0  = 1.0  (high initial uncertainty)

    Fully deterministic: identical price series → identical smoother values.
    No randomness, no wall-clock dependency.
    """

    def __init__(self, process_noise: float = 1e-5, measure_noise: float = 1e-3):
        """
        Args:
            process_noise (Q): Variance of the true latent price dynamics.
                               Smaller → smoother output; larger → follows raw price.
            measure_noise (R): Variance of the market observation noise.
        """
        if process_noise <= 0:
            raise ValueError(f"process_noise (Q) must be > 0, got {process_noise}")
        if measure_noise <= 0:
            raise ValueError(f"measure_noise (R) must be > 0, got {measure_noise}")
        self.Q = process_noise
        self.R = measure_noise
        self._estimate: float | None = None
        self._variance: float | None = None

    def update(self, timestamp: datetime, price: float) -> KalmanState:
        """
        Performs Predict and Update steps for one incoming price observation.

        Args:
            timestamp: Event timestamp (not wall clock).
            price:     Observed market price.

        Returns:
            KalmanState with posterior estimate and diagnostic fields.
        """
        # ── Cold-start ────────────────────────────────────────────────────────
        if self._estimate is None:
            self._estimate = price
            self._variance = 1.0
            gain = 1.0  # K = P/(P+R) = 1/(1+R) ≈ 1 at start with P=1

        # ── Predict step ──────────────────────────────────────────────────────
        p_pred = self._variance + self.Q          # P_t|t-1 = P_{t-1} + Q
        x_pred = self._estimate                   # x̂_t|t-1 = x̂_{t-1} (random walk)

        # ── Update step ───────────────────────────────────────────────────────
        gain = p_pred / (p_pred + self.R)             # K_t
        self._estimate = x_pred + gain * (price - x_pred)    # posterior mean
        self._variance = (1.0 - gain) * p_pred               # posterior variance

        return KalmanState(
            timestamp=timestamp,
            observed_price=price,
            smoother_price=self._estimate,
            error_variance=self._variance,
            kalman_gain=gain,
        )
