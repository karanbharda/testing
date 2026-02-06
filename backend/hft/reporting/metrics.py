from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass(frozen=True)
class DashboardMetric:
    """
    Represents a single data point for frontend visualization.
    """
    timestamp: datetime
    metric_name: str
    value: float
    tags: dict = None

@dataclass(frozen=True)
class IntradayDashboardState:
    """
    Aggregated state for real-time visualization.
    """
    symbol: str
    last_update: datetime
    
    # 1. Regimes
    current_regime: str
    regime_confidence: float
    
    # 2. Volatility
    current_volatility_bps: float
    
    # 3. Order Book Imbalance
    obi_value: float
    
    # 4. Shadow PnL
    shadow_daily_pnl: float
    shadow_trades_count: int
    
    def get_metrics_stream(self) -> List[DashboardMetric]:
        """
        Flattens the state into a stream of metrics for plotting/timeseries DB.
        """
        ts = self.last_update
        return [
            DashboardMetric(ts, "regime_confidence", self.regime_confidence),
            DashboardMetric(ts, "volatility_bps", self.current_volatility_bps),
            DashboardMetric(ts, "obi", self.obi_value),
            DashboardMetric(ts, "shadow_pnl", self.shadow_daily_pnl)
        ]
