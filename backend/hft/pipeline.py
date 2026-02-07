from datetime import datetime
from typing import Optional

from .config import HFTConfig, default_config
from .tick_engine import TickBuffer, Tick
from .microstructure import OrderBookImbalance, OrderBookSnapshot
from .intraday import (
    SpreadTracker, 
    VolumeDeltaTracker, 
    MicroMomentumTracker, 
    RegimeDetector,
    GarchVolatility,
    MarketRegime
)
from .feature_pipeline import FeatureVector
from .risk import RiskGate, RegimeThrottler
from .risk.limits import RiskConfig
from .shadow_execution import ShadowSimulator, ShadowOrder, Side, OrderStatus
from .execution_router import ExecutionRouter, ExecutionMode
from .reporting.karma import KarmaLogger

class HFTPipeline:
    """
    Main orchestration engine for the Shadow HFT system.
    Wires together: Tick Buffer -> Feature Extraction -> Risk -> Execution Router.
    """
    def __init__(self, config: HFTConfig = default_config):
        self.config = config
        
        # 0. Infrastructure
        self.karma = KarmaLogger(output_dir="karma_logs")

        # 1. Data Ingestion
        self.tick_buffer = TickBuffer(max_size=20000, drop_strategy="DROP_OLDEST")
        
        # 2. Feature Trackers
        self.spread_tracker = SpreadTracker(alpha=config.strategy.spread_ema_alpha)
        self.volume_tracker = VolumeDeltaTracker()
        self.momentum_tracker = MicroMomentumTracker(window_ticks=config.strategy.momentum_window_ticks)
        self.regime_detector = RegimeDetector()
        
        # 3. Risk & Execution
        self.risk_gate = RiskGate(constraints=config.risk)
        
        # Risk Config (Day 1 Integration)
        self.risk_config = RiskConfig(
            max_loss_per_min=5000.0,
            max_trades_per_min=20,
            max_drawdown_session=10000.0,
            max_order_qty=100
        )
        
        # Initializing Throttler (Day 1) with None config for now as placeholder or could use new config
        self.throttler = RegimeThrottler(config=None) 
        
        self.simulator = ShadowSimulator(risk_config=self.risk_config, karma_logger=self.karma)
        self.router = ExecutionRouter(simulator=self.simulator)
        
        # Ensure Router is in SHADOW_ONLY mode
        self.router.set_mode(ExecutionMode.SHADOW_ONLY)

    def process_tick(self, tick: Tick) -> None:
        """
        Main Event Loop Step:
        1. Store Tick
        2. Update Features
        3. Check Regime
        4. (Optional) Generate Feature Vector
        5. (Optional) Logic for Trade Generation (Placeholder)
        """
        # 1. Pipeline: Ingest
        if self.tick_buffer.add_tick(tick):
            self.karma.log_tick({"symbol": tick.symbol, "price": tick.price, "vol": tick.volume, "ts": tick.timestamp})
        
        # 2. Pipeline: Feature Update
        spread_metrics = self.spread_tracker.update(tick.timestamp, tick.bid, tick.ask)
        momentum_event = self.momentum_tracker.update(tick.timestamp, tick.price)
        
        # 3. Pipeline: Regime Detection
        # Placeholder volatility value (would normally come from Garch/ATR)
        current_vol = 0.0 
        market_regime = self.regime_detector.update(tick.timestamp, tick.price, current_vol)
        
        # 4. Pipeline: Feature Contract
        features = FeatureVector(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            spread_bps=spread_metrics.spread_relative if spread_metrics else 0.0,
            obi_value=0.0, # Placeholder
            volume_delta=0.0, # Placeholder
            micro_momentum_bps=momentum_event.velocity_bps_per_sec if momentum_event else 0.0,
            volatility_garch=current_vol,
            current_regime=market_regime.regime.value if market_regime else "UNKNOWN"
        )
        
        # 5. Logic: Throttle Check (Example)
        if market_regime:
             # This simple logic demonstrates using the throttling component
             effective_limit = self.throttler.get_effective_limit(market_regime.regime) 
             
    def submit_shadow_order(self, order: ShadowOrder):
        """
        Attempt to route a generated order through the safety gates.
        """
        # 1. Risk Gate Check
        if not self.risk_gate.check_trade(order.symbol, order.quantity, order.side.value):
            print(f"Order {order.order_id} blocked by Risk Gate.")
            return

        # 2. Routing (Guaranteed Shadow)
        try:
            self.router.route_order(order)
            print(f"Order {order.order_id} sent to Shadow Simulator.")
        except Exception as e:
            print(f"CRITICAL: Routing failed: {e}")
