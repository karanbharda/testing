import sys
import os
import time
from datetime import datetime

# Adjust path to include project root (2 levels up from backend/hft)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.hft.shadow_execution.fee_model import FeeModel
from backend.hft.models.trade_event import TradeType, RiskStopReason, TradeArtifact
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.tick_engine.tick_buffer import TickBuffer, Tick
from backend.hft.core.karma import KarmaLog
from backend.hft.config import default_config, ExecutionMode

def test_mode_guard():
    print("\n--- Testing Mode Guard ---")
    original_mode = default_config.system.execution_mode
    
    try:
        # 1. Test SHADOW_ONLY (Should work)
        print("Testing Shadow Mode (Expected: OK)")
        gate = RiskGate(RiskConfig())
        sim = ShadowSimulator(risk_gate=gate)
        print("Shadow Simulator initialized successfully.")
        
        # 2. Test LIVE (Should Fail)
        print("Testing Live Mode Trigger (Expected: SystemExit)")
        default_config.system.execution_mode = ExecutionMode.LIVE
        try:
             ShadowSimulator(risk_gate=gate)
             print("FAILURE: LIVE mode did not trigger shutdown!")
        except SystemExit:
             print("SUCCESS: LIVE mode triggered SystemExit.")
             
    finally:
        default_config.system.execution_mode = original_mode

def test_fee_model():
    print("\n--- Testing Deterministic Fee Model ---")
    fm = FeeModel()
    
    # Test Case 1: Intraday Buy
    fees = fm.calculate_fees(100.0, 100, "BUY", trade_type=TradeType.EQUITY_INTRADAY)
    print(f"Intraday Buy 100@100 Fees: {fees.total_tax_and_fees:.4f}")
    assert fees.tax_category == "BUSINESS_INCOME"

    # Test Case 2: Delivery Buy (STT check)
    fees_del = fm.calculate_fees(100.0, 100, "BUY", trade_type=TradeType.EQUITY_DELIVERY)
    print(f"Delivery Buy 100@100 Fees: {fees_del.total_tax_and_fees:.4f} (STT={fees_del.stt})")
    assert fees_del.tax_category == "STCG_OR_LTCG"

def test_liquidity_slippage():
    print("\n--- Testing Liquidity Slippage & Partial Fills ---")
    gate = RiskGate(RiskConfig(max_trades_per_min=100))
    karma = KarmaLog()
    sim = ShadowSimulator(risk_gate=gate, karma_log=karma)
    
    # 1. Small Order - Low Slippage
    order_small = ShadowOrder(
        order_id="SMALL_1", timestamp=datetime.now(), symbol="INFY", side=Side.BUY,
        quantity=10, limit_price=None, status=OrderStatus.OPEN, trade_type=TradeType.EQUITY_INTRADAY
    )
    sim.place_order(order_small, current_market_price=1000.0, spread_bps=2.0)
    fill_small = sim.audit_trail.fills[0]
    slippage_small = (fill_small.price - 1000.0) / 1000.0
    print(f"Small Order Slippage: {slippage_small*10000:.2f} bps")

    # 2. Large Order - High Slippage (Fragments)
    sim.set_regime(VolatilityRegime.HIGH)
    order_large = ShadowOrder(
        order_id="LARGE_1", timestamp=datetime.now(), symbol="INFY", side=Side.BUY,
        quantity=500, limit_price=None, status=OrderStatus.OPEN, trade_type=TradeType.EQUITY_INTRADAY
    )
    sim.place_order(order_large, current_market_price=1000.0, spread_bps=5.0)
    
    # Check chunks
    fills = [f for f in sim.audit_trail.fills if f.order_id == "LARGE_1"]
    print(f"Large Order generated {len(fills)} fills (chunks).")
    avg_price = sum(f.price * f.quantity for f in fills) / 500.0
    slippage_large = (avg_price - 1000.0) / 1000.0
    print(f"Large Order Avg Slippage: {slippage_large*10000:.2f} bps")
    
    if slippage_large <= slippage_small:
         print("WARNING: Large order slippage not higher than small order!")
    else:
         print("SUCCESS: Large order slippage > Small order slippage.")

def test_karma_integrity():
    print("\n--- Testing Karma Integrity ---")
    karma = KarmaLog()
    karma.append("TEST_EVENT", {"data": 123})
    karma.append("TEST_EVENT_2", {"data": 456})
    
    if karma.verify_integrity():
        print("Karma Chain Valid.")
    else:
        print("Karma Chain INVALID.")
        
    print(f"Log Size: {len(karma.get_log())}")
    print(f"Latest Hash: {karma.get_log()[-1]['hash']}")

def test_backpressure_hardening():
    print("\n--- Testing Tick Buffer Backpressure ---")
    # Small buffer to force overflow
    buffer = TickBuffer(max_size=10, drop_strategy="DROP_OLDEST")
    
    for i in range(15):
        buffer.add_tick(Tick("INFY", 100.0, 1, time.time()))
        
    metrics = buffer.get_backpressure_metrics()
    print(f"Metrics: {metrics}")
    
    if metrics['dropped_ticks'] == 5 and metrics['is_full']:
        print("SUCCESS: Buffer dropped oldest ticks and reports full.")
    else:
        print("FAILURE: Backpressure metrics incorrect.")

if __name__ == "__main__":
    test_mode_guard()
    test_fee_model()
    test_liquidity_slippage()
    test_karma_integrity()
    test_backpressure_hardening()

