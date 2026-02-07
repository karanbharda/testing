import sys
import os
import time
from datetime import datetime

# Adjust path to include project root (2 levels up from backend/hft)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.hft.shadow_execution.fee_model import FeeModel
from backend.hft.risk.throttling import RiskMonitor, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.tick_engine.tick_buffer import TickBuffer, Tick
from backend.hft.reporting.karma import KarmaLogger

def test_fee_model():
    print("\n--- Testing Fee Model ---")
    fm = FeeModel()
    
    # Test Case 1: Small Trade (Brokerage capped at 0.03% or Rs 20)
    # Buy 100 @ 100 = 10,000 Volume
    # Brokerage = 10000 * 0.0003 = 3.0 (Less than 20)
    fees = fm.calculate_fees(100.0, 100, "BUY", is_intraday=True)
    print(f"Buy 100@100 Fees: {fees.total_tax_and_fees:.4f} (Expected approx 3.0 + taxes)")
    print(f"Breakdown: Brok={fees.brokerage}, GST={fees.gst}, Stamp={fees.stamp_duty}")

    # Test Case 2: Large Trade (Brokerage capped at 20)
    # Buy 1000 @ 1000 = 1,000,000 Volume
    # Brokerage = 1,000,000 * 0.0003 = 300 -> Capped at 20? 
    # Wait, 0.03% of 10L is 300. Max is 20 per order.
    fees_large = fm.calculate_fees(1000.0, 1000, "BUY", is_intraday=True)
    print(f"Buy 1K@1K Fees: Brokerage={fees_large.brokerage} (Expected 20.0)")

def test_risk_hardening():
    print("\n--- Testing Risk Hardening ---")
    config = RiskConfig(
        max_loss_per_min=100.0,
        max_trades_per_min=5,
        max_drawdown_session=1000.0,
        max_order_qty=100
    )
    monitor = RiskMonitor(config)
    
    # 1. Test Trade Throttling
    print("Testing Throttling (Max 5 trades/min)...")
    for i in range(6):
        monitor.record_trade()
        allowed, reason = monitor.check_risk()
        print(f"Trade {i+1} Check: {allowed} ({reason})")
    
    # 2. Test Loss Limit
    print("Testing Loss Limit (Max 100)...")
    monitor = RiskMonitor(config) # Reset
    monitor.record_loss(50.0)
    print(f"Loss 50: {monitor.check_risk()[0]}")
    monitor.record_loss(60.0) # Total 110
    allowed, reason = monitor.check_risk()
    print(f"Loss 110: {allowed} ({reason})")

def test_shadow_simulation():
    print("\n--- Testing Shadow Execution ---")
    config = RiskConfig(max_loss_per_min=1000, max_trades_per_min=100, max_drawdown_session=10000, max_order_qty=100)
    sim = ShadowSimulator(config)
    
    order = ShadowOrder(
        order_id="ORD_1",
        timestamp=datetime.now(),
        symbol="INFY",
        side=Side.BUY,
        quantity=100,
        limit_price=1500.0,
        status=OrderStatus.OPEN
    )
    
    print("Placing Order...")
    res = sim.place_order(order, current_market_price=1500.0)
    print(f"Order Result: {res}")
    
    if sim.audit_trail.fills:
        fill = sim.audit_trail.fills[0]
        print(f"Fill Price: {fill.price} (Limit 1500)")
        print(f"Fees Paid: {fill.fee_breakdown.total_tax_and_fees}")
    else:
        print("No fill occurred (unexpected for market price match)")

def test_karma():
    print("\n--- Testing Karma ---")
    karma = KarmaLogger(output_dir="test_karma_logs")
    karma.log_tick({"symbol": "INFY", "price": 1500})
    print("Logged tick to test_karma_logs/")
    karma.close()

from backend.hft.pipeline import HFTPipeline
from backend.hft.tick_engine.tick_buffer import Tick

def test_pipeline_integration():
    print("\n--- Testing Full Pipeline Integration ---")
    pipeline = HFTPipeline()
    
    # 1. Process Tick
    tick = Tick(symbol="RELIANCE", price=2500.0, volume=100, timestamp=time.time())
    pipeline.process_tick(tick)
    print("Processed Tick through Pipeline.")
    
    # 2. Verify Tick Buffer and Karma
    print(f"Buffer Size: {pipeline.tick_buffer.size()}")
    
    # 3. Submit Order
    order = ShadowOrder(
        order_id="PIPE_ORD_1",
        timestamp=datetime.now(),
        symbol="RELIANCE",
        side=Side.BUY,
        quantity=50,
        limit_price=2500.0,
        status=OrderStatus.OPEN
    )
    # Bypass internal logic and call route directly to test integration
    pipeline.submit_shadow_order(order)
    print("Submitted Shadow Order via Pipeline.")

if __name__ == "__main__":
    test_fee_model()
    test_risk_hardening()
    test_shadow_simulation()
    test_karma()
    test_pipeline_integration()
