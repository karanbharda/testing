import sys
import os
import time
from datetime import datetime

# Adjust path to include project root (2 levels up from backend/hft)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.hft.shadow_execution.fee_model import FeeModel
from backend.hft.models.trade_event import TradeType, RiskStopReason
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.tick_engine.tick_buffer import TickBuffer, Tick
from backend.hft.reporting.karma import KarmaLogger

def test_fee_model():
    print("\n--- Testing Fee Model ---")
    fm = FeeModel()
    
    # Test Case 1: Intraday Buy
    # Buy 100 @ 100 = 10,000 Volume
    fees = fm.calculate_fees(100.0, 100, "BUY", trade_type=TradeType.EQUITY_INTRADAY)
    print(f"Intraday Buy 100@100 Fees: {fees.total_tax_and_fees:.4f}")
    print(f"Breakdown: Brok={fees.brokerage}, GST={fees.gst}, STT={fees.stt}, Stamp={fees.stamp_duty}")

    # Test Case 2: Intraday Sell (STT applies)
    fees_sell = fm.calculate_fees(100.0, 100, "SELL", trade_type=TradeType.EQUITY_INTRADAY)
    print(f"Intraday Sell 100@100 Fees: {fees_sell.total_tax_and_fees:.4f} (STT={fees_sell.stt})")

    # Test Case 3: Delivery Buy (High Stamp Duty, No STT on Buy?) -> Delivery has STT on Buy and Sell usually? 
    # Logic in fee_model: Delivery STT on Buy & Sell.
    fees_del = fm.calculate_fees(100.0, 100, "BUY", trade_type=TradeType.EQUITY_DELIVERY)
    print(f"Delivery Buy 100@100 Fees: {fees_del.total_tax_and_fees:.4f} (STT={fees_del.stt})")

def test_risk_hardening():
    print("\n--- Testing Risk Hardening ---")
    config = RiskConfig(
        max_loss_per_min=100.0,
        max_trades_per_min=5,
        max_drawdown_session=1000.0,
        max_order_qty=100
    )
    gate = RiskGate(config)
    
    # 1. Test Trade Throttling
    print("Testing Throttling (Max 5 trades/min)...")
    for i in range(6):
        gate.record_trade()
        allowed, reason = gate.check_risk()
        reason_str = reason.value if reason else "OK"
        print(f"Trade {i+1} Check: {allowed} ({reason_str})")
    
    # 2. Test Loss Limit
    print("Testing Loss Limit (Max 100)...")
    gate = RiskGate(config) # Reset
    gate.record_loss(50.0)
    print(f"Loss 50: {gate.check_risk()[0]}")
    gate.record_loss(60.0) # Total 110
    allowed, reason = gate.check_risk()
    reason_str = reason.value if reason else "OK"
    print(f"Loss 110: {allowed} ({reason_str})")

def test_shadow_simulation():
    print("\n--- Testing Shadow Execution ---")
    config = RiskConfig(max_loss_per_min=1000, max_trades_per_min=100, max_drawdown_session=10000, max_order_qty=100)
    gate = RiskGate(config)
    sim = ShadowSimulator(risk_gate=gate)
    
    order = ShadowOrder(
        order_id="ORD_1",
        timestamp=datetime.now(),
        symbol="INFY",
        side=Side.BUY,
        quantity=100,
        limit_price=1500.0,
        status=OrderStatus.OPEN,
        trade_type=TradeType.EQUITY_INTRADAY
    )
    
    print("Placing Order...")
    res = sim.place_order(order, current_market_price=1500.0, spread_bps=2.0)
    print(f"Order Result: {res}")
    
    if sim.audit_trail.fills:
        fill = sim.audit_trail.fills[0]
        print(f"Fill Price: {fill.price} (Limit 1500)")
        print(f"Fees Paid: {fill.fee_breakdown.total_tax_and_fees}")
    else:
        print("No fill occurred (unexpected for market price match)")

def test_pipeline_integration():
    print("\n--- Testing Pipeline Integration ---")
    from backend.hft.pipeline import HFTPipeline
    
    pipeline = HFTPipeline()
    
    # 1. Process Tick
    tick = Tick(symbol="RELIANCE", price=2500.0, volume=100, timestamp=time.time())
    pipeline.process_tick(tick)
    print(f"Processed Tick. Buffer Size: {pipeline.tick_buffer.size()}")
    
    # 2. Submit Shadow Order
    order = ShadowOrder(
        order_id="PIPE_ORD_1",
        timestamp=datetime.now(),
        symbol="RELIANCE",
        side=Side.BUY,
        quantity=50,
        limit_price=2500.0,
        status=OrderStatus.OPEN
    )
    pipeline.submit_shadow_order(order)
    print("Submitted Order via Pipeline.")

if __name__ == "__main__":
    test_fee_model()
    test_risk_hardening()
    test_shadow_simulation()
    test_pipeline_integration()
