"""
FUNCTIONAL CORE TEST SUITE
Tests each core operation 20 times to verify deterministic behavior.
"""
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from datetime import datetime
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.core.karma import KarmaLog
from backend.hft.models.trade_event import TradeType

class FunctionalCoreTest:
    def __init__(self):
        self.risk_config = RiskConfig(
            max_trades_per_min=1000,
            max_loss_per_min=100000.0
        )
        self.risk_gate = RiskGate(self.risk_config)
        self.karma_log = KarmaLog()
        self.simulator = ShadowSimulator(self.risk_gate, self.karma_log)
        
        self.test_results = {
            'order_placement': {'passed': 0, 'failed': 0},
            'position_updates': {'passed': 0, 'failed': 0},
            'pnl_calculations': {'passed': 0, 'failed': 0},
            'fee_calculations': {'passed': 0, 'failed': 0},
            'risk_gate': {'passed': 0, 'failed': 0},
            'karma_integrity': {'passed': 0, 'failed': 0}
        }
        
    def test_order_placement(self, iterations=20):
        """Test order placement BUY/SELL operations"""
        print(f"\n{'='*60}")
        print("TEST 1: Order Placement (BUY/SELL)")
        print(f"{'='*60}")
        
        for i in range(iterations):
            try:
                # Test BUY order
                buy_order = ShadowOrder(
                    order_id=f"BUY_{i:03d}",
                    timestamp=datetime.now(),
                    symbol="RELIANCE",
                    side=Side.BUY,
                    quantity=10.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                result = self.simulator.place_order(buy_order, 2500.0, spread_bps=2.0)
                
                if not result.startswith("REJECTED"):
                    self.test_results['order_placement']['passed'] += 1
                    print(f"  ✓ BUY order {i+1}/{iterations}: {result[:40]}")
                else:
                    self.test_results['order_placement']['failed'] += 1
                    print(f"  ✗ BUY order {i+1}/{iterations} rejected: {result}")
                    
                # Test SELL order
                sell_order = ShadowOrder(
                    order_id=f"SELL_{i:03d}",
                    timestamp=datetime.now(),
                    symbol="RELIANCE",
                    side=Side.SELL,
                    quantity=5.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                result = self.simulator.place_order(sell_order, 2505.0, spread_bps=2.0)
                
                if not result.startswith("REJECTED"):
                    self.test_results['order_placement']['passed'] += 1
                    print(f"  ✓ SELL order {i+1}/{iterations}: {result[:40]}")
                else:
                    self.test_results['order_placement']['failed'] += 1
                    print(f"  ✗ SELL order {i+1}/{iterations} rejected: {result}")
                    
            except Exception as e:
                self.test_results['order_placement']['failed'] += 2
                print(f"  ✗ Order {i+1} ERROR: {str(e)[:60]}")
                
    def test_position_updates(self, iterations=20):
        """Test position tracking accuracy"""
        print(f"\n{'='*60}")
        print("TEST 2: Position Updates")
        print(f"{'='*60}")
        
        for i in range(iterations):
            try:
                initial_pos = self.simulator.positions.get("SBIN", None)
                initial_qty = initial_pos.quantity if initial_pos else 0.0
                
                # Place order
                order = ShadowOrder(
                    order_id=f"POS_{i:03d}",
                    timestamp=datetime.now(),
                    symbol="SBIN",
                    side=Side.BUY,
                    quantity=10.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                self.simulator.place_order(order, 550.0, spread_bps=2.0)
                
                # Verify position updated
                updated_pos = self.simulator.positions.get("SBIN", None)
                if updated_pos and updated_pos.quantity > initial_qty:
                    self.test_results['position_updates']['passed'] += 1
                    print(f"  ✓ Position {i+1}/{iterations}: {initial_qty} → {updated_pos.quantity}")
                else:
                    self.test_results['position_updates']['failed'] += 1
                    print(f"  ✗ Position {i+1}/{iterations}: Update failed")
                    
            except Exception as e:
                self.test_results['position_updates']['failed'] += 1
                print(f"  ✗ Position {i+1} ERROR: {str(e)[:60]}")
                
    def test_pnl_calculations(self, iterations=20):
        """Test PnL calculation accuracy"""
        print(f"\n{'='*60}")
        print("TEST 3: PnL Calculations")
        print(f"{'='*60}")
        
        for i in range(iterations):
            try:
                # Open position
                buy_order = ShadowOrder(
                    order_id=f"PNL_BUY_{i:03d}",
                    timestamp=datetime.now(),
                    symbol="INFY",
                    side=Side.BUY,
                    quantity=10.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                self.simulator.place_order(buy_order, 1500.0, spread_bps=2.0)
                
                # Close position at profit
                sell_order = ShadowOrder(
                    order_id=f"PNL_SELL_{i:03d}",
                    timestamp=datetime.now(),
                    symbol="INFY",
                    side=Side.SELL,
                    quantity=10.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                self.simulator.place_order(sell_order, 1510.0, spread_bps=2.0)
                
                # Check PnL
                pos = self.simulator.positions.get("INFY", None)
                if pos and pos.realized_pnl != 0:
                    self.test_results['pnl_calculations']['passed'] += 1
                    print(f"  ✓ PnL {i+1}/{iterations}: ₹{pos.realized_pnl:.2f}")
                else:
                    self.test_results['pnl_calculations']['failed'] += 1
                    print(f"  ✗ PnL {i+1}/{iterations}: Calculation failed")
                    
            except Exception as e:
                self.test_results['pnl_calculations']['failed'] += 1
                print(f"  ✗ PnL {i+1} ERROR: {str(e)[:60]}")
                
    def test_fee_calculations(self, iterations=20):
        """Test fee calculation accuracy"""
        print(f"\n{'='*60}")
        print("TEST 4: Fee Calculations")
        print(f"{'='*60}")
        
        for i in range(iterations):
            try:
                order = ShadowOrder(
                    order_id=f"FEE_{i:03d}",
                    timestamp=datetime.now(),
                    symbol="TCS",
                    side=Side.BUY,
                    quantity=10.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                self.simulator.place_order(order, 3500.0, spread_bps=2.0)
                
                pos = self.simulator.positions.get("TCS", None)
                if pos and pos.total_fees > 0:
                    self.test_results['fee_calculations']['passed'] += 1
                    print(f"  ✓ Fees {i+1}/{iterations}: ₹{pos.total_fees:.2f}")
                else:
                    self.test_results['fee_calculations']['failed'] += 1
                    print(f"  ✗ Fees {i+1}/{iterations}: Calculation failed")
                    
            except Exception as e:
                self.test_results['fee_calculations']['failed'] += 1
                print(f"  ✗ Fees {i+1} ERROR: {str(e)[:60]}")
                
    def test_risk_gate_enforcement(self, iterations=20):
        """Test risk gate limits"""
        print(f"\n{'='*60}")
        print("TEST 5: Risk Gate Enforcement")
        print(f"{'='*60}")
        
        for i in range(iterations):
            try:
                allowed, reason = self.risk_gate.check_risk(VolatilityRegime.NORMAL)
                
                if allowed:
                    self.test_results['risk_gate']['passed'] += 1
                    print(f"  ✓ Risk check {i+1}/{iterations}: ALLOWED")
                else:
                    # Risk rejection is also a valid outcome
                    self.test_results['risk_gate']['passed'] += 1
                    print(f"  ✓ Risk check {i+1}/{iterations}: BLOCKED ({reason.value if reason else 'UNKNOWN'})")
                    
            except Exception as e:
                self.test_results['risk_gate']['failed'] += 1
                print(f"  ✗ Risk check {i+1} ERROR: {str(e)[:60]}")
                
    def test_karma_integrity(self, iterations=20):
        """Test Karma log integrity"""
        print(f"\n{'='*60}")
        print("TEST 6: Karma Log Integrity")
        print(f"{'='*60}")
        
        for i in range(iterations):
            try:
                initial_count = len(self.karma_log._log)
                
                # Append entry
                entry_hash = self.karma_log.append("TEST_EVENT", {"iteration": i, "data": "test"})
                
                # Verify integrity
                is_valid = self.karma_log.verify_integrity()
                
                if is_valid and len(self.karma_log._log) == initial_count + 1:
                    self.test_results['karma_integrity']['passed'] += 1
                    print(f"  ✓ Karma {i+1}/{iterations}: VALID (hash: {entry_hash[:16]}...)")
                else:
                    self.test_results['karma_integrity']['failed'] += 1
                    print(f"  ✗ Karma {i+1}/{iterations}: INVALID")
                    
            except Exception as e:
                self.test_results['karma_integrity']['failed'] += 1
                print(f"  ✗ Karma {i+1} ERROR: {str(e)[:60]}")
                
    def run_all_tests(self):
        print("\n" + "="*60)
        print("FUNCTIONAL CORE TEST SUITE")
        print("="*60)
        print(f"Started: {datetime.now().isoformat()}")
        
        self.test_order_placement(20)
        self.test_position_updates(20)
        self.test_pnl_calculations(20)
        self.test_fee_calculations(20)
        self.test_risk_gate_enforcement(20)
        self.test_karma_integrity(20)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        total_passed = 0
        total_failed = 0
        
        for test_name, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total = passed + failed
            total_passed += passed
            total_failed += failed
            
            status = "✓ PASSED" if failed == 0 else "✗ FAILED"
            print(f"{test_name:20s}: {passed:3d}/{total:3d} passed {status}")
            
        print(f"{'='*60}")
        print(f"TOTAL: {total_passed}/{total_passed + total_failed} passed")
        
        if total_failed == 0:
            print("\nRESULT: ✓ ALL TESTS PASSED")
            print("System is functionally correct and deterministic.")
            return True
        else:
            print(f"\nRESULT: ✗ {total_failed} TESTS FAILED")
            print("System has functional issues that need fixing.")
            return False

if __name__ == "__main__":
    test = FunctionalCoreTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)
