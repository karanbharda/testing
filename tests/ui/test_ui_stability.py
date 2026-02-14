"""
UI STABILITY TEST
Verifies that the HFT system can be safely integrated with UI without crashes.
Tests basic integration points and error handling.
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

class UIStabilityTest:
    def __init__(self):
        self.risk_config = RiskConfig(
            max_trades_per_min=1000,
            max_loss_per_min=100000.0
        )
        self.risk_gate = RiskGate(self.risk_config)
        self.karma_log = KarmaLog()
        self.simulator = ShadowSimulator(self.risk_gate, self.karma_log)
        
        self.test_results = {
            'position_query': {'passed': 0, 'failed': 0},
            'audit_trail_access': {'passed': 0, 'failed': 0},
            'karma_export': {'passed': 0, 'failed': 0},
            'error_handling': {'passed': 0, 'failed': 0},
        }
        
    def test_position_query(self):
        """Test UI can safely query positions"""
        print("\nTEST 1: Position Query Safety")
        print("="*60)
        
        try:
            # Place some orders
            for i in range(5):
                order = ShadowOrder(
                    order_id=f"UI_POS_{i:03d}",
                    timestamp=datetime.now(),
                    symbol=f"STOCK_{i}",
                    side=Side.BUY,
                    quantity=10.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                self.simulator.place_order(order, 1000.0 + i*10, spread_bps=2.0)
                
            # Query positions (simulating UI access)
            positions = dict(self.simulator.positions)
            
            if len(positions) >= 5:
                self.test_results['position_query']['passed'] += 1
                print(f"  OK: Retrieved {len(positions)} positions")
            else:
                self.test_results['position_query']['failed'] += 1
                print(f"  FAIL: Expected >= 5 positions, got {len(positions)}")
                
        except Exception as e:
            self.test_results['position_query']['failed'] += 1
            print(f"  FAIL: Exception during position query: {str(e)[:60]}")
            
    def test_audit_trail_access(self):
        """Test UI can safely access audit trail"""
        print("\nTEST 2: Audit Trail Access")
        print("="*60)
        
        try:
            # Access audit trail
            orders = list(self.simulator.audit_trail.orders)
            fills = list(self.simulator.audit_trail.fills)
            
            if len(orders) > 0:
                self.test_results['audit_trail_access']['passed'] += 1
                print(f"  OK: Retrieved {len(orders)} orders, {len(fills)} fills")
            else:
                self.test_results['audit_trail_access']['failed'] += 1
                print(f"  FAIL: No orders in audit trail")
                
        except Exception as e:
            self.test_results['audit_trail_access']['failed'] += 1
            print(f"  FAIL: Exception during audit trail access: {str(e)[:60]}")
            
    def test_karma_export(self):
        """Test UI can export Karma log"""
        print("\nTEST 3: Karma Log Export")
        print("="*60)
        
        try:
            # Export Karma log
            log_entries = list(self.karma_log._log)
            
            # Verify integrity
            is_valid = self.karma_log.verify_integrity()
            
            if is_valid and len(log_entries) > 0:
                self.test_results['karma_export']['passed'] += 1
                print(f"  OK: Exported {len(log_entries)} Karma entries (integrity: VALID)")
            else:
                self.test_results['karma_export']['failed'] += 1
                print(f"  FAIL: Karma export failed or invalid")
                
        except Exception as e:
            self.test_results['karma_export']['failed'] += 1
            print(f"  FAIL: Exception during Karma export: {str(e)[:60]}")
            
    def test_error_handling(self):
        """Test UI error handling for invalid operations"""
        print("\nTEST 4: Error Handling")
        print("="*60)
        
        error_cases = [
            ("Invalid price (NaN)", float('nan')),
            ("Invalid price (negative)", -100.0),
            ("Invalid price (zero)", 0.0),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, invalid_price in error_cases:
            try:
                order = ShadowOrder(
                    order_id=f"UI_ERR_{test_name}",
                    timestamp=datetime.now(),
                    symbol="TEST",
                    side=Side.BUY,
                    quantity=10.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                result = self.simulator.place_order(order, invalid_price, spread_bps=2.0)
                
                # Should be rejected
                if result.startswith("REJECTED"):
                    passed += 1
                    print(f"  OK: {test_name} properly rejected")
                else:
                    failed += 1
                    print(f"  FAIL: {test_name} was accepted (should reject)")
                    
            except Exception as e:
                # Exception is also acceptable (defensive programming)
                passed += 1
                print(f"  OK: {test_name} raised exception (safe)")
                
        self.test_results['error_handling']['passed'] = passed
        self.test_results['error_handling']['failed'] = failed
        
    def run_all_tests(self):
        print("="*80)
        print("UI STABILITY TEST SUITE")
        print("="*80)
        print(f"Started: {datetime.now().isoformat()}")
        print()
        
        self.test_position_query()
        self.test_audit_trail_access()
        self.test_karma_export()
        self.test_error_handling()
        
        # Print summary
        print()
        print("="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        total_passed = 0
        total_failed = 0
        
        for test_name, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total = passed + failed
            total_passed += passed
            total_failed += failed
            
            status = "OK PASSED" if failed == 0 else "FAIL FAILED"
            print(f"{test_name:25s}: {passed:2d}/{total:2d} passed {status}")
            
        print("="*80)
        print(f"TOTAL: {total_passed}/{total_passed + total_failed} passed")
        print()
        
        if total_failed == 0:
            print("RESULT: OK ALL UI TESTS PASSED")
            print("System is UI-safe and integration-ready.")
            return True
        else:
            print(f"RESULT: FAIL {total_failed} TESTS FAILED")
            print("System has UI integration issues.")
            return False

if __name__ == "__main__":
    test = UIStabilityTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)
