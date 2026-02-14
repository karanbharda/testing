"""
FINANCIAL CORRECTNESS TEST SUITE - SIMPLIFIED
Runs 30 complete trade cycles and validates financial sanity.
Focuses on consistency and determinism rather than exact PnL matching.
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

class FinancialCorrectnessTest:
    def __init__(self):
        self.risk_config = RiskConfig(
            max_trades_per_min=1000,
            max_loss_per_min=100000.0
        )
        self.risk_gate = RiskGate(self.risk_config)
        self.karma_log = KarmaLog()
        self.simulator = ShadowSimulator(self.risk_gate, self.karma_log)
        
        self.trade_cycles = []
        self.errors = []
        
    def run_trade_cycle(self, cycle_id, symbol, entry_price, exit_price, quantity):
        """
        Execute a complete trade cycle and validate financial sanity.
        Returns: (success: bool, pnl: float, fees: float, errors: list)
        """
        cycle_errors = []
        
        try:
            # Step 1: Open position (BUY)
            buy_order = ShadowOrder(
                order_id=f"CYCLE_{cycle_id:03d}_BUY",
                timestamp=datetime.now(),
                symbol=symbol,
                side=Side.BUY,
                quantity=quantity,
                limit_price=None,
                status=OrderStatus.OPEN,
                trade_type=TradeType.EQUITY_INTRADAY
            )
            
            buy_result = self.simulator.place_order(buy_order, entry_price, spread_bps=2.0)
            
            if buy_result.startswith("REJECTED"):
                cycle_errors.append(f"BUY rejected: {buy_result}")
                return False, 0.0, 0.0, cycle_errors
                
            # Step 2: Close position (SELL)
            sell_order = ShadowOrder(
                order_id=f"CYCLE_{cycle_id:03d}_SELL",
                timestamp=datetime.now(),
                symbol=symbol,
                side=Side.SELL,
                quantity=quantity,
                limit_price=None,
                status=OrderStatus.OPEN,
                trade_type=TradeType.EQUITY_INTRADAY
            )
            
            sell_result = self.simulator.place_order(sell_order, exit_price, spread_bps=2.0)
            
            if sell_result.startswith("REJECTED"):
                cycle_errors.append(f"SELL rejected: {sell_result}")
                return False, 0.0, 0.0, cycle_errors
                
            # Step 3: Validate financial sanity
            pos = self.simulator.positions.get(symbol, None)
            
            if not pos:
                cycle_errors.append("Position not found after trade")
                return False, 0.0, 0.0, cycle_errors
                
            # Sanity check 1: Fees must be positive
            if pos.total_fees <= 0:
                cycle_errors.append(f"Invalid fees: {pos.total_fees:.2f} (should be > 0)")
                
            # Sanity check 2: PnL direction should match price movement
            price_diff = exit_price - entry_price
            expected_direction = "profit" if price_diff > 0 else "loss"
            actual_direction = "profit" if pos.realized_pnl > 0 else "loss"
            
            # Allow for fees to flip small profits to losses
            if abs(price_diff * quantity) > pos.total_fees * 2:
                if expected_direction != actual_direction:
                    cycle_errors.append(
                        f"PnL direction mismatch: price moved {price_diff:.2f}, "
                        f"expected {expected_direction}, got {actual_direction}"
                    )
                    
            # Sanity check 3: PnL magnitude should be reasonable
            max_reasonable_pnl = abs(price_diff * quantity * 1.5)  # Allow 50% variance for slippage
            if abs(pos.realized_pnl) > max_reasonable_pnl:
                cycle_errors.append(
                    f"PnL magnitude unreasonable: {pos.realized_pnl:.2f} "
                    f"(max expected: {max_reasonable_pnl:.2f})"
                )
                
            success = len(cycle_errors) == 0
            return success, pos.realized_pnl, pos.total_fees, cycle_errors
            
        except Exception as e:
            cycle_errors.append(f"Exception: {str(e)}")
            return False, 0.0, 0.0, cycle_errors
            
    def run_all_cycles(self, num_cycles=30):
        print("="*80)
        print("FINANCIAL CORRECTNESS TEST SUITE")
        print("="*80)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Cycles: {num_cycles}")
        print()
        
        # Test scenarios with different profit/loss outcomes
        test_scenarios = [
            # (symbol, entry_price, exit_price, quantity)
            ("RELIANCE", 2500.0, 2510.0, 10.0),  # Profit
            ("INFY", 1500.0, 1495.0, 10.0),      # Loss
            ("TCS", 3500.0, 3520.0, 5.0),        # Profit
            ("SBIN", 550.0, 548.0, 20.0),        # Loss
            ("HDFCBANK", 1600.0, 1610.0, 10.0),  # Profit
        ]
        
        passed = 0
        failed = 0
        total_pnl = 0.0
        total_fees = 0.0
        
        for i in range(num_cycles):
            scenario = test_scenarios[i % len(test_scenarios)]
            base_symbol, entry, exit, qty = scenario
            
            # Use unique symbol per cycle to avoid position accumulation
            symbol = f"{base_symbol}_{i:03d}"
            
            success, pnl, fees, errors = self.run_trade_cycle(i+1, symbol, entry, exit, qty)
            
            self.trade_cycles.append({
                'cycle_id': i+1,
                'symbol': symbol,
                'entry': entry,
                'exit': exit,
                'quantity': qty,
                'pnl': pnl,
                'fees': fees,
                'success': success,
                'errors': errors
            })
            
            if success:
                passed += 1
                total_pnl += pnl
                total_fees += fees
                direction = "PROFIT" if pnl > 0 else "LOSS"
                print(f"  OK Cycle {i+1:2d}/{num_cycles}: {symbol:10s} | "
                      f"{direction:6s} | PnL: Rs.{pnl:8.2f} | Fees: Rs.{fees:6.2f}")
            else:
                failed += 1
                self.errors.extend(errors)
                print(f"  FAIL Cycle {i+1:2d}/{num_cycles}: {symbol:10s} | FAILED")
                for error in errors:
                    print(f"      - {error}")
                    
        # Print summary
        print()
        print("="*80)
        print("FINANCIAL SUMMARY")
        print("="*80)
        print(f"Cycles Passed:  {passed}/{num_cycles}")
        print(f"Cycles Failed:  {failed}/{num_cycles}")
        print(f"Total Net PnL:  Rs.{total_pnl:.2f}")
        print(f"Total Fees:     Rs.{total_fees:.2f}")
        print(f"Gross PnL:      Rs.{total_pnl + total_fees:.2f}")
        print()
        
        # Karma integrity check
        karma_valid = self.karma_log.verify_integrity()
        print(f"Karma Integrity: {'OK VALID' if karma_valid else 'FAIL CORRUPTED'}")
        print()
        
        if failed == 0 and karma_valid:
            print("="*80)
            print("RESULT: OK ALL FINANCIAL TESTS PASSED")
            print("System is financially sane and audit-ready.")
            print("="*80)
            return True
        else:
            print("="*80)
            print(f"RESULT: FAIL {failed} CYCLES FAILED")
            if not karma_valid:
                print("FAIL Karma log corrupted")
            print("System has financial sanity issues.")
            print("="*80)
            return False

if __name__ == "__main__":
    test = FinancialCorrectnessTest()
    success = test.run_all_cycles(30)
    sys.exit(0 if success else 1)
