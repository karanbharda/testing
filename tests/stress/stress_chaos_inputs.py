"""
STRESS TEST: Chaos Inputs Attack
Injects NaN, Inf, None, negative prices, zero quantities.

Expected Failures:
- TypeError
- ValueError
- Corrupt calculations in logs
"""
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

import math
from datetime import datetime
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.core.karma import KarmaLog
from backend.hft.models.trade_event import TradeType
from datetime import datetime
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.core.karma import KarmaLog
from backend.hft.models.trade_event import TradeType

class ChaosInputsStressTest:
    def __init__(self):
        self.risk_config = RiskConfig(
            max_trades_per_min=1000,
            max_loss_per_min=100000.0
        )
        self.risk_gate = RiskGate(self.risk_config)
        self.karma_log = KarmaLog()
        self.simulator = ShadowSimulator(self.risk_gate, self.karma_log)
        
        self.test_cases = []
        self.errors = []
        
    def generate_chaos_inputs(self):
        """Generate malicious/invalid inputs"""
        chaos_prices = [
            float('nan'),
            float('inf'),
            float('-inf'),
            -100.0,
            0.0,
            -0.0,
            None,
            "not_a_number",
            [],
            {}
        ]
        
        chaos_quantities = [
            float('nan'),
            float('inf'),
            -10.0,
            0.0,
            None,
            "invalid",
            -1.0
        ]
        
        chaos_spreads = [
            float('nan'),
            float('inf'),
            -5.0,
            None
        ]
        
        return chaos_prices, chaos_quantities, chaos_spreads
    
    def run(self):
        print("=" * 80)
        print("CHAOS INPUTS STRESS TEST")
        print("=" * 80)
        print(f"Starting: {datetime.now().isoformat()}")
        print()
        
        chaos_prices, chaos_quantities, chaos_spreads = self.generate_chaos_inputs()
        
        test_id = 0
        
        # Test 1: Chaos Prices
        print("Test 1: Chaos Prices")
        for price in chaos_prices:
            test_id += 1
            try:
                order = ShadowOrder(
                    order_id=f"CHAOS_PRICE_{test_id:03d}",
                    timestamp=datetime.now(),
                    symbol="RELIANCE",
                    side=Side.BUY,
                    quantity=10.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                
                result = self.simulator.place_order(order, price, spread_bps=2.0)
                
                # Check if system properly rejected the input
                if result.startswith("REJECTED"):
                    print(f"  ✓ REJECTED price={price}: {result[:60]}")
                else:
                    print(f"  ✗ ACCEPTED invalid price: {price} (result: {result})")
                    self.errors.append({
                        'test': 'chaos_price',
                        'input': price,
                        'error': 'No validation - accepted invalid input'
                    })
                
            except (TypeError, ValueError, AttributeError) as e:
                print(f"  ✓ REJECTED price={price}: {type(e).__name__}")
            except Exception as e:
                print(f"  ? UNEXPECTED ERROR for price={price}: {type(e).__name__} - {str(e)[:60]}")
                self.errors.append({
                    'test': 'chaos_price',
                    'input': price,
                    'error': f'{type(e).__name__}: {str(e)[:60]}'
                })
        
        print()
        
        # Test 2: Chaos Quantities
        print("Test 2: Chaos Quantities")
        for qty in chaos_quantities:
            test_id += 1
            try:
                order = ShadowOrder(
                    order_id=f"CHAOS_QTY_{test_id:03d}",
                    timestamp=datetime.now(),
                    symbol="RELIANCE",
                    side=Side.BUY,
                    quantity=qty,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                result = self.simulator.place_order(order, 2500.0, spread_bps=2.0)
                
                # Check if system properly rejected the input
                if result.startswith("REJECTED"):
                    print(f"  ✓ REJECTED qty={qty}: {result[:60]}")
                else:
                    print(f"  ✗ ACCEPTED invalid quantity: {qty} (result: {result})")
                    self.errors.append({
                        'test': 'chaos_quantity',
                        'input': qty,
                        'error': 'No validation - accepted invalid input'
                    })
                
            except (TypeError, ValueError, AttributeError) as e:
                print(f"  ✓ REJECTED qty={qty}: {type(e).__name__}")
            except Exception as e:
                print(f"  ? UNEXPECTED ERROR for qty={qty}: {type(e).__name__} - {str(e)[:60]}")
                self.errors.append({
                    'test': 'chaos_quantity',
                    'input': qty,
                    'error': f'{type(e).__name__}: {str(e)[:60]}'
                })
        
        print()
        
        # Test 3: Chaos Spreads
        print("Test 3: Chaos Spreads")
        for spread in chaos_spreads:
            test_id += 1
            try:
                order = ShadowOrder(
                    order_id=f"CHAOS_SPREAD_{test_id:03d}",
                    timestamp=datetime.now(),
                    symbol="RELIANCE",
                    side=Side.BUY,
                    quantity=10.0,
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                result = self.simulator.place_order(order, 2500.0, spread_bps=spread)
                
                # Check if system properly rejected the input
                if result.startswith("REJECTED"):
                    print(f"  ✓ REJECTED spread={spread}: {result[:60]}")
                else:
                    print(f"  ✗ ACCEPTED invalid spread: {spread} (result: {result})")
                    self.errors.append({
                        'test': 'chaos_spread',
                        'input': spread,
                        'error': 'No validation - accepted invalid input'
                    })
                
            except (TypeError, ValueError, AttributeError) as e:
                print(f"  ✓ REJECTED spread={spread}: {type(e).__name__}")
            except Exception as e:
                print(f"  ? UNEXPECTED ERROR for spread={spread}: {type(e).__name__} - {str(e)[:60]}")
                self.errors.append({
                    'test': 'chaos_spread',
                    'input': spread,
                    'error': f'{type(e).__name__}: {str(e)[:60]}'
                })
        
        print()
        print("=" * 80)
        print(f"Total Errors: {len(self.errors)}")
        
        if self.errors:
            print("\n--- ERROR SUMMARY ---")
            for err in self.errors:
                print(f"  {err['test']}: {err['input']} -> {err['error']}")
        
        print()
        print("=" * 80)
        
        if self.errors:
            print("RESULT: ✗ FAILED")
            print("System lacks input validation. Hardening required.")
            return False
        else:
            print("RESULT: ✓ PASSED")
            print("System properly validates all inputs.")
            return True

if __name__ == "__main__":
    test = ChaosInputsStressTest()
    success = test.run()
    sys.exit(0 if success else 1)
