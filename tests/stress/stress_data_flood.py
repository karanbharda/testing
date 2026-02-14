"""
STRESS TEST: Data Flood Attack
Feeds 10,000 ticks per second into the system.

Expected Failures:
- Memory spike
- System freeze
- Buffer overflow
"""
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

import time
import random
from datetime import datetime
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.core.karma import KarmaLog
from backend.hft.models.trade_event import TradeType
import psutil
import os as os_module
import random
from datetime import datetime
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.core.karma import KarmaLog
from backend.hft.models.trade_event import TradeType
import psutil
import os as os_module

class DataFloodStressTest:
    def __init__(self):
        self.risk_config = RiskConfig(
            max_trades_per_min=10000,
            max_loss_per_min=1000000.0
        )
        self.risk_gate = RiskGate(self.risk_config)
        self.karma_log = KarmaLog()
        self.simulator = ShadowSimulator(self.risk_gate, self.karma_log)
        
        self.process = psutil.Process(os_module.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def run(self):
        print("=" * 80)
        print("DATA FLOOD STRESS TEST")
        print("=" * 80)
        print(f"Starting: {datetime.now().isoformat()}")
        print(f"Initial Memory: {self.initial_memory:.2f} MB")
        print(f"Target: 10,000 orders in rapid succession")
        print()
        
        start_time = time.time()
        errors = 0
        
        for i in range(10000):
            try:
                order = ShadowOrder(
                    order_id=f"FLOOD_{i:06d}",
                    timestamp=datetime.now(),
                    symbol="RELIANCE",
                    side=Side.BUY if i % 2 == 0 else Side.SELL,
                    quantity=random.uniform(1, 100),
                    limit_price=None,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                price = 2500.0 + random.uniform(-50, 50)
                self.simulator.place_order(order, price, spread_bps=2.0)
                
                # Progress indicator
                if (i + 1) % 1000 == 0:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    delta = current_memory - self.initial_memory
                    print(f"  {i+1:,} orders | Memory: {current_memory:.2f} MB (+{delta:.2f} MB)")
                    
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR at order {i}: {type(e).__name__} - {str(e)[:80]}")
        
        elapsed = time.time() - start_time
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_delta = final_memory - self.initial_memory
        
        print()
        print(f"Completed: {datetime.now().isoformat()}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Throughput: {10000/elapsed:.0f} orders/sec")
        print(f"Final Memory: {final_memory:.2f} MB")
        print(f"Memory Delta: {memory_delta:.2f} MB")
        print(f"Errors: {errors}")
        print(f"Karma Entries: {len(self.karma_log._log)}")
        print()
        
        # Check integrity
        karma_valid = self.karma_log.verify_integrity()
        print(f"Karma Integrity: {'✓ VALID' if karma_valid else '✗ CORRUPTED'}")
        
        print()
        print("=" * 80)
        
        # Determine pass/fail
        if errors > 0 or not karma_valid or memory_delta > 500:
            print("RESULT: ✗ FAILED")
            if errors > 0:
                print(f"  - {errors} errors during execution")
            if not karma_valid:
                print("  - Karma log corrupted")
            if memory_delta > 500:
                print(f"  - Excessive memory usage: {memory_delta:.2f} MB")
            return False
        else:
            print("RESULT: ✓ PASSED")
            print("System survived data flood.")
            return True

if __name__ == "__main__":
    test = DataFloodStressTest()
    success = test.run()
    sys.exit(0 if success else 1)
