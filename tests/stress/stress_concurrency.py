"""
STRESS TEST: Concurrency Attack
Spawns 50 threads to simultaneously call ShadowSimulator.place_order and RiskGate.check_risk.

Expected Failures:
- RuntimeError: deque mutated during iteration (RiskGate)
- Karma Integrity Failure (forked hash chain)
- PnL calculation errors (race condition in positions update)
"""
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

import threading
import time
from datetime import datetime
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.core.karma import KarmaLog
from backend.hft.models.trade_event import TradeType
import time
from datetime import datetime
from backend.hft.shadow_execution.simulator import ShadowSimulator, ShadowOrder, Side, OrderStatus
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.core.karma import KarmaLog
from backend.hft.models.trade_event import TradeType

class ConcurrencyStressTest:
    def __init__(self):
        self.risk_config = RiskConfig(
            max_trades_per_min=1000,
            max_loss_per_min=100000.0
        )
        self.risk_gate = RiskGate(self.risk_config)
        self.karma_log = KarmaLog()
        self.simulator = ShadowSimulator(self.risk_gate, self.karma_log)
        
        self.errors = []
        self.lock = threading.Lock()
        
    def worker_place_order(self, thread_id: int):
        """Worker that places orders"""
        try:
            for i in range(10):
                order = ShadowOrder(
                    order_id=f"T{thread_id:02d}_O{i:03d}",
                    timestamp=datetime.now(),
                    symbol="RELIANCE",
                    side=Side.BUY if i % 2 == 0 else Side.SELL,
                    quantity=10.0,
                    limit_price=2500.0,
                    status=OrderStatus.OPEN,
                    trade_type=TradeType.EQUITY_INTRADAY
                )
                
                result = self.simulator.place_order(
                    order=order,
                    current_market_price=2500.0,
                    spread_bps=2.0
                )
                
                # Small random delay to increase contention
                time.sleep(0.001)
                
        except Exception as e:
            with self.lock:
                self.errors.append({
                    'thread': thread_id,
                    'type': 'place_order',
                    'error': str(e),
                    'exception_type': type(e).__name__
                })
    
    def worker_check_risk(self, thread_id: int):
        """Worker that checks risk"""
        try:
            for i in range(20):
                allowed, reason = self.risk_gate.check_risk(VolatilityRegime.NORMAL)
                time.sleep(0.0005)
        except Exception as e:
            with self.lock:
                self.errors.append({
                    'thread': thread_id,
                    'type': 'check_risk',
                    'error': str(e),
                    'exception_type': type(e).__name__
                })
    
    def run(self):
        print("=" * 80)
        print("CONCURRENCY STRESS TEST")
        print("=" * 80)
        print(f"Starting: {datetime.now().isoformat()}")
        print(f"Threads: 50 (25 order placers + 25 risk checkers)")
        print(f"Operations: ~500 total")
        print()
        
        threads = []
        
        # Spawn 25 order-placing threads
        for i in range(25):
            t = threading.Thread(target=self.worker_place_order, args=(i,))
            threads.append(t)
        
        # Spawn 25 risk-checking threads
        for i in range(25, 50):
            t = threading.Thread(target=self.worker_check_risk, args=(i,))
            threads.append(t)
        
        # Start all threads
        start_time = time.time()
        for t in threads:
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        elapsed = time.time() - start_time
        
        print(f"Completed: {datetime.now().isoformat()}")
        print(f"Elapsed: {elapsed:.2f}s")
        print()
        
        # Check Karma integrity
        print("Checking Karma Log Integrity...")
        karma_valid = self.karma_log.verify_integrity()
        print(f"Karma Integrity: {'✓ VALID' if karma_valid else '✗ CORRUPTED'}")
        print(f"Karma Entries: {len(self.karma_log._log)}")
        print()
        
        # Report errors
        print(f"Errors Detected: {len(self.errors)}")
        if self.errors:
            print("\n--- ERROR DETAILS ---")
            error_types = {}
            for err in self.errors:
                key = err['exception_type']
                error_types[key] = error_types.get(key, 0) + 1
            
            for exc_type, count in error_types.items():
                print(f"  {exc_type}: {count} occurrences")
            
            print("\nSample Errors:")
            for err in self.errors[:5]:
                print(f"  Thread {err['thread']} ({err['type']}): {err['exception_type']} - {err['error'][:100]}")
        
        print()
        print("=" * 80)
        
        # Determine pass/fail
        if self.errors or not karma_valid:
            print("RESULT: ✗ FAILED")
            print("System is NOT thread-safe. Hardening required.")
            return False
        else:
            print("RESULT: ✓ PASSED")
            print("System survived concurrency attack.")
            return True

if __name__ == "__main__":
    test = ConcurrencyStressTest()
    success = test.run()
    sys.exit(0 if success else 1)
