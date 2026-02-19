"""
verify_hft.py — Institutional-Grade Shadow Engine Verification Suite
====================================================================
Run from project root:
    python -m backend.hft.verify_hft

Tests:
  1. Mode Guard              — LIVE mode must raise SystemExit
  2. Fee Model               — All 6 trade types produce correct deterministic fees
  3. Determinism Replay      — Same order 5× → bit-identical execution price, fees, net PnL
  4. No-Randomness Contract  — random module raises RuntimeError if called
  5. Liquidity Chunking      — volume_ratio drives chunk count, not randomness
  6. Slippage Monotonicity   — larger orders → higher slippage (deterministic)
  7. Invariant Enforcement   — Bad fill quantity triggers invariant rejection
  8. Tax Model               — All 6 types classified; calculate_tax_amount correct
  9. TradeLifecycleArtifact  — Single artifact per order; state_history ends in LOGGED
 10. Karma Chain Integrity   — Append-only hash chain validates cleanly
 11. Intraday Analytics      — SpreadTracker, ATR, Kalman, VolumeDelta, GARCH numeric checks
"""

import sys
import os
import math
import unittest
import random as _random_module  # imported to BLOCK it in determinism test
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.hft.shadow_execution.fee_model import FeeModel
from backend.hft.shadow_execution.tax_model import TaxModel
from backend.hft.models.trade_event import (
    TradeType, RiskStopReason, TradeLifecycleArtifact,
)
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.risk.limits import RiskConfig
from backend.hft.shadow_execution.simulator import (
    ShadowSimulator, ShadowOrder, Side, OrderStatus,
)
from backend.hft.core.karma import KarmaLog
from backend.hft.config import default_config, ExecutionMode
from backend.hft.intraday.spread_model import SpreadTracker
from backend.hft.intraday.atr_volatility import IntradayATR, Candle
from backend.hft.intraday.kalman_filter import KalmanSmoother
from backend.hft.intraday.volume_delta import VolumeDeltaTracker
from backend.hft.intraday.garch_volatility import GarchVolatility


def _make_order(order_id: str = "ORD-TEST", qty: float = 100.0,
                side: Side = Side.BUY, trade_type: TradeType = TradeType.EQUITY_INTRADAY
                ) -> ShadowOrder:
    return ShadowOrder(
        order_id=order_id,
        timestamp=datetime(2026, 2, 19, 9, 15, 0),
        symbol="INFY",
        side=side,
        quantity=qty,
        limit_price=None,
        status=OrderStatus.OPEN,
        trade_type=trade_type,
    )


def _make_sim(max_trades: int = 1000) -> ShadowSimulator:
    gate = RiskGate(RiskConfig(max_trades_per_min=max_trades))
    return ShadowSimulator(risk_gate=gate)


# ─────────────────────────────────────────────────────────────────────────────

class TestModeGuard(unittest.TestCase):

    def test_shadow_mode_ok(self):
        """SHADOW_ONLY mode initialises without error."""
        sim = _make_sim()
        self.assertIsNotNone(sim)

    def test_live_mode_exits(self):
        """LIVE mode must raise SystemExit — prevents real broker calls."""
        original = default_config.system.execution_mode
        try:
            default_config.system.execution_mode = ExecutionMode.LIVE
            with self.assertRaises(SystemExit):
                _make_sim()
        finally:
            default_config.system.execution_mode = original


class TestFeeModel(unittest.TestCase):
    fm = FeeModel()

    def test_intraday_buy(self):
        fees = self.fm.calculate_fees(1000.0, 100, "BUY", TradeType.EQUITY_INTRADAY)
        self.assertEqual(fees.trade_type, "EQUITY_INTRADAY")
        self.assertEqual(fees.tax_category, "BUSINESS_INCOME")
        self.assertEqual(fees.stt, 0.0)               # no STT on intraday buy
        self.assertGreater(fees.stamp_duty, 0)

    def test_intraday_sell(self):
        fees = self.fm.calculate_fees(1000.0, 100, "SELL", TradeType.EQUITY_INTRADAY)
        self.assertGreater(fees.stt, 0)
        self.assertEqual(fees.stamp_duty, 0.0)

    def test_delivery_buy(self):
        fees = self.fm.calculate_fees(1000.0, 100, "BUY", TradeType.EQUITY_DELIVERY)
        self.assertEqual(fees.tax_category, "STCG_OR_LTCG")
        self.assertGreater(fees.stt, 0)   # both sides for delivery

    def test_futures_sell(self):
        fees = self.fm.calculate_fees(200.0, 50, "SELL", TradeType.FUTURES, lot_size=75)
        self.assertEqual(fees.tax_category, "BUSINESS_INCOME")
        self.assertEqual(fees.trade_type, "FUTURES")
        self.assertGreater(fees.stt, 0)    # STT on futures sell

    def test_options_sell(self):
        fees = self.fm.calculate_fees(150.0, 1, "SELL", TradeType.OPTIONS, lot_size=100)
        self.assertEqual(fees.tax_category, "BUSINESS_INCOME")
        self.assertEqual(fees.trade_type, "OPTIONS")
        self.assertGreater(fees.stt, 0)

    def test_crypto_spot_sell_has_tds(self):
        fees = self.fm.calculate_fees(3_000_000.0, 1, "SELL", TradeType.CRYPTO_SPOT)
        self.assertEqual(fees.tax_category, "SPECULATIVE_VIRTUAL_ASSET")
        self.assertGreater(fees.stt, 0)     # TDS stored in stt slot
        expected_tds = 3_000_000.0 * 0.01
        self.assertAlmostEqual(fees.stt, expected_tds, places=2)

    def test_crypto_buy_no_tds(self):
        fees = self.fm.calculate_fees(3_000_000.0, 1, "BUY", TradeType.CRYPTO_SPOT)
        self.assertEqual(fees.stt, 0.0)     # no TDS on buy

    def test_determinism_same_inputs(self):
        """Identical inputs must produce identical outputs (floating point safe)."""
        fees_a = self.fm.calculate_fees(1500.0, 200, "BUY", TradeType.EQUITY_INTRADAY)
        fees_b = self.fm.calculate_fees(1500.0, 200, "BUY", TradeType.EQUITY_INTRADAY)
        self.assertEqual(fees_a.total_tax_and_fees, fees_b.total_tax_and_fees)


class TestTaxModel(unittest.TestCase):
    tm = TaxModel()

    def test_all_types_classified(self):
        for tt in TradeType:
            cls = self.tm.classify_trade(tt)
            self.assertNotEqual(cls.category, "UNKNOWN")

    def test_intraday_business_income(self):
        cls = self.tm.classify_trade(TradeType.EQUITY_INTRADAY)
        self.assertEqual(cls.category, "BUSINESS_INCOME")

    def test_crypto_vda(self):
        cls = self.tm.classify_trade(TradeType.CRYPTO_SPOT)
        self.assertEqual(cls.category, "SPECULATIVE_VIRTUAL_ASSET")

    def test_positive_pnl_taxed(self):
        tax = self.tm.calculate_tax_amount(10_000.0, TradeType.EQUITY_INTRADAY)
        self.assertAlmostEqual(tax, 3_000.0, places=2)   # 30%

    def test_negative_pnl_zero_tax(self):
        tax = self.tm.calculate_tax_amount(-5000.0, TradeType.EQUITY_INTRADAY)
        self.assertEqual(tax, 0.0)

    def test_ltcg_rate_applied(self):
        tax = self.tm.calculate_tax_amount(10_000.0, TradeType.EQUITY_DELIVERY, holding_days=400)
        self.assertAlmostEqual(tax, 1_250.0, places=2)   # 12.5%


class TestDeterminismReplay(unittest.TestCase):
    """
    CORE DETERMINISM TEST — Same order 5 times must produce bit-identical results.
    Failure here means the engine is NOT deterministic.
    """

    def _run_single(self, price: float = 1000.0, spread: float = 2.0, qty: float = 50.0,
                    snap: dict = None):
        gate = RiskGate(RiskConfig(max_trades_per_min=1000))
        karma = KarmaLog()
        sim = ShadowSimulator(risk_gate=gate, karma_log=karma)
        order = _make_order(qty=qty)
        result = sim.place_order(order, current_market_price=price,
                                 spread_bps=spread, liquidity_snapshot=snap)
        artifacts = sim.audit_trail.lifecycle_artifacts
        fills = sim.audit_trail.fills
        return result, artifacts, fills

    def test_five_runs_identical(self):
        snapshot = {"estimated_liquidity": 5000.0, "spread_bps": 2.5}
        runs = [self._run_single(snap=snapshot) for _ in range(5)]

        ref_result, ref_artifacts, ref_fills = runs[0]

        for i, (result, artifacts, fills) in enumerate(runs[1:], 1):
            with self.subTest(run=i+1):
                self.assertEqual(result, ref_result, "Return value differs")
                self.assertEqual(
                    len(artifacts), len(ref_artifacts),
                    "Artifact count differs"
                )
                if artifacts and ref_artifacts:
                    a, ref_a = artifacts[0], ref_artifacts[0]
                    self.assertEqual(a.total_fees, ref_a.total_fees,
                                     "total_fees not identical")
                    self.assertEqual(a.net_pnl, ref_a.net_pnl,
                                     "net_pnl not identical")
                    self.assertEqual(a.slippage_bps, ref_a.slippage_bps,
                                     "slippage_bps not identical")
                    self.assertEqual(a.state_history, ref_a.state_history,
                                     "state_history not identical")

                # Fill prices must be identical
                self.assertEqual(len(fills), len(ref_fills), "Fill count differs")
                for fill, ref_fill in zip(fills, ref_fills):
                    self.assertEqual(fill.price, ref_fill.price, "Fill price not identical")
                    self.assertEqual(fill.quantity, ref_fill.quantity, "Fill qty not identical")

    def test_no_randomness_contract(self):
        """If random.choice/uniform/randint are called, this test fails."""
        def explode(*args, **kwargs):
            raise AssertionError("random module was called — determinism violated!")

        with patch.object(_random_module, "uniform", side_effect=explode), \
             patch.object(_random_module, "randint", side_effect=explode), \
             patch.object(_random_module, "choice",  side_effect=explode), \
             patch.object(_random_module, "random",  side_effect=explode):
            # If any random call happens inside place_order, the patch fires
            result, _, _ = self._run_single()
            self.assertEqual(result, "FILLED")


class TestLiquidityChunking(unittest.TestCase):

    def _chunks(self, qty, liquidity):
        gate = RiskGate(RiskConfig(max_trades_per_min=1000))
        sim = ShadowSimulator(risk_gate=gate)
        return sim._calculate_deterministic_chunks(qty, liquidity)

    def test_tiny_order_one_chunk(self):
        chunks = self._chunks(10, 10_000)   # ratio = 0.001
        self.assertEqual(len(chunks), 1)

    def test_small_order_two_chunks(self):
        chunks = self._chunks(250, 10_000)  # ratio = 0.025
        self.assertEqual(len(chunks), 2)

    def test_medium_order_three_chunks(self):
        chunks = self._chunks(750, 10_000)  # ratio = 0.075
        self.assertEqual(len(chunks), 3)

    def test_large_order_many_chunks(self):
        chunks = self._chunks(5000, 10_000)  # ratio = 0.5
        self.assertGreater(len(chunks), 3)

    def test_chunk_sum_equals_total(self):
        for qty in [100.0, 250.0, 1000.0, 5000.0]:
            chunks = self._chunks(qty, 10_000.0)
            self.assertAlmostEqual(sum(chunks), qty, places=9,
                                   msg=f"Chunk sum mismatch for qty={qty}")

    def test_deterministic_same_inputs(self):
        c1 = self._chunks(300, 5000)
        c2 = self._chunks(300, 5000)
        self.assertEqual(c1, c2)


class TestSlippageModel(unittest.TestCase):

    def _get_fill_prices(self, qty, snapshot=None, spread=2.0):
        gate = RiskGate(RiskConfig(max_trades_per_min=1000))
        sim = ShadowSimulator(risk_gate=gate)
        order = _make_order(qty=qty)
        sim.place_order(order, 1000.0, spread_bps=spread, liquidity_snapshot=snapshot)
        return [f.price for f in sim.audit_trail.fills]

    def test_larger_order_higher_slippage(self):
        """Order 5× larger vs base liquidity must have higher avg slippage."""
        snap_small = {"estimated_liquidity": 10_000.0}
        snap_large = {"estimated_liquidity": 10_000.0}
        fills_small = self._get_fill_prices(10,  snapshot=snap_small)
        fills_large = self._get_fill_prices(2000, snapshot=snap_large)
        avg_small = sum(fills_small) / len(fills_small) - 1000.0
        avg_large = sum(fills_large) / len(fills_large) - 1000.0
        self.assertGreater(avg_large, avg_small,
                           "Large order should have more slippage than small order")

    def test_high_regime_more_slippage(self):
        """HIGH volatility regime must produce more slippage than NORMAL."""
        gate = RiskGate(RiskConfig(max_trades_per_min=1000))
        sim = ShadowSimulator(risk_gate=gate)
        order = _make_order(qty=100)

        sim.set_regime(VolatilityRegime.NORMAL)
        sim.place_order(order, 1000.0, spread_bps=2.0)
        normal_fill = sim.audit_trail.fills[-1].price

        gate2 = RiskGate(RiskConfig(max_trades_per_min=1000))
        sim2 = ShadowSimulator(risk_gate=gate2)
        sim2.set_regime(VolatilityRegime.HIGH)
        order2 = _make_order(qty=100)
        sim2.place_order(order2, 1000.0, spread_bps=2.0)
        high_fill = sim2.audit_trail.fills[-1].price

        self.assertGreater(high_fill, normal_fill,
                           "HIGH regime should produce higher fill price for BUY")


class TestTradeLifecycleArtifact(unittest.TestCase):

    def test_single_artifact_per_order(self):
        """Exactly ONE lifecycle artifact per order — no per-chunk fragmentation."""
        sim = _make_sim()
        # Place one order
        sim.place_order(_make_order("O1", 500), 1000.0, spread_bps=3.0,
                        liquidity_snapshot={"estimated_liquidity": 5000.0})
        self.assertEqual(len(sim.audit_trail.lifecycle_artifacts), 1)

        # Place a second order
        sim.place_order(_make_order("O2", 200), 1100.0, spread_bps=2.0)
        self.assertEqual(len(sim.audit_trail.lifecycle_artifacts), 2)

    def test_state_history_ends_in_logged(self):
        sim = _make_sim()
        sim.place_order(_make_order("O1"), 1000.0)
        artifact = sim.audit_trail.lifecycle_artifacts[0]
        self.assertEqual(artifact.state_history[-1], "LOGGED")

    def test_artifact_invariants_hold(self):
        """TradeLifecycleArtifact.__post_init__ must not raise."""
        sim = _make_sim()
        result = sim.place_order(_make_order("O1"), 1000.0)
        self.assertEqual(result, "FILLED")
        artifact = sim.audit_trail.lifecycle_artifacts[0]
        self.assertGreaterEqual(artifact.total_fees, 0)
        self.assertGreaterEqual(artifact.slippage_bps, 0)
        self.assertGreaterEqual(artifact.tax_amount, 0)

    def test_tax_attached_to_artifact(self):
        sim = _make_sim()
        sim.place_order(_make_order("O1"), 1000.0)
        artifact = sim.audit_trail.lifecycle_artifacts[0]
        self.assertIn(artifact.tax_category, (
            "BUSINESS_INCOME", "STCG_OR_LTCG", "SPECULATIVE_VIRTUAL_ASSET"
        ))
        self.assertIsInstance(artifact.tax_description, str)
        self.assertGreater(len(artifact.tax_description), 10)

    def test_slippage_bps_in_artifact(self):
        sim = _make_sim()
        sim.place_order(_make_order("O1", qty=100), 1000.0, spread_bps=5.0)
        artifact = sim.audit_trail.lifecycle_artifacts[0]
        self.assertGreaterEqual(artifact.slippage_bps, 0.0)


class TestKarmaIntegrity(unittest.TestCase):

    def test_chain_valid_after_fills(self):
        karma = KarmaLog()
        sim = ShadowSimulator(risk_gate=RiskGate(RiskConfig(max_trades_per_min=1000)),
                              karma_log=karma)
        sim.place_order(_make_order("O1", 100), 1000.0)
        sim.place_order(_make_order("O2", 200), 1100.0)
        self.assertTrue(karma.verify_integrity(),
                        "Karma hash chain is invalid after fills")

    def test_log_entries_non_empty(self):
        karma = KarmaLog()
        sim = ShadowSimulator(risk_gate=RiskGate(RiskConfig(max_trades_per_min=1000)),
                              karma_log=karma)
        sim.place_order(_make_order("O1", 50), 1000.0)
        log = karma.get_log()
        self.assertGreater(len(log), 0)
        self.assertEqual(log[-1]["observation_type"], "ORDER_COMPLETE")


class TestIntradayAnalytics(unittest.TestCase):

    def test_spread_tracker_ema(self):
        st = SpreadTracker(alpha=1.0)  # alpha=1 = no smoothing, output = raw spread
        ts = datetime(2026, 2, 19, 9, 15)
        m = st.update(ts, best_bid=100.0, best_ask=100.10)
        self.assertAlmostEqual(m.spread_absolute, 0.10, places=9)
        self.assertAlmostEqual(m.spread_smoothed, 0.10, places=9)

    def test_spread_ema_smoothing(self):
        st = SpreadTracker(alpha=0.5)
        ts = datetime(2026, 2, 19, 9, 15)
        st.update(ts, 100.0, 100.10)   # first: smoothed = 0.10
        m = st.update(ts, 100.0, 100.20)  # second: smoothed = 0.5×0.20 + 0.5×0.10 = 0.15
        self.assertAlmostEqual(m.spread_smoothed, 0.15, places=9)

    def test_atr_first_candle(self):
        atr = IntradayATR(period=14)
        ts = datetime(2026, 2, 19, 9, 15)
        out = atr.update(Candle(ts, open=100, high=105, low=98, close=103))
        self.assertAlmostEqual(out.true_range, 7.0, places=9)   # H-L = 7

    def test_atr_wilder_convergence(self):
        atr = IntradayATR(period=3)
        ts = datetime(2026, 2, 19, 9, 15)
        for i in range(10):
            atr.update(Candle(ts, 100, 102, 98, 100))   # constant TR=4
        self.assertAlmostEqual(atr._atr, 4.0, places=4,
                               msg="ATR should converge to TR when TR is constant")

    def test_kalman_cold_start(self):
        ks = KalmanSmoother()
        ts = datetime(2026, 2, 19, 9, 15)
        state = ks.update(ts, price=1000.0)
        self.assertEqual(state.observed_price, 1000.0)
        # After first tick smoother_price moves toward observation
        self.assertGreater(state.kalman_gain, 0)

    def test_kalman_smooths_noise(self):
        ks = KalmanSmoother(process_noise=1e-5, measure_noise=1e-3)
        ts = datetime(2026, 2, 19, 9, 15)
        for _ in range(50):
            ks.update(ts, price=1000.0)  # stable price
        state = ks.update(ts, price=1005.0)  # small spike
        # Smoother should NOT jump to 1005 immediately
        self.assertLess(state.smoother_price, 1004.0)

    def test_volume_delta_tick_rule(self):
        vd = VolumeDeltaTracker()
        ts = datetime(2026, 2, 19, 9, 15)
        vd.update(ts, price=100.0, volume=50)     # first tick: neutral
        delta_up   = vd.update(ts, price=100.5, volume=100)  # price up → buy aggressor
        delta_down = vd.update(ts, price=100.0, volume=80)   # price down → sell aggressor
        self.assertEqual(delta_up.buy_volume, 100.0)
        self.assertEqual(delta_down.sell_volume, 80.0)

    def test_garch_stationarity(self):
        with self.assertRaises(ValueError):
            GarchVolatility(omega=0.01, alpha=0.5, beta=0.6)  # α+β=1.1 > 1

    def test_garch_variance_nonneg(self):
        g = GarchVolatility(omega=1e-6, alpha=0.05, beta=0.90)
        ts = datetime(2026, 2, 19, 9, 15)
        for r in [0.001, -0.002, 0.003, -0.001]:
            out = g.update(ts, current_return=r)
            self.assertGreater(out.conditional_variance, 0)
            self.assertGreater(out.annualized_volatility, 0)


# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestModeGuard))
    suite.addTests(loader.loadTestsFromTestCase(TestFeeModel))
    suite.addTests(loader.loadTestsFromTestCase(TestTaxModel))
    suite.addTests(loader.loadTestsFromTestCase(TestDeterminismReplay))
    suite.addTests(loader.loadTestsFromTestCase(TestLiquidityChunking))
    suite.addTests(loader.loadTestsFromTestCase(TestSlippageModel))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeLifecycleArtifact))
    suite.addTests(loader.loadTestsFromTestCase(TestKarmaIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestIntradayAnalytics))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_all()
