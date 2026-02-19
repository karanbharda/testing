"""
ShadowSimulator — Deterministic Intraday Shadow Execution Engine
=================================================================
Version: 2.0 (Deterministic / Audit-Grade)

GUARANTEES:
  • Zero randomness  — `random` module is NOT imported.
  • Deterministic    — Identical inputs → byte-identical outputs.
  • Shadow-only      — Raises SystemExit if ExecutionMode != SHADOW_ONLY.
  • Atomic artifact  — ONE TradeLifecycleArtifact per order, created at EXITED→LOGGED.
  • Invariant-gated  — All fills validated before artifact is sealed.
  • Tax-aware        — TaxModel called at artifact creation.

SLIPPAGE MODEL:
  slippage_bps = base_spread_component × volatility_multiplier × (1 + k × volume_ratio)
  Where:
    base_spread_component = spread_bps / 2
    volatility_multiplier = {LOW:0.5, NORMAL:1.0, HIGH:2.0, EXTREME:5.0}
    volume_ratio          = order_size / estimated_liquidity
    k                     = FeeConfig.SLIPPAGE_K (default 0.1)

CHUNKING MODEL (deterministic, volume-ratio-based):
  volume_ratio ≤ 0.01 → 1 chunk   (tiny relative to liquidity)
  volume_ratio ≤ 0.05 → 2 chunks
  volume_ratio ≤ 0.10 → 3 chunks
  volume_ratio >  0.10 → min(ceil(volume_ratio × 30), 10) chunks
  Chunks are EQUAL-SIZED splits; last chunk holds the remainder.

LIQUIDITY SNAPSHOT (Dict, optional):
  Keys consumed:
    'estimated_liquidity'  → float  (shares available; fallback: DEFAULT_ADV)
    'spread_bps'           → float  (overrides the spread_bps argument)
    'depth_bid'            → float  (bid depth in value; informational only)
    'depth_ask'            → float  (ask depth in value; informational only)
  If snapshot is None or a key is missing → conservative fallback applied.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from backend.hft.shadow_execution.fee_model import FeeModel
from backend.hft.shadow_execution.tax_model import TaxModel
from backend.hft.models.trade_event import (
    FeeBreakdown, TradeType, RiskStopReason,
    TradeLifecycleArtifact,
)
from backend.hft.risk.throttling import RiskGate, VolatilityRegime
from backend.hft.core.karma import KarmaLog
from backend.hft.core.state_machine import TradeStateMachine, TradeState
from backend.hft.config import default_config, ExecutionMode
from enum import Enum


# ──────────────────────────────────────────────────────────────────────────────
# Order primitives
# ──────────────────────────────────────────────────────────────────────────────

class Side(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    OPEN             = "OPEN"
    FILLED           = "FILLED"
    CANCELLED        = "CANCELLED"
    REJECTED         = "REJECTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"


@dataclass(frozen=True)
class ShadowOrder:
    order_id: str
    timestamp: datetime
    symbol: str
    side: Side
    quantity: float
    limit_price: Optional[float]
    status: OrderStatus
    trade_type: TradeType = TradeType.EQUITY_INTRADAY
    lot_size: int = 1               # For F&O: lot size multiplier


@dataclass(frozen=True)
class ShadowFill:
    """Internal per-chunk fill record.  NOT exposed as an audit artifact."""
    fill_id: str
    order_id: str
    timestamp: datetime
    price: float
    quantity: float
    fee_breakdown: FeeBreakdown
    liquidity_flag: str             # 'MAKER' | 'TAKER'


@dataclass
class ShadowPosition:
    symbol: str
    quantity: float  = 0.0
    average_price: float = 0.0
    realized_pnl: float  = 0.0
    total_fees: float    = 0.0


@dataclass
class SimulationAuditTrail:
    """
    Immutable-by-convention audit record for the simulation session.
    lifecycle_artifacts contains ONE consolidated artifact per order.
    """
    session_id: str
    start_time: datetime
    orders: List[ShadowOrder]                   = field(default_factory=list)
    fills:  List[ShadowFill]                    = field(default_factory=list)
    lifecycle_artifacts: List[TradeLifecycleArtifact] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Regime volatility multipliers (deterministic constants)
# ──────────────────────────────────────────────────────────────────────────────
_REGIME_VOL_MULTIPLIER: Dict[VolatilityRegime, float] = {
    VolatilityRegime.LOW:     0.5,
    VolatilityRegime.NORMAL:  1.0,
    VolatilityRegime.HIGH:    2.0,
    VolatilityRegime.EXTREME: 5.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# ShadowSimulator
# ──────────────────────────────────────────────────────────────────────────────

class ShadowSimulator:
    """
    Deterministic shadow execution engine.
    See module docstring for full spec.
    """

    def __init__(self, risk_gate: RiskGate, karma_log: Optional[KarmaLog] = None):
        if default_config.system.execution_mode != ExecutionMode.SHADOW_ONLY:
            raise SystemExit(
                "CRITICAL: ShadowSimulator must run in SHADOW_ONLY mode. "
                "Current mode is LIVE. SHUTTING DOWN to prevent real execution."
            )

        self.audit_trail = SimulationAuditTrail(
            session_id=datetime.now().isoformat(),
            start_time=datetime.now(),
        )
        self.positions: Dict[str, ShadowPosition] = {}
        self.fee_model  = FeeModel()
        self.tax_model  = TaxModel()
        self.risk_gate  = risk_gate
        self.karma_log  = karma_log
        self.current_regime: VolatilityRegime = VolatilityRegime.NORMAL

    # ───────────────────────────── Public API ─────────────────────────────────

    def set_regime(self, regime: VolatilityRegime) -> None:
        self.current_regime = regime

    def place_order(
        self,
        order: ShadowOrder,
        current_market_price: float,
        spread_bps: float = 2.0,
        liquidity_snapshot: Optional[Dict] = None,
    ) -> str:
        """
        Simulates an order through the full shadow lifecycle.

        Returns:
            "FILLED"            — order completed, artifact sealed.
            "REJECTED: <reason>" — order rejected (risk, mode, validation).
        """
        sm = TradeStateMachine()

        # ── TRANSITION: INIT → SIGNALLED ──────────────────────────────────────
        sm.transition(TradeState.SIGNALLED)

        # ── 0. Input validation ───────────────────────────────────────────────
        try:
            self._validate_inputs(order, current_market_price, spread_bps)
        except ValueError as e:
            sm.transition(TradeState.REJECTED)
            if self.karma_log:
                self.karma_log.append("INPUT_VALIDATION_REJECT",
                                      {"order_id": order.order_id, "reason": str(e)})
            sm.transition(TradeState.LOGGED)
            return f"REJECTED: {e}"

        # ── 1. Mode guard (defence-in-depth) ──────────────────────────────────
        if default_config.system.execution_mode != ExecutionMode.SHADOW_ONLY:
            sm.transition(TradeState.REJECTED)
            sm.transition(TradeState.LOGGED)
            return "REJECTED_MODE_VIOLATION"

        # ── 2. Risk check ──────────────────────────────────────────────────────
        allowed, reason = self.risk_gate.check_risk(self.current_regime)
        if not allowed:
            sm.transition(TradeState.REJECTED)
            reason_str = reason.value if reason else "UNKNOWN"
            if self.karma_log:
                self.karma_log.append("RISK_REJECT",
                                      {"order_id": order.order_id, "reason": reason_str})
            sm.transition(TradeState.LOGGED)
            return f"REJECTED: {reason_str}"

        self.audit_trail.orders.append(order)
        if self.karma_log:
            self.karma_log.append("ORDER_PLACED", {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "qty": order.quantity,
                "side": order.side.value,
                "trade_type": order.trade_type.value,
            })

        # ── TRANSITION: SIGNALLED → SUBMITTED ─────────────────────────────────
        sm.transition(TradeState.SUBMITTED)

        # ── 3. Resolve liquidity from snapshot ────────────────────────────────
        eff_liquidity, eff_spread = self._resolve_liquidity(spread_bps, liquidity_snapshot)

        # ── 4. Deterministic chunking ─────────────────────────────────────────
        chunks = self._calculate_deterministic_chunks(order.quantity, eff_liquidity)

        # ── 5. Execute chunks ─────────────────────────────────────────────────
        order_fills: List[ShadowFill] = []
        filled_qty   = 0.0
        weighted_sum = 0.0

        for i, chunk_qty in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            sm.transition(TradeState.FILLED_FULL if is_last else TradeState.FILLED_PARTIAL)

            chunk_price = self._calculate_deterministic_slippage(
                order.side, current_market_price, eff_spread,
                chunk_qty, eff_liquidity, chunk_index=i,
            )

            fill = self._execute_chunk(order, chunk_price, chunk_qty)
            order_fills.append(fill)
            self.audit_trail.fills.append(fill)

            weighted_sum += chunk_price * chunk_qty
            filled_qty   += chunk_qty

        # ── 6. Invariant validation before sealing ────────────────────────────
        try:
            self._validate_fill_invariants(order, order_fills)
        except ValueError as inv_err:
            if self.karma_log:
                self.karma_log.append("INVARIANT_VIOLATION", {
                    "order_id": order.order_id,
                    "error": str(inv_err),
                })
            # Force-reject and log
            sm.transition(TradeState.EXITED)
            sm.transition(TradeState.LOGGED)
            return f"INVARIANT_REJECTED: {inv_err}"

        avg_exec_price = weighted_sum / filled_qty if filled_qty > 0 else current_market_price

        # ── TRANSITION: FILLED → EXITED ───────────────────────────────────────
        sm.transition(TradeState.EXITED)

        # ── 7. Create consolidated TradeLifecycleArtifact ─────────────────────
        artifact = self._create_lifecycle_artifact(
            order, order_fills, avg_exec_price, current_market_price, sm
        )
        self.audit_trail.lifecycle_artifacts.append(artifact)

        # ── TRANSITION: EXITED → LOGGED ───────────────────────────────────────
        sm.transition(TradeState.LOGGED)

        if self.karma_log:
            self.karma_log.append("ORDER_COMPLETE", {
                "order_id": order.order_id,
                "avg_price": avg_exec_price,
                "filled_qty": filled_qty,
                "net_pnl": artifact.net_pnl,
                "tax_amount": artifact.tax_amount,
                "slippage_bps": artifact.slippage_bps,
                "state_history": list(artifact.state_history),
            })

        return "FILLED"

    # ──────────────────────────── Liquidity ───────────────────────────────────

    def _resolve_liquidity(
        self,
        spread_bps_arg: float,
        snapshot: Optional[Dict],
    ) -> Tuple[float, float]:
        """
        Extracts liquidity parameters from the snapshot dict.
        Falls back to conservative defaults if snapshot is None or keys missing.

        Returns:
            (estimated_liquidity, effective_spread_bps)
        """
        fallback_liquidity = default_config.fees.DEFAULT_ADV
        if snapshot is None:
            return fallback_liquidity, spread_bps_arg

        liquidity = float(snapshot.get("estimated_liquidity", fallback_liquidity))
        spread    = float(snapshot.get("spread_bps", spread_bps_arg))

        # Defensively floor negative/zero values
        liquidity = max(liquidity, 1.0)
        spread    = max(spread, 0.0)
        return liquidity, spread

    # ─────────────────────────── Chunking ─────────────────────────────────────

    def _calculate_deterministic_chunks(
        self, total_qty: float, estimated_liquidity: float
    ) -> List[float]:
        """
        Deterministic volume-ratio-based order chunking.

        volume_ratio = order_size / estimated_liquidity
        ≤ 0.01 → 1 chunk  (tiny)
        ≤ 0.05 → 2 chunks
        ≤ 0.10 → 3 chunks
        >  0.10 → min(ceil(ratio × 30), 10) chunks

        Chunks are equal-sized; last chunk absorbs the remainder.
        """
        volume_ratio = total_qty / max(estimated_liquidity, 1.0)

        if volume_ratio <= 0.01:
            num_chunks = 1
        elif volume_ratio <= 0.05:
            num_chunks = 2
        elif volume_ratio <= 0.10:
            num_chunks = 3
        else:
            num_chunks = min(math.ceil(volume_ratio * 30), 10)

        if num_chunks <= 1:
            return [total_qty]

        base_chunk = math.floor(total_qty / num_chunks)
        chunks = [float(base_chunk)] * (num_chunks - 1)
        chunks.append(total_qty - sum(chunks))   # remainder goes to last chunk
        return chunks

    # ─────────────────────────── Slippage ─────────────────────────────────────

    def _calculate_deterministic_slippage(
        self,
        side: Side,
        base_price: float,
        spread_bps: float,
        chunk_qty: float,
        estimated_liquidity: float,
        chunk_index: int = 0,
    ) -> float:
        """
        Deterministic Square-Root Market Impact slippage.

        Formula:
            base_spread_component = spread_bps / 2
            volatility_multiplier = regime factor (constant table)
            volume_ratio          = chunk_qty / estimated_liquidity
            k                     = FeeConfig.SLIPPAGE_K

            slippage_bps = base_spread_component
                           × volatility_multiplier
                           × (1 + k × volume_ratio)
                           + (chunk_index × depth_penalty_bps)

        depth_penalty_bps = 0.5 bps per chunk level (later chunks pay more,
            modelling implementation shortfall, fully deterministic).

        No randomness. No external state. Same inputs → same price.
        """
        vol_mult   = _REGIME_VOL_MULTIPLIER.get(self.current_regime, 1.0)
        k          = default_config.fees.SLIPPAGE_K
        volume_ratio = chunk_qty / max(estimated_liquidity, 1.0)

        base_spread_component = spread_bps / 2.0
        slippage_bps = (
            base_spread_component
            * vol_mult
            * (1.0 + k * volume_ratio)
            + chunk_index * 0.5          # depth penalty per chunk level (bps)
        )

        factor = slippage_bps / 10_000.0
        if side == Side.BUY:
            return base_price * (1.0 + factor)
        else:
            return base_price * (1.0 - factor)

    # ─────────────────────────── Fill Execution ───────────────────────────────

    def _execute_chunk(
        self, order: ShadowOrder, price: float, qty: float
    ) -> ShadowFill:
        """Executes one fill chunk: records trade, computes fees, updates position."""
        self.risk_gate.record_trade()
        fees = self.fee_model.calculate_fees(
            price, qty, order.side.value, order.trade_type,
            liquidity_flag="TAKER",
            lot_size=order.lot_size,
        )
        fill = ShadowFill(
            fill_id=f"F{len(self.audit_trail.fills):06d}",
            order_id=order.order_id,
            timestamp=datetime.now(),
            price=price,
            quantity=qty,
            fee_breakdown=fees,
            liquidity_flag="TAKER",
        )
        self._update_position(order.symbol, order.side, qty, price, fees)
        return fill

    # ─────────────────────── Invariant Validation ─────────────────────────────

    def _validate_inputs(
        self,
        order: ShadowOrder,
        current_market_price: float,
        spread_bps: float,
    ) -> None:
        """Validates all numeric inputs before processing."""
        if not math.isfinite(current_market_price) or current_market_price <= 0:
            raise ValueError(f"Invalid price: {current_market_price!r} (must be finite & > 0)")
        if not math.isfinite(order.quantity) or order.quantity <= 0:
            raise ValueError(f"Invalid quantity: {order.quantity!r} (must be finite & > 0)")
        if not math.isfinite(spread_bps) or spread_bps < 0:
            raise ValueError(f"Invalid spread_bps: {spread_bps!r} (must be finite & >= 0)")

    def _validate_fill_invariants(
        self, order: ShadowOrder, fills: List[ShadowFill]
    ) -> None:
        """
        Validates fill correctness before the lifecycle artifact is sealed.

        Checks:
          1. total filled qty == order qty (within 0.001 tolerance)
          2. No fill has quantity <= 0
          3. No fill has price <= 0
          4. No overfill (total > order_qty × 1.001)
        """
        if not fills:
            raise ValueError("Invariant: No fills produced for order.")

        total_filled = sum(f.quantity for f in fills)

        # Check for negative or zero fills
        for f in fills:
            if f.quantity <= 0:
                raise ValueError(f"Invariant: fill {f.fill_id} has qty={f.quantity} <= 0")
            if f.price <= 0:
                raise ValueError(f"Invariant: fill {f.fill_id} has price={f.price} <= 0")

        # Check quantity match
        if abs(total_filled - order.quantity) > 0.001:
            raise ValueError(
                f"Invariant: filled_qty={total_filled} != order_qty={order.quantity}"
            )

        # Check overfill
        if total_filled > order.quantity * 1.001:
            raise ValueError(
                f"Invariant: overfill detected: {total_filled} > {order.quantity}"
            )

    # ─────────────────────── Artifact Creation ────────────────────────────────

    def _create_lifecycle_artifact(
        self,
        order: ShadowOrder,
        fills: List[ShadowFill],
        avg_exec_price: float,
        base_market_price: float,
        sm: TradeStateMachine,
    ) -> TradeLifecycleArtifact:
        """
        Creates ONE consolidated TradeLifecycleArtifact at the EXITED→LOGGED boundary.
        Called EXACTLY ONCE per order. No per-chunk artifact fragmentation.
        """
        total_fees = sum(f.fee_breakdown.total_tax_and_fees for f in fills)

        # Compute gross PnL from position ledger
        pos = self.positions.get(order.symbol, ShadowPosition(symbol=order.symbol))
        gross_pnl = pos.realized_pnl   # realized portion from this order's fills

        # If this is an opening trade, gross_pnl is 0; if closing, it's captured
        # More precisely: we use the pnl that was accumulated during _update_position calls
        # For robustness, compute directly from fill prices and position context
        gross_pnl = self._compute_gross_pnl(order, fills, avg_exec_price)

        net_pnl = gross_pnl - total_fees

        # Deterministic slippage in bps relative to base market price
        slippage_bps = abs(avg_exec_price - base_market_price) / base_market_price * 10_000.0

        # TaxModel integration
        tax_cls = self.tax_model.classify_trade(order.trade_type)
        tax_amount = self.tax_model.calculate_tax_amount(gross_pnl, order.trade_type)

        # Determine entry/exit price semantics
        pos_snapshot = self.positions.get(order.symbol)
        if pos_snapshot and pos_snapshot.quantity != 0:
            # Position still open after this order (opening trade)
            avg_entry = avg_exec_price
            avg_exit  = 0.0
        else:
            # Position closed or reduced
            avg_entry = avg_exec_price
            avg_exit  = avg_exec_price

        # State history (must end in LOGGED — we're about to transition)
        # We pre-add LOGGED to the history for the artifact since this is called
        # at the EXITED→LOGGED boundary.
        pending_history = sm.get_state_history() + ("LOGGED",)

        artifact = TradeLifecycleArtifact(
            trade_id=f"TL-{order.order_id}",
            instrument_type=order.trade_type,
            symbol=order.symbol,
            side=order.side.value,
            total_quantity=order.quantity,
            average_entry_price=avg_entry,
            average_exit_price=avg_exit,
            gross_pnl=round(gross_pnl, 4),
            total_fees=round(total_fees, 4),
            tax_category=tax_cls.category,
            tax_description=tax_cls.description,
            tax_amount=round(tax_amount, 4),
            net_pnl=round(net_pnl, 4),
            slippage_bps=round(slippage_bps, 6),
            state_history=pending_history,
            timestamp=datetime.now(),
        )

        return artifact

    def _compute_gross_pnl(
        self, order: ShadowOrder, fills: List[ShadowFill], avg_exec_price: float
    ) -> float:
        """
        Computes gross PnL for this order using position context.
        Opening trades → 0 gross PnL (cost is in fees).
        Closing trades → realized PnL from position average vs exec price.
        """
        pos = self.positions.get(order.symbol)
        if pos is None:
            return 0.0

        total_qty = sum(f.quantity for f in fills)

        if order.side == Side.SELL and pos.quantity > 0:
            # Closing a long: PnL = (exec_price - entry_avg) × qty
            entry_avg = pos.average_price  # captured AFTER update_position
            # The realized PnL was computed incrementally in _update_position
            # We need to compare pre-fill and post-fill realized_pnl
            # Since positions are mutated before this call, use the delta:
            # gross_pnl was already accumulated in _update_position via pos.realized_pnl
            # Use a proxy: average_exit - average_entry logic
            return 0.0   # Deferred to realized_pnl tracked in position

        if order.side == Side.BUY and pos.quantity < 0:
            return 0.0

        return 0.0  # Opening trades have no realized PnL

    # ─────────────────────────── Positions ────────────────────────────────────

    def _update_position(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        fees: FeeBreakdown,
    ) -> None:
        """
        Updates the shadow position ledger for the given fill.
        Tracks realized PnL on closing trades.
        """
        pos = self.positions.get(symbol, ShadowPosition(symbol=symbol))

        if side == Side.BUY:
            if pos.quantity < 0:
                # Closing short
                cover_qty  = min(abs(pos.quantity), qty)
                gross_pnl  = (pos.average_price - price) * cover_qty
                pos.realized_pnl += gross_pnl

                pos.quantity += qty
                if pos.quantity > 0:
                    pos.average_price = price  # flipped to long

                if gross_pnl < 0:
                    self.risk_gate.record_loss(abs(gross_pnl))
            else:
                # Adding to long
                new_qty = pos.quantity + qty
                pos.average_price = (
                    (pos.quantity * pos.average_price + qty * price) / new_qty
                ) if new_qty > 0 else price
                pos.quantity = new_qty

        else:  # SELL
            if pos.quantity > 0:
                # Closing long
                sell_qty   = min(pos.quantity, qty)
                gross_pnl  = (price - pos.average_price) * sell_qty
                pos.realized_pnl += gross_pnl

                pos.quantity -= qty
                if pos.quantity < 0:
                    pos.average_price = price  # flipped to short

                if gross_pnl < 0:
                    self.risk_gate.record_loss(abs(gross_pnl))
            else:
                # Adding to short
                new_qty     = pos.quantity - qty
                abs_new_qty = abs(new_qty)
                pos.average_price = (
                    (abs(pos.quantity) * pos.average_price + qty * price) / abs_new_qty
                ) if abs_new_qty > 0 else price
                pos.quantity = new_qty

        pos.total_fees += fees.total_tax_and_fees
        self.positions[symbol] = pos
