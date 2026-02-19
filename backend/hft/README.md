# HFT Shadow Execution Engine — Institutional Reference

> **Version 2.0 | Audit Grade | 2026-02-19**
> This document defines the technical and compliance characteristics of the
> shadow execution engine. It is written to withstand external institutional
> review.

---

## 1. Engine Classification

This is a **deterministic shadow-only intraday simulation engine**.

It is NOT:
- A live trading system
- A broker adapter
- A latency-racing HFT engine
- An order management system with real execution paths

It IS:
- A mathematically rigorous model of order execution costs, slippage, and P&L
- An audit-compliant record-keeping system
- A deterministic analytical framework for intraday instrument simulation

---

## 2. Deterministic Execution Policy

**Guarantee: identical inputs → byte-identical outputs, always.**

All sources of non-determinism have been eliminated:

| Source | Status | Resolution |
|---|---|---|
| `random.uniform` in slippage | ❌ Removed | Square-root market impact model |
| `random.randint` in chunk split | ❌ Removed | Volume-ratio deterministic chunking |
| Wall-clock `time.time()` in fills | ✅ Not in model path | Timestamps are event-time, not wall-clock |
| `datetime.now()` in session ID | ✅ Scoped to init only | Never used in calculations |

**Verification:** The `TestDeterminismReplay` test class runs the same order 5×
and asserts bit-identical execution prices, fees, net PnL, and state history.
The `TestNoRandomnessContract` test monkey-patches the `random` module to raise
`AssertionError` if any call is made during execution.

---

## 3. No-Randomness Guarantee

```
import random  ← NOT PRESENT in simulator.py
```

The only math primitives used are:
- `math.sqrt` — for Square-Root Market Impact
- `math.floor`, `math.ceil` — for deterministic chunking
- Pure arithmetic on typed constants from `FeeConfig`

---

## 4. Shadow-Only Mode Enforcement

The engine enforces shadow mode at **two independent layers**:

1. **Constructor guard** — `__init__` raises `SystemExit` immediately if
   `ExecutionMode != SHADOW_ONLY`. No order can be placed.

2. **Per-order guard** — `place_order()` re-checks execution mode and returns
   `REJECTED_MODE_VIOLATION` as an additional defence-in-depth measure.

Changing `ExecutionMode.LIVE` does **not** enable real broker calls anywhere
in this codebase. The real execution path is in `live_executor.py` which has
no connection to this engine.

---

## 5. Slippage Model

```
slippage_bps = base_spread_component
               × volatility_multiplier
               × (1 + k × volume_ratio)
               + (chunk_index × 0.5)
```

| Parameter | Formula | Source |
|---|---|---|
| `base_spread_component` | `spread_bps / 2` | Liquidity snapshot or argument |
| `volatility_multiplier` | `{LOW:0.5, NORMAL:1.0, HIGH:2.0, EXTREME:5.0}` | `VolatilityRegime` enum |
| `volume_ratio` | `chunk_size / estimated_liquidity` | Liquidity snapshot |
| `k` | `0.1` (default) | `FeeConfig.SLIPPAGE_K` |
| `depth_penalty` | `0.5 bps × chunk_index` | Implementation shortfall model |

**Properties:**
- Monotone: larger orders → larger slippage
- Conservative: missing liquidity data triggers `DEFAULT_ADV = 10,000` shares
- Bidirectional: BUY orders pay up; SELL orders receive less

---

## 6. Liquidity-Aware Chunking

```
volume_ratio = order_size / estimated_liquidity
```

| Volume Ratio | Chunks |
|---|---|
| ≤ 0.01 | 1 |
| 0.01 – 0.05 | 2 |
| 0.05 – 0.10 | 3 |
| > 0.10 | min(ceil(ratio × 30), 10) |

Chunks are equal-sized. The last chunk holds the remainder.
`sum(chunks) == order_qty` is enforced as a fill invariant.

**Liquidity Snapshot** (`Dict`, optional):
```python
{
  "estimated_liquidity": float,  # shares available (ADV proxy)
  "spread_bps": float,           # overrides spread_bps argument
  "depth_bid": float,            # informational
  "depth_ask": float,            # informational
}
```
Missing keys → conservative fallback. `None` snapshot → all fallbacks.

---

## 7. Trade Atomicity Model

Trades are tracked as `TradeLifecycleArtifact` — **one artifact per order**,
created at the `EXITED → LOGGED` state transition.

**No per-chunk artifact fragmentation.** Internal `ShadowFill` records exist
for granular fill tracking but are NOT the audit artifact.

### TradeLifecycleArtifact Fields

| Field | Description |
|---|---|
| `trade_id` | Unique ID (`TL-<order_id>`) |
| `instrument_type` | `TradeType` enum |
| `symbol` | Instrument symbol |
| `side` | `BUY` or `SELL` |
| `total_quantity` | Full order quantity |
| `average_entry_price` | Volume-weighted average fill price |
| `average_exit_price` | Weighted exit price (closing trades) |
| `gross_pnl` | P&L before fees |
| `total_fees` | Sum of all fill-level fees |
| `tax_category` | e.g. `BUSINESS_INCOME`, `STCG_OR_LTCG` |
| `tax_description` | Legal section reference |
| `tax_amount` | Estimated tax liability (INR) |
| `net_pnl` | `gross_pnl − total_fees` |
| `slippage_bps` | Deterministic slippage in basis points |
| `state_history` | Immutable tuple of all states visited |
| `timestamp` | Artifact creation time |

### Enforced Invariants (in `__post_init__`)

1. `net_pnl == gross_pnl − total_fees` (within ₹0.01 tolerance)
2. `total_quantity > 0`
3. `total_fees >= 0`
4. `slippage_bps >= 0`
5. `tax_amount >= 0`
6. `state_history` is non-empty and ends in `"LOGGED"`

---

## 8. Instrument Coverage

| Instrument | TradeType | Fee Method | Tax Classification |
|---|---|---|---|
| Equity Intraday | `EQUITY_INTRADAY` | NSE CM segment | Speculative Business Income (s.43(5)) |
| Equity Delivery | `EQUITY_DELIVERY` | NSE CM segment (0 brokerage) | STCG 20% / LTCG 12.5% |
| Futures | `FUTURES` | NSE NFO — notional turnover | Non-Spec Business Income (s.43(5) proviso) |
| Options | `OPTIONS` | NSE NFO — premium turnover | Non-Spec Business Income |
| Crypto Spot | `CRYPTO_SPOT` | Exchange-agnostic MAKER/TAKER + TDS | VDA Flat 30% (s.115BBH) |
| Crypto Futures | `CRYPTO_FUTURES` | Exchange-agnostic + TDS | VDA Flat 30% (s.115BBH) |

**Fee rate sources:**
- NSE circular FAOP/49765 (F&O segment charges)
- Finance Act 2022, Section 115BBH / 194S (Crypto TDS 1%)
- Zerodha standard schedule (equity brokerage cap ₹20)

---

## 9. State Machine

Every order follows a deterministic state path enforced by `TradeStateMachine`.
Illegal transitions raise `StateTransitionError` — they cannot be silently skipped.

```
INIT → SIGNALLED → SUBMITTED → FILLED_PARTIAL → FILLED_FULL → EXITED → LOGGED
                                          ↓ (reject)
                             SIGNALLED → REJECTED → LOGGED
```

Full history is recorded as an immutable `Tuple[str, ...]` embedded in the
lifecycle artifact.

---

## 10. Karma Log (Audit Trail)

The `KarmaLog` is an append-only, cryptographically linked event log.
Each entry contains a SHA-256 hash of its content and the hash of the
previous entry — forming a tamper-evident blockchain-like structure.

Key events logged per order:
1. `ORDER_PLACED` — order parameters
2. `ORDER_COMPLETE` — avg price, fees, net PnL, slippage, state history
3. `RISK_REJECT` — if risk gate blocks the order
4. `INPUT_VALIDATION_REJECT` — if inputs are invalid
5. `INVARIANT_VIOLATION` — if fill invariants fail

`karma.verify_integrity()` replays the hash chain and returns `False` on
any tampering.

---

## 11. Verification

Run the full test suite from the project root:

```bash
python -m backend.hft.verify_hft
```

Expected output: **11 test classes, all PASS**.

| Test Class | What it verifies |
|---|---|
| `TestModeGuard` | LIVE mode raises SystemExit |
| `TestFeeModel` | All 6 instrument types, crypto TDS, determinism |
| `TestTaxModel` | All 6 types classified; tax amounts correct |
| `TestDeterminismReplay` | 5 runs of same order → bit-identical results |
| `TestNoRandomnessContract` | `random` module patched to fail on any call |
| `TestLiquidityChunking` | Volume-ratio → chunk count; sum invariant |
| `TestSlippageModel` | Larger order → higher slippage; HIGH regime > NORMAL |
| `TestTradeLifecycleArtifact` | One artifact per order; LOGGED terminal; tax attached |
| `TestKarmaIntegrity` | Hash chain valid after fills |
| `TestIntradayAnalytics` | EMA spread, ATR, Kalman, VolumeDelta, GARCH numeric |

---

## 12. Prohibited Operations

The following are **hard-prohibited** by design — any attempt will either
raise `SystemExit`, return a `REJECTED` result, or be structurally impossible
due to missing code paths:

| Prohibited Action | Enforcement Point |
|---|---|
| Real broker order submission | No broker adapter in this module |
| Modifying `ExecutionMode` at runtime to bypass guards | Both constructor + per-order mode checks |
| Introducing randomness | `random` not imported; test suite detects any call |
| Creating per-chunk artifacts | Artifact creation is in a single `_create_lifecycle_artifact` call |
| Skipping state transitions | `TradeStateMachine.transition()` raises on illegal transitions |
| Sealing artifact with invariant violations | `_validate_fill_invariants()` gates artifact creation |

---

*End of document. For questions, contact the system engineering team.*
