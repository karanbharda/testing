# High Frequency Trading (HFT) Layer

## Overview
This directory contains the **Shadow HFT System**, designed for:
1. **Financial Realism**: Accurate Indian market fee and tax modeling.
2. **Risk Determinism**: Hardened risk gates with explicit stop reasons.
3. **System Integrity**: Robust tick processing and safety-locked execution.

## Institutional Clarifications
### 1. What is Shadow Mode?
`SHADOW_ONLY` is the **default and mandatory** state of this system. In this mode:
- **No external orders** are ever sent to a broker.
- **Execution is simulated** against live market data using liquidity-aware slippage models.
- **PnL is theoretical** but tax-aware (STT, Stamp Duty, GST deducted).
- **Audit Trails** are generated as if the trades were real, stored in append-only Karma logs.

### 2. What is Live Mode?
`LIVE` mode is a strictly restricted state that requires explicit configuration overrides. **It is currently NOT ENABLED for this repository.** Any attempt to run components in LIVE mode without proper flags will trigger an immediate system shutdown (Exit Code 1).

### 3. What is "OFF"?
The system is "OFF" when:
- The `TRADING_MODE` config is set to anything other than `SHADOW_ONLY` (safety mechanism).
- The `RiskGate` has triggered a Stop (e.g., Max Loss reached).

### 4. Authority
The **System Config** (`backend/hft/config.py`) is the single source of truth. Runtime changes to config are ignored to prevent hot-swapping into unsafe states.

## Modules

### 1. Models (`hft/models/`)
- `TradeArtifact`: **Immutable, atomic record** of a trade event.
- `FeeBreakdown`: detailed tax analysis.

### 2. Shadow Execution (`hft/shadow_execution/`)
- **`ShadowSimulator`**: The core execution engine.
    - **Liquidity-Aware Slippage**: Prices degrade with size and volatility.
    - **Partial Fills**: Orders are broken into chunks.
    - **Trade State Machine**: Enforces `INIT -> SIGNALLED -> SUBMITTED -> FILLED -> LOGGED`.
- **`FeeModel`**: Deterministic fee calculations for Intraday vs Delivery.

### 3. Core (`hft/core/`)
- **`KarmaLog`**: Append-only audit log.
- **`TradeStateMachine`**: Validation logic for trade lifecycles.

### 4. Tick Engine (`hft/tick_engine/`)
- **`TickBuffer`**: Hardened buffer with **Backpressure Monitoring**.

## Verification
Run the verification script to test all components:
```bash
python backend/hft/verify_hft.py
```
This script validates:
- Deterministic Fee Checks.
- Risk Gate triggers.
- Shadow Simulation flow.
- Buffer Backpressure.
