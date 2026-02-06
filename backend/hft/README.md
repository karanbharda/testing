# HFT Module (Shadow Mode)

## Overview
This module implements a high-frequency trading (HFT) ready architecture designed for **Shadow Execution Only**. It replicates the full lifecycle of an HFT system—from tick processing to microstructure analysis and trade simulation—without any connectivity to live brokers.

**Current State**: `SHADOW_ONLY`
- **Live Trading**: OFF (Hardcoded logic/router).
- **Simulation**: ON (Internal matching engine).

## Architecture

### 1. Conceptual "OFF Switch"
The `ExecutionRouter` (`backend/hft/execution_router.py`) acts as the final gatekeeper.
- **Guarantee**: It accepts orders and strictly routes them to the `ShadowSimulator`.
- **Safety**: The `ExecutionConfig` is hardcoded to `SHADOW_ONLY`. Attempts to enable live mode will raise critical runtime errors.

### 2. Intraday Tick Buffer (`tick_engine/`)
- **Buffer**: `TickBuffer` stores raw tick data in memory.
- **Abstraction**: Decouples strategy logic from raw data feed formats.

### 3. Microstructure Feature Layer (`microstructure/`)
- **Order Book Imbalance (OBI)**: `obi.py` computes buy/sell pressure.
- **Queue Position**: `queue_position.py` estimates fill probability.
- **Snapshots**: `order_book.py` captures L1/L2 states.

### 4. Quant Feature Pipeline (`intraday/` & `feature_pipeline/`)
- **Statistical Models**:
  - `kalman_filter.py`: Noise smoothing.
  - `garch_volatility.py`: Real-time volatility estimation.
  - `atr_volatility.py`: Intraday range tracking.
- **Features**: `volume_delta.py`, `micro_momentum.py`.
- **Contract**: `contract.py` defines the strictly read-only `FeatureVector` passed to ML agents.

### 5. Shadow Trade Simulator (`shadow_execution/`)
- **Simulator**: `simulator.py` accepts orders, simulates fills based on price/liquidity, and tracks PnL.
- **Audit**: Generates `SimulationAuditTrail` for post-session analysis.
- **Model**: `fee_model.py` applies Indian market taxes and fees to calculate "Realizable Alpha".

### 6. Intraday Risk Envelopes (`risk/`)
- **Envelopes**: `envelopes.py` defines hard limits (Max Loss/Min, Max Trades/Min).
- **Throttling**: `throttling.py` implements Regime-Aware Throttling, reducing activity during `HIGH` or `EXTREME` volatility.

## Shadow vs. Live
| Feature | Shadow Mode (Implemented) | Live Mode (Disabled) |
| :--- | :--- | :--- |
| **Execution** | Internal Simulator | Broker API (e.g., Fyers) |
| **Fills** | Probabilistic/Assumption-based | Real Market Fills |
| **PnL** | Theoretical (after Fees) | Real Money |
| **Latency** | Zero/Simulated | Network + Exchange Latency |

## HFT-Ready vs. HFT
- **HFT-Ready**: The architecture supports event-driven, tick-level processing, O(1) lookups, and minimal logic overhead. It *can* be scaled to HFT.
- **HFT**: Requires colocated servers, FPGA/C++ acceleration, and kernel bypass networking. This implementation is the logical Python prototype.

## Usage
Initialize the `ExecutionRouter` with a `ShadowSimulator`. Feed ticks into `TickBuffer`, compute features, check `RiskGate`, and route orders via `ExecutionRouter`.
