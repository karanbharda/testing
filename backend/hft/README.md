# High Frequency Trading (HFT) Layer

## Overview
This directory contains the **Shadow HFT System**, designed for:
1. **Financial Realism**: Accurate Indian market fee and tax modeling.
2. **Risk Determinism**: Hardened risk gates with explicit stop reasons.
3. **System Integrity**: Robust tick processing and safety-locked execution.

## Modules

### 1. Models (`hft/models/`)
- `TradeType`: `EQUITY_INTRADAY` vs `EQUITY_DELIVERY`.
- `FeeBreakdown`: Detailed tax analysis (STT, GST, Stamp Duty, etc.).
- `RiskStopReason`: Structured enums for risk rejections (e.g., `MAX_LOSS_MINUTE`).

### 2. Shadow Execution (`hft/shadow_execution/`)
- **`FeeModel`**: Calculates brokerage (Zerodha/NSE rates), STT, Exchange Txn Charges, SEBI Fees, Stamp Duty, and GST.
- **`ShadowSimulator`**: Simulates order execution with:
    - **Regime-aware Slippage**: Higher slippage in high volatility.
    - **Partial Fills**: Probabilistic fill logic.
    - **Safety**: Physically decoupled from broker API in SHADOW mode.

### 3. Risk Engine (`hft/risk/`)
- **`RiskGate`**: The central guardian. Checks every order against:
    - Max Trades per Minute
    - Max Loss per Minute
    - Market Regime (e.g., halts in EXTREME volatility)
- **`RegimeThrottler`**: Dynamically adjusts rate limits based on volatility.

### 4. Tick Engine (`hft/tick_engine/`)
- **`TickBuffer`**: Fixed-size circular buffer with monotonicity checks and overflow protection.

### 5. Config (`config.py`)
- **`MODE`**: Defaults to `SHADOW`. 
- **`RiskConfig`**: Central place to set limits.

## Usage

```python
from backend.hft.pipeline import HFTPipeline
from backend.hft.tick_engine import Tick

# Initialize
pipeline = HFTPipeline()

# Ingest Ticks
pipeline.process_tick(Tick(symbol="INFY", price=1500.0, ...))

# Generate Orders (Internal logic or External signal)
# ...
```

## Verification
Run the verification script to test all components:
```bash
python backend/hft/verify_hft.py
```
