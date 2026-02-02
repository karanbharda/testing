# Stop Loss Rules

## Stop Loss Mechanisms

Stop loss orders are risk management tools that automatically exit positions when price moves against the trade beyond acceptable levels.

## Types of Stop Loss

### Fixed Percentage Stop Loss
- Exit when loss reaches predetermined percentage
- Example: 2% stop loss on ₹10,000 position = exit at ₹9,800
- Simple to implement but doesn't account for volatility

### Volatility-Based Stop Loss
- ATR (Average True Range) based stops
- 1-2x ATR below entry for long positions
- Adjusts for market volatility
- More sophisticated risk management

### Trailing Stop Loss
- Moves with favorable price movement
- Locks in profits while limiting losses
- Percentage or point-based trailing
- Reduces emotional decision making

## SEBI Requirements for Stop Loss

### Mandatory Stop Loss
- Required for all retail positions above ₹5 lakh
- Must be placed at time of order entry
- Cannot be modified during market hours
- Minimum 1% stop loss for equity positions

### Institutional Exemptions
- Institutions can have different risk parameters
- Higher position limits
- Customized stop loss levels
- Subject to internal risk policies

## Implementation Rules

### Order Placement
- Stop loss must be visible in order book
- Cannot be hidden or iceberg orders
- Must be executable at market open
- Price bands apply to stop loss triggers

### Execution Priority
- Market orders when stop loss triggered
- Best available price execution
- No guaranteed execution at exact stop price
- Possibility of slippage in fast markets

## Risk Gates and Limits

### Position-Level Limits
- Maximum loss per position: 5-10% of capital
- Stop loss activation: 2-3% adverse movement
- Time-based exits: Maximum holding period

### Portfolio-Level Limits
- Maximum drawdown: 10-15% of portfolio value
- Daily loss limit: 2-3% of capital
- Sector exposure limits: 20-30% per sector

### Circuit Breaker Integration
- Automatic position closure at circuit levels
- ±10%, ±15%, ±20% triggers
- No manual intervention allowed
- Forced liquidation if required

## Practical Implementation

### Equity Trading
- Blue-chip stocks: 3-5% stop loss
- Mid-cap stocks: 5-8% stop loss
- Small-cap stocks: 8-10% stop loss
- Higher stops for volatile stocks

### Derivatives Trading
- Futures: 2-3% stop loss
- Options: Based on delta and theta decay
- Spreads: Wider stops due to limited risk
- Hedged positions: Tighter stops allowed

### Risk-Based Exit Logic
1. Stop loss hit → Immediate exit
2. Target achieved → Profit booking
3. Time decay → Position closure
4. Adverse news → Risk reduction
5. Portfolio rebalancing → Position adjustment