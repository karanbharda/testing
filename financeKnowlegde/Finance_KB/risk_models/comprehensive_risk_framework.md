# Risk Management & Position Sizing Framework

## Risk Management Principles

### Position Sizing Rules

**Percentage Risk Rule (Kelly Criterion Simplified):**
- Risk per trade: 1-2% of total capital maximum
- Formula: Position Size = (Capital × Risk %) / Stop Loss Distance
- Example: ₹1,00,000 capital, 1% risk (₹1000), SL 20 paise away
  - Position size = ₹1000 / ₹0.20 = 5000 shares

**Volatility-Adjusted Sizing:**
- High volatility: Reduce position size
- Low volatility: Can increase position size
- Use ATR for standardization
- Risk per trade = ATR × Position Size (should be ₹1000-2000)

**Account Heat Rule:**
- Daily loss limit: 2% of capital
- Weekly loss limit: 5% of capital
- Monthly loss limit: 10% of capital
- Stop trading after hitting daily limit

### Stop Loss Implementation

**Types of Stop Losses:**
1. **Fixed Stop Loss**
   - ₹X amount loss per trade
   - Simple to implement
   - Ignores volatility changes
   - Best for consistent position sizes

2. **Percentage Stop Loss**
   - X% of entry price
   - Adjusts for stock price levels
   - Example: 5% SL on ₹500 stock = ₹475

3. **Technical Stop Loss**
   - Below recent swing low
   - Supports/Resistance levels
   - Aligns with trade thesis
   - More meaningful placement

4. **Trailing Stop Loss**
   - Follows price upward/downward
   - Protects profits while allowing for moves
   - Example: 10% trailing on ₹500 stock
   - Automatic lock-in of profits

**Stop Loss Discipline:**
- MUST be entered at trade entry
- NO moving stops against position
- EXIT at stop: No exceptions
- Pre-calculate exit before entry

### Profit Taking Strategies

**Scaling Out:**
- Sell 50% at first target (2x risk)
- Sell 30% at second target (4x risk)
- Trail 20% with stop at breakeven
- Locks in gains while capturing upside

**Target Levels:**
- First target: 1-2x risk reward ratio
- Second target: 3-4x risk reward ratio
- Runner: Trail for bigger move
- No predetermined target: True SL protection only

**Profit Booking Rules:**
- Book profits on technical weakness
- Never hold entire position into resistance
- Re-entry on fresh signals
- Avoid "diamond hand" mentality

## Value at Risk (VaR)

**Concept:**
- Maximum loss at certain confidence level (95%, 99%)
- Over specified time period (1 day, 10 days)
- Example: 1-day VaR 95% = ₹50,000
  - 95% chance max loss is ₹50,000 or less

**Calculation Methods:**

1. **Historical Method**
   - Use past 250 days of returns
   - Take 5th percentile return (95% confidence)
   - Multiply by position value

2. **Variance-Covariance (Parametric)**
   - Assume normal distribution
   - Calculate std dev of returns
   - VaR = Position × 1.645 × Std Dev (95% confidence)

3. **Monte Carlo Simulation**
   - Simulate thousands of price paths
   - Calculate percentile of outcomes
   - Most accurate but computationally intensive

**Portfolio VaR:**
- Not just sum of individual VaRs
- Correlations between assets matter
- Diversification reduces portfolio VaR
- Must account for hedge positions

## Expected Shortfall (ES/CVaR)

**Definition:**
- Average loss beyond VaR threshold
- More realistic than VaR
- Account for tail risk
- Example: ES may be 30% worse than VaR

**Usage:**
- Stress scenario planning
- Liquidity risk assessment
- Regulatory capital requirements
- Extreme event preparation

## Correlation & Diversification

**Correlation Analysis:**
- Perfect positive (+1): Move together
- Perfect negative (-1): Move opposite
- Zero correlation (0): Unrelated
- Low/negative correlation: Better diversification

**Portfolio Diversification Rules:**
1. Across sectors: 4-5 different sectors
2. Across market caps: Mix of large, mid, small
3. Across instruments: Equity, debt, derivatives
4. Across volatility: Mix of stable & volatile
5. Across time: Stagger entry/exit

**Correlation Breakdown:**
- In crisis: Correlations tend to +1
- Diversification breaks down when needed most
- Manage downside risk separately
- Use hedges for extreme tail risk

## Drawdown Management

**Maximum Drawdown:**
- Peak-to-trough decline
- Measures volatility and downside risk
- Example: Peak ₹1,00,000 → Trough ₹75,000 = 25% drawdown

**Monitoring Drawdown:**
- Daily NAV tracking
- Alert at 10% drawdown
- Action plan at 15% drawdown
- Risk reduction at 20% drawdown

**Recovery Calculation:**
- % gain needed to recover from loss
- 20% loss requires 25% gain to break even
- 50% loss requires 100% gain to break even
- Larger losses take exponentially longer to recover

## Leverage & Margin Risk

**Margin Multiplier Effect:**
- 2x leverage: 2x gains, 2x losses
- 10x leverage: 10x gains, 10x losses
- Margin call risk: Position liquidated forcefully
- Gap risk: Opening gap can exceed stop loss

**Safe Leverage Guidelines:**
- Conservative: 1:1 (no leverage)
- Moderate: 1:2 to 1:3
- Aggressive: 1:5 and above
- Must maintain margin buffer: 20-30% above minimum

**Leverage During Volatility:**
- High volatility: Reduce leverage
- Low volatility: Can increase leverage
- Volatility spike: Immediate leverage reduction
- Monitor daily volatility changes

## Sector & Correlation Risk

**Sector Concentration:**
- No sector >30% of portfolio
- Banking/IT: Natural correlation
- Auto & Real Estate: Cyclical together
- FMCG & Pharma: Defensive, low correlation

**Correlation in Portfolio:**
- Banking + IT: High correlation
- Auto + Real Estate: High correlation
- Pharma + FMCG: Low correlation
- Finance + Materials: Moderate correlation

## Counterparty Risk

**Broker Risk:**
- Segregation of client funds
- SEBI regulated & capital requirements
- Fail to deliver: Rare in regulated markets
- Insurance protection: Limited

**Clearing House Risk:**
- NSCCL guarantees trades
- Default fund: Protects other participants
- Real-time monitoring
- Minimal in India's regulated environment

## Liquidity Risk

**Bid-Ask Spread Analysis:**
- Tight spread (<1 paisa): Liquid
- Normal spread (1-5 paise): Reasonable
- Wide spread (>10 paise): Illiquid
- Avoid positions in illiquid securities

**Slippage Estimation:**
- Assume 0.5x average bid-ask for entry
- Add 1-2% for market impact on large orders
- Factor into profit targets
- Example: 5000 share order, 2 paisa spread, 1% market impact

**Exit Planning:**
- Plan exit before entry
- Verify liquidity at entry
- Don't hold illiquid positions overnight
- Scale out of illiquid positions gradually

## Stress Testing & Scenario Analysis

**Scenario 1: 5% Index Move (Intraday)**
- NIFTY moves -5%
- Average stock -4%
- Impact on portfolio: -4% to -6%
- Check position sizing

**Scenario 2: 20% Index Correction**
- NIFTY moves -20%
- Average stock -15% to -20%
- Impact on portfolio: -15% to -20%
- Check drawdown limits

**Scenario 3: Volatility Spike (VIX x2)**
- Options losses: Significant vega impact
- Delta adjusted positions: Protected
- Unhedged long: Major loss
- Liquidity drying up: Wider spreads

**Scenario 4: Gap Down at Open**
- Stop losses might not execute
- Gap size: 3-5% typical, 10%+ rare
- Impact: Position larger than intended loss
- Mitigation: Reduce position size near support

## Risk Dashboard Metrics

**Daily Monitoring:**
- Daily P&L: Absolute and %
- Drawdown: From peak
- Margin utilization: % of available
- VaR: 1-day 95% and 99%

**Weekly Monitoring:**
- Win rate: % of winning trades
- Profit factor: Avg win / Avg loss
- Sharpe ratio: Return per unit risk
- Maximum loss: Largest drawdown

**Monthly Monitoring:**
- Return: Absolute and %
- Volatility: Std dev of daily returns
- Sortino ratio: Return per downside volatility
- Hit ratio: % of profitable days
