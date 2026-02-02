# Trading Strategies & System Design

## Momentum Trading System

### Concept
- Buy uptrend momentum
- Ride the move
- Exit on reversal signals

### Criteria
**Entry:**
1. Price > SMA50 > SMA200 (uptrend)
2. RSI > 50 (positive momentum)
3. MACD above signal line
4. Volume increasing
5. All rules confirmed on daily chart

**Exit:**
1. RSI > 70 + price rejection (overbought exit)
2. MACD crosses below signal line
3. Price closes below SMA20
4. Stop loss: Below recent swing low
5. Target: 1.5 to 2x risk

### Position Management
- Entry: Scale in over 2-3 days
- Profit target: Sell 50% at 1.5x, 30% at 2.5x
- Trailing stop: 8-10% below recent high
- Risk per trade: 1% of capital

### Performance Metrics
- Win rate: 55-60% typical
- Profit factor: 1.5-2.0x
- Average winner: 3-4x risk
- Average loser: 1x risk

## Mean Reversion System

### Concept
- Identify overbought/oversold extremes
- Trade bounce/reversion
- Exit at mean or beyond

### Criteria
**Entry (Bounce Up):**
1. Price bounces from support
2. RSI < 30 (oversold)
3. Stochastic < 20
4. Volume on down: Decreasing
5. Setup on weekly, confirm on daily

**Entry (Bounce Down):**
1. Price bounces from resistance
2. RSI > 70 (overbought)
3. Stochastic > 80
4. Volume on up: Decreasing
5. Setup on weekly, confirm on daily

**Exit Rules:**
1. Target: Return to SMA20
2. Stop loss: Below entry + 0.5% buffer
3. Time stop: 5-10 days max
4. Profit taking: Scale out at 50%

### Position Management
- Risk per trade: 1-2% (tight stops)
- Position size: Can be larger (tight SL)
- Multiple setups: Can combine
- Holding period: 3-10 days typical

### Performance Metrics
- Win rate: 60-65% typical
- Profit factor: 1.2-1.5x
- Average winner: 1.5-2x risk
- Average loser: 1x risk
- Sharpe ratio: Higher than momentum

## Breakout System

### Concept
- Support/resistance breakout
- Trend initiation signal
- Sustained move expected

### Criteria
**Resistance Breakout (Buy):**
1. Price consolidation above resistance
2. Resistance = prior swing high
3. Breakout with volume spike (>2x avg)
4. Confirmed close above resistance
5. At least 3-4 tests of resistance before break

**Support Breakout (Sell):**
1. Price consolidation below support
2. Support = prior swing low
3. Breakdown with volume spike (>2x avg)
4. Confirmed close below support
5. Multiple rejections at support level

**Exit Rules:**
1. Target: 2-3x risk from entry
2. Stop loss: Other side of consolidation
3. Trailing stop: 15-20% trail
4. Scale out: 50% at first target

### Position Management
- Risk per trade: 1.5-2% (larger moves expected)
- Pyramiding: Add on subsequent breakouts
- Timeframe: Trend > consolidation size
- Execution: Patience for confirmation

### Performance Metrics
- Win rate: 50-55% typical
- Profit factor: 2.0-2.5x
- Average winner: 3-5x risk
- Average loser: 1x risk
- Trend ratio: Important metric

## Options Income System

### Covered Call (Sell Calls)
**Setup:**
- Own 100 shares or 1 lot
- Sell 1 call at OTM strike
- Collect premium
- Capped upside, defined downside

**Premium Collection:**
- Target: 2-3% monthly return
- Frequency: Weekly or monthly
- Roll: Rewrite before expiry
- Yield: 24-36% annualized

**Risk Management:**
- Stop loss on shares: 5-8%
- Assignment: Expected if profitable
- Replacement: Keep cash for buyback
- Max upside: Strike price

### Cash-Secured Put (Sell Puts)
**Setup:**
- Cash equal to strike × 100
- Sell 1 put, keep cash in account
- Collect premium
- Own at strike if assigned

**Premium Collection:**
- Target: 2-3% monthly return
- Risk: Full strike price
- Assignment: Gets stock at discount
- Entry: Like buying stock at discount

**Assignment Handling:**
- Accept if thesis positive
- Roll down/out if negative
- Close if cash needed
- Portfolio addition strategy

### Iron Condor (High Probability)
**Setup:**
- Sell OTM call + Buy further OTM call
- Sell OTM put + Buy further OTM put
- Profit from range-bound market
- Max profit: Credit received

**Probability:**
- Target 60-70% win rate
- Win if index stays in range
- ~21 DTE entry
- Exit: At 50% profit (2x risk)

**Management:**
- Width: Usually ₹200-300 wide
- Frequency: Monthly (21 DTE minimum)
- Exit: Early if profit target hit
- Stop loss: Debit at entry × 2

## Algorithmic Trading Strategies

### Statistical Arbitrage
- Identify correlated pairs
- Mean reversion trades
- Market neutral (hedge both sides)
- Low volatility, consistent returns

### Cross-Currency Pairs
- USD-INR, EUR-INR futures
- Interest rate differential
- PPP (Purchasing Power Parity)
- Directional or mean-reversion

### Sector Rotation
- Identify cycle phases
- Rotate from defensive to cyclical
- Use index futures for execution
- Quarterly rebalancing

## Strategy Combination

**Portfolio Approach:**
1. **60% Momentum Trades**: ₹60,000
   - Large cap, liquid
   - Higher risk, higher reward
   - 3-10 day holding

2. **30% Mean Reversion**: ₹30,000
   - Mid-cap, decent liquidity
   - Lower risk, consistent wins
   - 3-10 day holding

3. **10% Income (Options)**: ₹10,000
   - Premium collection
   - Monthly rolls
   - Defined, lower risk

**Allocation:**
- Risk per strategy: 1% of capital
- Diversification: Reduces correlation
- Different holding periods: Smooths returns
- Flexibility: Pivot with market

## Performance Measurement

**Key Metrics:**
1. **Returns**: Total, annualized, monthly
2. **Win Rate**: % of winning trades
3. **Profit Factor**: Total wins / Total losses
4. **Sharpe Ratio**: Return / Volatility
5. **Sortino Ratio**: Return / Downside volatility
6. **Maximum Drawdown**: Largest peak-to-trough
7. **Recovery Factor**: Total profit / Max drawdown

**Targets:**
- Sharpe ratio: >1.0 (good), >2.0 (excellent)
- Win rate: >50% (profitable)
- Profit factor: >1.5x (sustainable)
- Max drawdown: <20% of capital
- Monthly return: 2-5% consistent

## Trading Psychology

**Emotional Discipline:**
- Follow system rules always
- No revenge trades
- No oversizing after loss
- No undersizing after win

**Common Mistakes:**
- Revenge trading after stop loss
- Holding losers hoping recovery
- Adding to losing positions
- Exiting winners too early
- Oversizing on confidence

**Risk Management First:**
- Capital preservation
- Consistent 2-5% monthly
- Compound over time
- Longevity over heroic returns
