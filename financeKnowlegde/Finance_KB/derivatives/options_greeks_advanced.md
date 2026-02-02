# Options Greeks & Advanced Derivative Concepts

## The Greeks Explained

### Delta (∆) - Direction Risk
**Definition:** Rate of change of option price relative to underlying asset price
- Range: 0 to 1 (Call), -1 to 0 (Put)
- ATM (At The Money): ~0.5 call, ~-0.5 put
- ITM (In The Money): Call ~0.7-1.0, Put ~-0.7 to -1.0
- OTM (Out The Money): Call ~0.1-0.3, Put ~-0.1 to -0.3

**Practical Uses:**
- Hedge position: Sell delta-equivalent options
- Directional betting: Buy high delta for conviction
- Delta-neutral strategies: Net delta = 0

**Example:** Call with delta 0.6 means if underlying rises ₹1, call rises ₹0.60
Leverage effect: Small capital controls larger position

### Gamma (Γ) - Delta Change Rate
**Definition:** Rate of change of delta with respect to underlying price
- Always positive for long options (call/put)
- Always negative for short options
- Highest at ATM, lowest deep ITM/OTM

**Implications:**
- Long gamma = Gains from volatility, loses from time decay
- Short gamma = Loses from volatility, gains from time decay
- Gamma acceleration near expiry: Increases dramatically

**Risk Management:**
- Monitor gamma exposure in portfolio
- Gamma risk during market gaps
- Use gamma hedging for stable P&L

### Vega (ν) - Volatility Risk
**Definition:** Change in option price per 1% change in implied volatility
- Positive for long options (both calls & puts)
- Negative for short options
- Same for calls and puts at same strike
- Highest at ATM, lowest deep ITM/OTM

**Volatility Dynamics:**
- Historical volatility: Backward-looking (realized)
- Implied volatility: Forward-looking (market expectation)
- Volatility smile: Different IV for different strikes
- Volatility term structure: IV varies by expiry

**Trading Applications:**
- Sell when IV high, buy when IV low
- Volatility arbitrage strategies
- Risk-reversal trades
- Calendar spreads exploiting time value

### Theta (Θ) - Time Decay
**Definition:** Daily decrease in option price due to time decay
- Positive for short options (time decay helps)
- Negative for long options (time decay hurts)
- Accelerates near expiry (day-of-expiry: Maximum decay)
- Call theta ≈ Put theta for same strike

**Time Decay Impact:**
- OTM options: Rapid theta decay
- ATM options: Steady theta decay
- ITM options: Minimal time value, low theta

**Expiry Management:**
- Day-of-expiry: 6-7x higher theta than 1 week before
- Weekly options: Accelerated decay
- Month-end expiry: Planning critical for position management

### Rho (ρ) - Interest Rate Risk
**Definition:** Change in option price for 1% change in interest rates
- Less significant for equities (typically ignored)
- More important for currency and bond options
- Longer duration options have higher rho
- Call: Positive rho, Put: Negative rho

**Applicability:**
- Currency derivatives: High rho impact
- Commodity options: Carry cost effects
- Equity options: Minor impact usually ignored

## Option Pricing Models

### Black-Scholes Model
**Formula Components:**
- Stock price (S)
- Strike price (K)
- Time to expiry (T)
- Risk-free rate (r)
- Volatility (σ)

**Key Assumptions:**
- European options (exercise only at expiry)
- No dividends
- Lognormal price distribution
- Constant volatility
- No transaction costs

**Limitations:**
- Doesn't account for dividends
- Doesn't handle American options
- Assumes constant volatility (real: smile/skew)
- Ignores transaction costs

### Binomial Model
**Advantages:**
- Handles American options
- Incorporates dividend dates
- Allows varying volatility
- More flexible than Black-Scholes

**Process:**
1. Build price tree
2. Calculate option value at nodes
3. Work backward to present value
4. Step intervals affect accuracy

### Volatility Surface & Smile

**Volatility Smile:**
- ATM options: Lower implied vol
- OTM calls / ITM puts: Higher vol (volatility smile)
- Reflects market's risk assessment

**Volatility Skew:**
- Asymmetric smile
- Deep OTM puts: Much higher IV (crash protection premium)
- Equity markets: Typically negative skew

**Term Structure:**
- Short-term volatility: High (near events)
- Medium-term: Can be lower
- Long-term: Stabilizes around long-run average

## Option Strategies

### Directional Strategies

**Bull Call Spread:**
- Buy ATM call, Sell OTM call
- Limited upside, limited loss
- Lower capital requirement
- Reduced theta decay vs naked call

**Bear Put Spread:**
- Sell OTM put, Buy further OTM put
- Profit from sideways market
- Defined risk, defined return
- Effective in low-IV environments

### Volatility Strategies

**Long Straddle:**
- Buy ATM call + Buy ATM put
- Profit from big move either direction
- High cost (double vega exposure)
- Useful before major announcements

**Iron Condor:**
- Bull put + Bear call spread
- Profit from range-bound market
- Limited risk, limited reward
- High probability strategy

**Butterfly Spread:**
- Sell 2 ATM, Buy 1 OTM call + Buy 1 ITM call
- Profit from stock staying at center strike
- Very low capital requirement
- Precise timing needed

### Time-based Strategies

**Calendar Spread:**
- Sell near-term, Buy far-term option (same strike)
- Profit from near-term time decay
- Gamma negative initially
- Roll forward as near-term expires

**Diagonal Spread:**
- Different strikes + different expirations
- Combines directional + time strategy
- Flexible adjustment
- Complex management required

## Advanced Concepts

### Implied Volatility Rank & Percentile
- IV Rank: Current IV vs 52-week range (0-100)
- IV Percentile: % of time IV has been lower
- High IV rank: Sell premium strategies attractive
- Low IV rank: Buy premium attractive

### Greeks of Greeks (Vanna, Volga)

**Vanna:** Change in delta per 1% IV change
- Useful for IV trading effects on delta hedge
- Positive for ATM long options

**Volga:** Change in vega per 1% IV change
- Long straddle: Positive volga
- Exposed to volatility of volatility

## Risk Management for Options

**Position Greeks:**
- Monitor net delta, gamma, vega, theta
- Set limits per Greek
- Rebalance when limits breached
- Daily P&L attribution

**Scenario Analysis:**
- Price shock scenarios (±5%, ±10%)
- Volatility increase/decrease scenarios
- Time decay impact
- Combined scenarios (price + vol move)

**Stop Loss Frameworks:**
- Delta-based stops
- Vega-based stops
- Absolute P&L stops
- Combination stops

## Indian Market Specifics

**NSE Index Options (Nifty 50):**
- American style (exercise any time)
- Settlement: Cash
- Expiry: Last Thursday of month
- Strike intervals: ₹100 intervals

**Stock Options:**
- Limited universe (liquid stocks only)
- American style
- Settlement: Physical delivery of shares
- Margin: SPAN + exposure margin

**Settlement & Expiry:**
- Expiry time: 3:30 PM on last Thursday
- Exercised options: Settlement next day
- Unexercised options: Expire worthless
- Position limits: Applied separately for options
