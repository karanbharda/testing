# Futures Contracts & F&O Market Rules

## NSE Index Futures

### NIFTY 50 Futures
**Contract Specifications:**
- Underlying: NIFTY 50 Index
- Contract size: Single unit (100x index points)
- Expiry: Last Thursday of month, 3:30 PM
- Tick size: 0.05 points
- Price band: None (but circuit breaker applies)

**Settlement:**
- Cash settlement only
- Final settlement rate: Closing price on expiry
- Mark-to-market: Daily at 3:30 PM
- P&L calculation: Difference × 100

**Margin Requirements:**
- Initial margin: ~12-15% of contract value
- Maintenance margin: ~7-10% of contract value
- SPAN algorithm: Risk-based margin calculation
- Exposure margin: Additional 5-10%

### Bank Nifty Futures
**Specifications:**
- Underlying: 12 large-cap banking stocks
- Contract size: 25 units
- More volatile than NIFTY 50
- Higher margin requirements

**Use Cases:**
- Sector-specific hedging
- Banking sector bets
- Correlation trading with equities
- Income generation through covered calls

## Stock Futures

**Contract Specifications:**
- Underlying: Individual stocks
- Trading hours: 9:15 AM - 3:30 PM
- Settlement: Monthly expiry
- Eligible securities: Limited to liquid stocks

**Price Bands:**
- Fixed upper/lower bands (10-20% based on stock category)
- Frozen stock: No trading at limit prices
- Halts: If bid/ask differs significantly

**Position Limits:**
- Market-wide limit: 20% of NSE market capitalization
- Client position limit: 1-2% of open interest
- May vary by stock liquidity
- Limits enforced by exchange in real-time

## Options on Futures

**Structure:**
- Options on stock index futures
- Options on stock futures (limited availability)
- European style for index, American for stocks

**Key Features:**
- Leverage: Multiple times than spot trading
- Time decay: Affects option value significantly
- Volatility impact: Options on futures more volatile

## Basis & Arbitrage

### Futures-Spot Basis
**Definition:** Difference between futures price and spot price
- Formula: Basis = Futures Price - Spot Price
- Positive basis: Normal market (futures premium)
- Negative basis: Unusual (reverse structure)

**Cost of Carry:**
- Interest cost: Cost of financing position
- Dividend yield: Expected dividends
- Storage cost: Typically zero for stocks
- Formula: Futures = Spot × e^((r-q)T)

### Calendar Spread (Futures)
**Strategy:**
- Buy near month, Sell far month futures
- Profit from basis convergence
- Lower capital requirement than outright position
- Used for yield enhancement

**Execution:**
- Initiate: Buy nearby, sell further expiry
- Convergence: Basis should narrow
- Close: Exit at defined profit target
- Risk: Basis widens instead of narrowing

## Leverage & Risk in Futures

**Leverage Effect:**
- 10:1 leverage typical for equity index futures
- ₹1 lakh margin controls ₹10 lakh exposure
- Gains/losses magnified 10x
- Risk management critical

**Price Movement Impact:**
- 1% move in index = 10% move in profit/loss
- Daily circuit breakers limit max loss to -20% usually
- Forced liquidation if margin falls below maintenance
- Gap risk on market opening

### Leverage Risk Management
**Position Sizing:**
- Risk per trade: 1-2% of capital
- Maximum position: 5-10% of portfolio
- Correlations: Monitor with existing positions
- Diversification: Spread across sectors

**Stop Loss Rules:**
- Hard stop loss: Must be set and monitored
- Profit targets: Defined exit points
- Time stops: Exit if thesis doesn't play out
- Margin alerts: Action at 10% above minimum margin

## Contract Rolling

**Rolling Strategy:**
- Exit near month position
- Enter far month position
- Timing: Typically 1-2 weeks before expiry
- Cost: Entry/exit commissions + potential slippage

**Roll Mechanics:**
- Simultaneous buy-sell: To minimize slippage
- Volume consideration: Far month is liquid enough
- Overnight risk: Gap at opening during roll

**Typical Roll Schedule:**
- Week 1 of month: Monitor coming expiry
- Week 3: Consider rolling if holding large positions
- Week 4: Definitely roll or exit before 3:30 PM final Thursday
- Post-expiry: Position automatically square off

## Index Futures vs Equity Spot

**Comparison:**
| Aspect | Futures | Spot |
|--------|---------|------|
| Leverage | 10:1 typical | No leverage |
| Holding cost | Carry cost | Dividend received |
| Margin | Yes | Yes (for margin trading) |
| Duration | Fixed expiry | Indefinite |
| Liquidity | Very high | Varies by stock |
| Tax | STCG after 1 year | Securities transaction tax |

## Settlement & Clearing

**Daily Settlement:**
- Time: 3:30 PM
- P&L calculation: Daily MTM
- Payment: T+1 settlement
- Settlement guarantee: NSCCL guarantees

**Expiry Settlement:**
- Final settlement rate: 4:00 PM closing price
- All positions squared: Automatic
- STT applicable: On final settlement
- Mark-to-market: Final P&L booked

## Advanced Strategies

### Synthetic Long Position
- Buy call + Sell put at same strike
- Creates synthetic futures-like payoff
- Lower capital than futures (no carry cost)
- Useful for dividend capture

### Futures-Options Calendar Spread
- Use futures + options for income
- Sell near-term options, hold futures hedge
- Roll options monthly
- Captures theta decay while hedged

### Index Arbitrage
- Exploit futures-spot basis discrepancies
- Buy cheaper, sell expensive
- Lock-in basis spread as profit
- Requires capital and market access

## Regulatory Framework

**SEBI Guidelines:**
- Minimum turnover for trading
- Circuit breaker implementation
- Surveillance monitoring
- Insider trading prevention

**Client Protections:**
- Fair valuation of derivatives
- Disclosure of risks
- Margin adequacy standards
- Dispute resolution mechanism

**Compliance Requirements:**
- Trade confirmation: Within 3 hours
- Statement delivery: Periodic
- Record keeping: 7 years minimum
- Audit trail maintenance
