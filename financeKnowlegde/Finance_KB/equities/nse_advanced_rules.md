# NSE Advanced Trading Rules & Regulations

## NSE Market Structure & Architecture

### Market Segments
1. **Capital Market (CM)**
   - Equities: Cash segment (T+2 settlement)
   - Derivatives: F&O segment with monthly expiry
   - Wholesale Debt Market (WDM)
   - Currency Derivatives

2. **Wholesale Debt Market (WDM)**
   - Government Securities (G-Secs)
   - Corporate Bonds
   - Treasury Bills
   - State Development Loans (SDLs)

### Trading Hours & Sessions
- **Pre-Open Session**: 09:00 AM - 09:15 AM (Order accumulation, no matching)
- **Market Open Session**: 09:15 AM - 3:30 PM (Continuous trading)
- **Closing Auction**: 3:30 PM - 3:40 PM
- **Post Market Session**: For specific instruments

### Trading Halts & Circuit Breakers

**Circuit Breaker Levels:**
1. **Level 1**: 10% index movement → 15-minute halt
2. **Level 2**: 15% index movement → 15-minute halt
3. **Level 3**: 20% index movement → Market close for the day

**Stock-level Circuit Breakers:**
- Price bands apply individually for each stock
- Upper/Lower circuit: Typically ±5% to ±10% based on category
- Frozen stock: No trading allowed at that price

### Settlement & Clearing

**T+2 Settlement:**
- Pay-in (T+1): 11:00 AM for securities & cash
- Auction: 11:00 AM - 2:00 PM for failed settlements
- Pay-out (T+2): 1:00 PM - 4:00 PM

**Rolling Settlement:**
- No netting across settlement cycles
- Each settlement is independent
- Fail in one cycle doesn't offset another

### Risk Management & Margin Framework

**Margin Requirements:**
1. **Initial Margin**: Minimum capital required to initiate position
2. **Maintenance Margin**: Minimum to hold position
3. **Exposure Margin**: Additional buffer (10-30% of position value)
4. **Volatility Margin**: Based on 99th percentile movement

**Position Limits:**
- Client-level position limits (exposure % of open interest)
- Market-level position limits
- Index options: More liberal limits
- Individual stocks: Stricter limits

**Mark-to-Market (MTM):**
- Daily settlement at 3:30 PM
- Client accounts credited/debited based on closing prices
- Margin shortfall triggers liquidation

### Index & Order Types

**Major Indices:**
- NIFTY 50: Blue-chip large caps
- NIFTY NEXT 50: Next 50 companies
- NIFTY 100: Top 100 companies
- Sectoral indices: Bank Nifty, IT Index, etc.

**Order Types:**
1. **Market Order**: Immediate execution at best price
2. **Limit Order**: Execute at/better than specified price
3. **Stop Loss Order**: Converts to market order when trigger hit
4. **Immediate or Cancel (IOC)**: Execute or cancel instantly
5. **Good Till Cancelled (GTC)**: Valid until manually cancelled

### Price Improvement & Execution Quality

**Price Improvement Rules:**
- If limit order better than best bid/ask, executes at better price
- Price improvement mandatory for client orders
- Regulated tick size: 0.05 for most stocks

**Execution Mechanisms:**
- Order matching engine: Price-time priority
- Multi-tier matching for better prices
- Auction for trades at specified prices

## Advanced Concepts

### Derivatives Market Rules

**Contract Specifications:**
- Monthly expiry: Last Thursday of each month
- Expiry time: 3:30 PM
- Settlement: Physical (equities) or cash (indices)

**Options Specific:**
- American style (exercise any time before expiry)
- Strike intervals based on index level
- No shorting for buying options, full margin for selling

### Special Settlement Rules

**IPO Allocation & Listing:**
- T+3 listing for equity IPOs
- T+1 settlement for allotted quantities
- Lock-in period: As per company norms

**Corporate Actions:**
- Ex-date: Dividend/bonus not credited beyond this date
- Record date: Official date for corporate actions
- Payment date: Actual credit to demat account

## Regulatory Framework

### SEBI Requirements
- Real-time risk monitoring
- Stress testing of members
- Client protection fund
- Know Your Client (KYC) mandatory
- Anti-money laundering (AML) compliance

### Insider Trading Rules (SEBI PIT Regulations)
- Trading windows: Restricted periods
- Disclosure requirements for transactions
- Consequences: Civil penalties and market bans

### Price Manipulation Rules
- Artificial volume creation prohibited
- Painting the tape illegal
- Matched trades without genuine transfer
- Consequences: Criminal prosecution possible

## Liquidity & Market Microstructure

**Liquidity Tiers:**
1. **Excellent**: NIFTY 50 components, typical bid-ask 0.5-1 paisa
2. **Good**: NIFTY NEXT 50, bid-ask 1-5 paise
3. **Fair**: Mid-cap, bid-ask 5-20 paise
4. **Low**: Small-cap, bid-ask highly variable

**Order Book Dynamics:**
- Level 1: Best bid/ask
- Level 2-5: Additional price levels (depth 5 available)
- Time priority: First order at price level gets priority
- Partial fills possible for large orders

## Best Practices & Risk Guidelines

**Trade Planning:**
- Define stop loss before entry
- Use position sizing rules
- Monitor correlation with portfolio
- Maintain adequate margin buffer (20-30%)

**Risk Management:**
- Daily P&L monitoring
- Position limit checks
- Margin adequacy verification
- Volatility-adjusted sizing

**Compliance:**
- Regular KYC updates
- Record keeping for 7 years
- Trade reporting accuracy
- Settlement failure avoidance
