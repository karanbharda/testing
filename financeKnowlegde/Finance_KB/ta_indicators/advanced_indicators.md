# Technical Analysis Indicators & Market Patterns

## Trend Indicators

### Moving Averages (MA)
**Types & Usage:**
1. **Simple Moving Average (SMA)**
   - Arithmetic mean of last N periods
   - 50-day SMA: Medium-term trend
   - 200-day SMA: Long-term trend
   - Equal weighting: Responsive to latest data

2. **Exponential Moving Average (EMA)**
   - Weighted towards recent prices
   - 12-EMA & 26-EMA: Short-term signals
   - Faster than SMA in responding to price changes
   - More relevant for current market conditions

**Trading Rules:**
- Price > MA50 > MA200: Uptrend
- Price < MA50 < MA200: Downtrend
- MA crossover: Generate buy/sell signals
- MA slope: Indicates trend strength

**Limitations:**
- Lagging indicator (catches trend after start)
- Whipsaws in range-bound markets
- False signals during consolidation
- Not effective in choppy markets

### Exponential Moving Convergence Divergence (MACD)
**Components:**
- MACD Line: 12-EMA minus 26-EMA
- Signal Line: 9-EMA of MACD line
- Histogram: MACD minus Signal line

**Trading Signals:**
1. **MACD Crossover**
   - MACD > Signal line: Bullish (buy signal)
   - MACD < Signal line: Bearish (sell signal)
   - Positive histogram: Upside momentum
   - Negative histogram: Downside momentum

2. **Divergence**
   - Price higher, MACD lower: Bearish divergence (sell)
   - Price lower, MACD higher: Bullish divergence (buy)
   - Signals potential reversal

**Histogram Analysis:**
- Growing histogram: Momentum increasing
- Shrinking histogram: Momentum decreasing
- Crosses zero: Inflection point in momentum

### Average Directional Index (ADX)
**Interpretation:**
- ADX < 20: No clear trend (range-bound)
- ADX 20-40: Developing trend
- ADX 40+: Strong trend
- Not directional (shows strength only)

**+DI vs -DI:**
- +DI > -DI: Uptrend
- -DI > +DI: Downtrend
- Crossover: Potential reversal

## Momentum Indicators

### Relative Strength Index (RSI)
**Calculation:**
- RSI = 100 - [100 / (1 + RS)]
- RS = Average gain / Average loss (14 periods typical)
- Range: 0-100

**Levels:**
- RSI > 70: Overbought (potential sell)
- RSI < 30: Oversold (potential buy)
- RSI = 50: Neutral point
- Divergence from price: Reversal signal

**Trading Applications:**
1. **Oversold Bounce**
   - RSI < 20: Extreme oversold
   - Bounce likely in uptrend
   - Entry: When RSI rises above 30

2. **Overbought Pullback**
   - RSI > 80: Extreme overbought
   - Pullback likely in downtrend
   - Entry: When RSI falls below 70

**Limitations:**
- False signals in strong trends
- RSI can stay overbought/oversold for extended periods
- Less effective in choppy markets
- Works better in range-bound markets

### Stochastic Oscillator
**Fast vs Slow Stochastic:**
- Fast: K=14, D=3 (more sensitive, more false signals)
- Slow: K=14, D=3 applied to fast (smoother)
- %K > %D: Upside momentum
- %K < %D: Downside momentum

**Signal Generation:**
- %K > 80: Overbought
- %K < 20: Oversold
- Crossover: %K above/below %D
- Divergence with price: Reversal setup

### Commodity Channel Index (CCI)
**Calculation:**
- Based on typical price and moving average
- Identifies cyclical patterns
- Extreme readings: ±100
- Useful in trending markets

## Volume Indicators

### On-Balance Volume (OBV)
**Concept:**
- Volume added on up days, subtracted on down days
- Cumulative indicator
- OBV rising: Bullish (buying pressure)
- OBV falling: Bearish (selling pressure)

**Divergence Trading:**
- Price up, OBV down: Bullish divergence (buy)
- Price down, OBV up: Bearish divergence (sell)
- Confirms trend strength

### Accumulation/Distribution (A/D) Line
**Calculation:**
- Incorporates close, high, low, volume
- Money flow indicator
- Rising line: Accumulation
- Falling line: Distribution

**Multi-timeframe Analysis:**
- Confirm daily signals with weekly A/D
- Divergence: Strength/weakness indicator
- Use with price action for confirmation

### Volume Rate of Change
- Compares current volume to historical average
- Volume spike: Conviction in move
- Declining volume: Weakening trend
- Volume dry-up: Reversal signal

## Support & Resistance

### Identification Methods

**Price-Based:**
- Previous highs: Resistance
- Previous lows: Support
- Round numbers: Psychological levels
- Exponential moves: Invalidate old levels

**Indicator-Based:**
- Moving averages: Dynamic support/resistance
- Pivot points: Mathematical levels
- Fibonacci levels: Retracement zones
- Bollinger Bands: Range boundaries

### Trading Support & Resistance

**Breakout Trading:**
- Resistance break: Buy signal
- Support break: Sell signal
- Volume required: Confirm breakout
- Retest: Price often returns to break level

**Bounce Trading:**
- Price rebounds from support: Buy
- Price rebounds from resistance: Sell
- Multiple tests: Level becomes stronger
- Failed test: Level may break

## Pattern Recognition

### Candlestick Patterns

**Single Candle:**
- Doji: Indecision (confirmation needed)
- Hammer: Bottom reversal in downtrend
- Shooting star: Top reversal in uptrend

**Two Candle:**
- Bullish engulfing: Reversal up
- Bearish engulfing: Reversal down
- Piercing line: Reversal up
- Dark cloud cover: Reversal down

### Chart Patterns

**Trend Continuation:**
- Triangle: Consolidation before breakout
- Flag: Continuation after sharp move
- Pennant: Similar to flag, smaller
- Rectangle: Range consolidation

**Trend Reversal:**
- Head and shoulders: Major top
- Inverse H&S: Major bottom
- Double top: Reversal top
- Double bottom: Reversal bottom

## Volatility Indicators

### Bollinger Bands
**Components:**
- Middle band: 20-period SMA
- Upper band: Middle + 2 std dev
- Lower band: Middle - 2 std dev

**Trading Rules:**
- Price > Upper band: Potential sell (overbought)
- Price < Lower band: Potential buy (oversold)
- Band squeeze: Low volatility (breakout coming)
- Band expansion: High volatility (trend moving)

**Bandwidth Strategy:**
- Narrow bandwidth: Range-bound market
- Wide bandwidth: Trending market
- Use 5-period bands for tighter stops

### Average True Range (ATR)
**Calculation:**
- True Range: High-Low, High-Close, Low-Close (max)
- ATR: Average of True Range (14 periods typical)
- No direction, just magnitude

**Applications:**
- Position sizing: Risk = ATR × shares
- Stop loss placement: Entry ± 2 ATR
- Volatility clustering: ATR changes over time
- Breakout threshold: Move > 2 ATR is significant

## Ichimoku Cloud

**Components:**
1. **Tenkan-Sen**: 9-period high-low average
2. **Kijun-Sen**: 26-period high-low average
3. **Senkou Span A**: Average of Tenkan & Kijun (26 periods ahead)
4. **Senkou Span B**: 52-period high-low average (26 periods ahead)
5. **Chikou Span**: Close price (26 periods back)

**Cloud Interpretation:**
- Price > Cloud: Uptrend
- Price < Cloud: Downtrend
- Cloud thickness: Support/Resistance strength
- Cloud color: Green (bullish), Red (bearish)

## Multi-timeframe Analysis

**Framework:**
- Long-term: 200-day SMA for trend
- Intermediate: 50-day SMA for support/resistance
- Short-term: 20-day SMA for entry/exit
- Intra-day: 5-min/15-min for execution

**Strategy Integration:**
- Confirm direction on higher timeframe
- Enter on lower timeframe signal
- Avoid counter-trend entries on lower timeframes
- Use multiple indicators for confluence

## Indicator Limitations

**Common Pitfalls:**
- Over-reliance on single indicator
- Lagging nature of most indicators
- False signals in choppy markets
- Need for entry/exit rules clarity
- Position sizing discipline critical
- Risk management paramount

**Best Practices:**
- Use 2-3 indicators maximum
- Combine leading + lagging indicators
- Confirm with price action and volume
- Backtest on historical data
- Forward test before live trading
