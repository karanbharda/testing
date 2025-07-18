# Indian Stock Trading Bot - Modular Structure

This folder contains the modularized Indian stock trading bot code, separated into logical components for better maintainability and organization.

## File Structure

### `__init__.py`
Package initialization file that defines the module structure and version information.

### `portfolio.py`
**Portfolio Management Module**
- `DataFeed`: Fetches live stock prices using yfinance
- `VirtualPortfolio`: Manages virtual portfolio with cash, holdings, and trade logging
- `PaperExecutor`: Executes paper trades using Dhan API
- `PerformanceReport`: Generates daily performance reports
- `PortfolioTracker`: Logs and tracks portfolio metrics

### `analysis.py`
**Stock Analysis and Sentiment Analysis Module**
- `Stock`: Main class for stock analysis including:
  - Sentiment analysis from multiple sources (NewsAPI, GNews, Reddit, Google News)
  - Financial data retrieval (income statement, balance sheet, cash flow)
  - Technical analysis and MPT metrics calculation
  - Reinforcement learning with adversarial events
  - Currency conversion and exchange rate handling

### `models.py`
**Machine Learning and Reinforcement Learning Models Module**
- `MLModels`: Base class for ML operations
- `AdversarialStockTradingEnv`: Gym environment for RL training with adversarial events
- `AdversarialQLearningAgent`: Q-learning agent for trading decisions
- `LSTMModel`: LSTM neural network for price prediction
- `TransformerModel`: Transformer neural network for price prediction
- Adversarial training methods for robust model training

### `main.py`
**Main Trading Bot Logic**
- Market status checking (NSE trading hours)
- Data preparation with technical indicators
- Stock analysis integration
- Trade execution logic
- Main trading loop with graceful shutdown handling
- Configuration management

## Key Features

### Exact Logic Preservation
The modularized code maintains the exact logic and flow from the original `testindia.py` file without adding extra features, as per user preferences.

### Comprehensive Analysis
- **Technical Indicators (50% weight)**: SMA, RSI, MACD, Bollinger Bands, ATR, OBV
- **Sentiment Analysis (25% weight)**: Multi-source news and social media sentiment
- **ML/RL Predictions (25% weight)**: Adversarial training and reinforcement learning

### Risk Management
- Position sizing based on portfolio percentage
- Stop-loss and take-profit levels
- Maximum position limits
- Risk per trade controls

### Indian Market Focus
- NSE trading hours and calendar integration
- Indian stock tickers (.NS suffix)
- INR currency handling
- Dhan API integration for Indian brokers

## Usage

Run the trading bot using the main entry point:

```python
from india.main import main
main()
```

Or run the original file:

```bash
python testindia.py
```

## Configuration

The bot uses environment variables for API keys and credentials:
- `DHAN_CLIENT_ID`: Dhan API client ID
- `DHAN_ACCESS_TOKEN`: Dhan API access token
- `NEWSAPI_KEY`: NewsAPI key for sentiment analysis
- `GNEWS_API_KEY`: GNews API key
- `REDDIT_CLIENT_ID`: Reddit API client ID
- `REDDIT_CLIENT_SECRET`: Reddit API client secret
- `REDDIT_USER_AGENT`: Reddit API user agent

## Dependencies

The bot requires the following Python packages:
- yfinance
- pandas
- numpy
- torch
- sklearn
- dhanhq
- praw
- gnews
- vaderSentiment
- pandas_market_calendars
- requests
- python-dotenv

## Logging

Detailed logging is provided for:
- Trading signals and decision-making process
- Portfolio metrics and performance
- Error handling and debugging
- Market status and trading activities

## Graceful Shutdown

The bot handles Ctrl+C gracefully with a proper shutdown message, as preferred by the user.
