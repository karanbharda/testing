"""
Main module for the Indian stock trading bot.
Integrates portfolio management, stock analysis, and ML/RL models.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import traceback
import warnings
from dotenv import load_dotenv
import signal
import sys
import argparse

from .portfolio import DataFeed, VirtualPortfolio, TradingExecutor, PerformanceReport, PortfolioTracker
from .analysis import Stock
from .models import MLModels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_bot_india.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def is_market_open():
    """Check if the Indian stock market is currently open."""
    try:
        import pytz

        # Get current time in IST
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)

        # NSE trading hours: 9:15 AM to 3:30 PM IST
        market_open_time = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

        # Check if it's a weekday (Monday=0, Sunday=6)
        if now_ist.weekday() >= 5:  # Saturday or Sunday
            logger.info("Market is closed - Weekend.")
            return False

        # Check if current time is within trading hours
        if market_open_time <= now_ist <= market_close_time:
            logger.info(f"Market is open. Current IST time: {now_ist.strftime('%H:%M:%S')}")
            return True
        else:
            logger.info(f"Market is closed. Opens at {market_open_time.strftime('%H:%M:%S')}, closes at {market_close_time.strftime('%H:%M:%S')}. Current IST time: {now_ist.strftime('%H:%M:%S')}")
            return False

    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        # Fallback: assume market is open during weekday business hours
        now = datetime.now()
        if now.weekday() < 5 and 9 <= now.hour < 16:
            logger.info("Using fallback market check - assuming market is open.")
            return True
        return False

def prepare_data(ticker, period="3y", interval="1d"):
    """Prepare historical data for analysis."""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period, interval=interval)
        
        if history.empty:
            logger.error(f"No historical data available for {ticker}")
            return None
        
        # Calculate technical indicators
        history['SMA_20'] = history['Close'].rolling(window=20).mean()
        history['SMA_50'] = history['Close'].rolling(window=50).mean()
        history['SMA_200'] = history['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = history['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.where(loss != 0, 1e-10)
        history['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = history['Close'].ewm(span=12, adjust=False).mean()
        exp2 = history['Close'].ewm(span=26, adjust=False).mean()
        history['MACD'] = exp1 - exp2
        history['MACD_Signal'] = history['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        history['BB_Middle'] = history['Close'].rolling(window=20).mean()
        std_dev = history['Close'].rolling(window=20).std()
        history['BB_Upper'] = history['BB_Middle'] + (std_dev * 2)
        history['BB_Lower'] = history['BB_Middle'] - (std_dev * 2)
        
        # Calculate ATR
        high_low = history['High'] - history['Low']
        high_close = (history['High'] - history['Close'].shift()).abs()
        low_close = (history['Low'] - history['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        history['ATR'] = true_range.rolling(window=14).mean()
        
        # Calculate OBV
        history['OBV'] = (np.sign(history['Close'].diff()) * history['Volume']).fillna(0).cumsum()
        
        # Fill NaN values
        history = history.fillna(method='bfill').fillna(method='ffill')
        
        return history
    except Exception as e:
        logger.error(f"Error preparing data for {ticker}: {e}")
        return None

def analyze_stock(ticker, config, stock_analyzer):
    """Analyze a stock and generate trading signals."""
    try:
        logger.info(f"Analyzing {ticker}...")
        
        # Prepare data
        history = prepare_data(ticker, period=config["period"], interval="1d")
        if history is None:
            return None
        
        # Get current price
        current_price = history['Close'].iloc[-1]
        
        # Calculate technical indicators
        sma_20 = history['SMA_20'].iloc[-1]
        sma_50 = history['SMA_50'].iloc[-1]
        sma_200 = history['SMA_200'].iloc[-1]
        rsi = history['RSI'].iloc[-1]
        macd = history['MACD'].iloc[-1]
        macd_signal = history['MACD_Signal'].iloc[-1]
        
        # Generate technical signals
        tech_signals = {
            "price_above_sma20": current_price > sma_20,
            "price_above_sma50": current_price > sma_50,
            "price_above_sma200": current_price > sma_200,
            "sma20_above_sma50": sma_20 > sma_50,
            "sma50_above_sma200": sma_50 > sma_200,
            "rsi_oversold": rsi < 30,
            "rsi_overbought": rsi > 70,
            "macd_bullish": macd > macd_signal,
            "macd_bearish": macd < macd_signal
        }
        
        # Get sentiment analysis
        sentiment = stock_analyzer.fetch_combined_sentiment(ticker)
        
        # Calculate sentiment score
        total_mentions = sum(sentiment["aggregated"].values())
        if total_mentions > 0:
            sentiment_score = sentiment["aggregated"]["positive"] / total_mentions
        else:
            sentiment_score = 0.5  # Neutral if no mentions
        
        # Calculate MPT metrics
        mpt_metrics = stock_analyzer.calculate_mpt_metrics(history, config["benchmark_tickers"])
        
        # Generate buy/sell signals
        buy_score = 0
        sell_score = 0
        
        # Technical indicators (50% weight)
        if tech_signals["price_above_sma20"]: buy_score += 0.05
        else: sell_score += 0.05
        
        if tech_signals["price_above_sma50"]: buy_score += 0.1
        else: sell_score += 0.1
        
        if tech_signals["price_above_sma200"]: buy_score += 0.1
        else: sell_score += 0.1
        
        if tech_signals["sma20_above_sma50"]: buy_score += 0.1
        else: sell_score += 0.1
        
        if tech_signals["sma50_above_sma200"]: buy_score += 0.15
        else: sell_score += 0.15
        
        if tech_signals["rsi_oversold"]: buy_score += 0.1
        if tech_signals["rsi_overbought"]: sell_score += 0.1
        
        if tech_signals["macd_bullish"]: buy_score += 0.15
        if tech_signals["macd_bearish"]: sell_score += 0.15
        
        # Sentiment analysis (25% weight)
        sentiment_factor = 0.25
        if sentiment_score > 0.6:
            buy_score += sentiment_factor
        elif sentiment_score < 0.4:
            sell_score += sentiment_factor
        
        # ML/RL predictions (25% weight)
        ml_models = MLModels()
        adv_history = ml_models.generate_adversarial_financial_data(history)
        
        # Train RL model
        rl_result = stock_analyzer.train_rl_with_adversarial_events(
            history=adv_history,
            ml_predicted_price=current_price * 1.05,  # Simplified prediction
            current_price=current_price,
            num_episodes=50,
            adversarial_freq=0.2,
            max_event_magnitude=0.1
        )
        
        if rl_result["success"]:
            if rl_result["recommendation"] == "BUY":
                buy_score += 0.25
            elif rl_result["recommendation"] == "SELL":
                sell_score += 0.25
        
        # Generate final recommendation
        if buy_score > sell_score + 0.3:
            recommendation = "STRONG BUY"
        elif buy_score > sell_score + 0.1:
            recommendation = "BUY"
        elif sell_score > buy_score + 0.3:
            recommendation = "STRONG SELL"
        elif sell_score > buy_score + 0.1:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        # Calculate risk metrics
        volatility = history['Close'].pct_change().std() * np.sqrt(252)
        sharpe_ratio = mpt_metrics["sharpe_ratio"] if mpt_metrics["sharpe_ratio"] != "N/A" else 0
        
        # Determine trend direction
        trend_direction = "UPTREND" if sma_50 > sma_200 else "DOWNTREND"
        
        # Generate detailed explanation
        explanation = stock_analyzer._generate_detailed_recommendation(
            stock_data={
                "name": ticker,
                "symbol": ticker,
                "current_price": {"INR": current_price}
            },
            recommendation=recommendation,
            buy_score=buy_score,
            sell_score=sell_score,
            price_to_sma200=current_price / sma_200,
            trend_direction=trend_direction,
            sentiment_score=sentiment_score,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio
        )
        
        return {
            "ticker": ticker,
            "current_price": current_price,
            "recommendation": recommendation,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "technical_signals": tech_signals,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "mpt_metrics": mpt_metrics,
            "rl_recommendation": rl_result.get("recommendation", "HOLD"),
            "explanation": explanation,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        logger.error(traceback.format_exc())
        return None

def execute_trades(analysis_results, portfolio, executor, config):
    """Execute trades based on analysis results."""
    try:
        for result in analysis_results:
            if result is None:
                continue

            ticker = result["ticker"]
            recommendation = result["recommendation"]
            current_price = result["current_price"]

            # Calculate position size (simple 1% of portfolio per trade)
            portfolio_value = portfolio.get_value({ticker: {"price": current_price}})
            position_size = portfolio_value * 0.01  # 1% risk per trade
            qty = max(1, int(position_size / current_price))

            # Execute trades based on recommendation
            if recommendation in ["STRONG BUY", "BUY"]:
                logger.info(f"Executing BUY order for {ticker}: {qty} shares at ₹{current_price:.2f}")
                result = executor.execute_trade(
                    action="BUY",
                    ticker=ticker,
                    qty=qty,
                    price=current_price,
                    stop_loss=current_price * 0.95,  # 5% stop loss
                    take_profit=current_price * 1.10  # 10% take profit
                )
                if result["success"]:
                    portfolio.buy(ticker, qty, current_price)
                    logger.info(f"Successfully bought {qty} shares of {ticker}")
                else:
                    logger.error(f"Failed to buy {ticker}: {result['message']}")

            elif recommendation in ["STRONG SELL", "SELL"]:
                # Only sell if we have holdings
                if ticker in portfolio.holdings and portfolio.holdings[ticker]["qty"] > 0:
                    holdings_qty = portfolio.holdings[ticker]["qty"]
                    sell_qty = min(qty, holdings_qty)
                    logger.info(f"Executing SELL order for {ticker}: {sell_qty} shares at ₹{current_price:.2f}")
                    result = executor.execute_trade(
                        action="SELL",
                        ticker=ticker,
                        qty=sell_qty,
                        price=current_price,
                        stop_loss=current_price * 1.05,  # 5% stop loss for short
                        take_profit=current_price * 0.90  # 10% take profit for short
                    )
                    if result["success"]:
                        portfolio.sell(ticker, sell_qty, current_price)
                        logger.info(f"Successfully sold {sell_qty} shares of {ticker}")
                    else:
                        logger.error(f"Failed to sell {ticker}: {result['message']}")
                else:
                    logger.info(f"No holdings to sell for {ticker}")

            else:
                logger.info(f"HOLD recommendation for {ticker} - no action taken")

    except Exception as e:
        logger.error(f"Error executing trades: {e}")
        logger.error(traceback.format_exc())

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Bot shutdown signal received. Shutting down gracefully...")
    print("\n🤖 Bot shut down successfully!")
    sys.exit(0)

def main():
    """Main trading bot function."""
    try:
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("Starting Indian Stock Trading Bot...")

        # Parse command line arguments for mode
        parser = argparse.ArgumentParser(description='Indian Stock Trading Bot')
        parser.add_argument('--mode', choices=['live', 'paper'],
                           default=os.getenv('TRADING_MODE', 'paper'),
                           help='Trading mode: live or paper (default: paper)')
        args = parser.parse_args()

        # Configuration with risk management
        config = {
            "tickers": [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
                "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS", "LT.NS",
                "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS", "NESTLEIND.NS"
            ],
            "starting_balance": 1000000,  # ₹10 lakh
            "current_portfolio_value": 1000000,  # Initial portfolio value
            "current_pnl": 0,  # Initial PnL
            "mode": args.mode,
            "dhan_client_id": os.getenv("DHAN_CLIENT_ID"),
            "dhan_access_token": os.getenv("DHAN_ACCESS_TOKEN"),
            "period": "3y",
            "prediction_days": 30,
            "benchmark_tickers": ["^NSEI"],
            "trading_threshold": 0.15,  # Minimum signal strength to trade
            "max_positions": 10,  # Maximum number of positions
            "risk_per_trade": 0.01,  # 1% risk per trade
            "check_interval": 300,  # Check every 5 minutes
            # Risk management settings from .env
            "stop_loss_pct": float(os.getenv("STOP_LOSS_PCT", "0.05")),
            "max_capital_per_trade": float(os.getenv("MAX_CAPITAL_PER_TRADE", "0.25")),
            "max_trade_limit": int(os.getenv("MAX_TRADE_LIMIT", "10"))
        }

        # Display mode information
        mode_display = "🔴 LIVE TRADING" if args.mode == "live" else "📝 PAPER TRADING"
        logger.info(f"Starting Indian Stock Trading Bot in {mode_display} mode")

        # Display startup banner
        print("\n" + "="*60)
        print("🚀 INDIAN STOCK TRADING BOT")
        print("="*60)
        print(f"📊 Mode: {mode_display}")
        print(f"💰 Starting Balance: ₹{config['starting_balance']:,}")
        print(f"🛡️  Stop Loss: {config['stop_loss_pct']*100}%")
        print(f"💼 Max Capital per Trade: {config['max_capital_per_trade']*100}%")
        print(f"📈 Max Trade Limit: {config['max_trade_limit']}")
        print(f"⏰ Check Interval: {config['check_interval']} seconds")
        print(f"📁 Portfolio File: {config['mode']}_mode")
        print("="*60)

        if args.mode == "live":
            if not config["dhan_client_id"] or not config["dhan_access_token"]:
                logger.error("Dhan API credentials not found in .env file. Cannot run in live mode.")
                print("❌ Error: Dhan API credentials required for live trading!")
                return
            print("⚠️  WARNING: Live trading mode enabled. Real money will be used!")
            confirmation = input("Type 'CONFIRM' to proceed with live trading: ")
            if confirmation != "CONFIRM":
                print("❌ Live trading cancelled.")
                return

        # Initialize components
        data_feed = DataFeed(config["tickers"])
        portfolio = VirtualPortfolio(config)
        executor = TradingExecutor(portfolio, config)
        performance_report = PerformanceReport(portfolio)
        portfolio_tracker = PortfolioTracker(portfolio, config)

        # Initialize stock analyzer
        stock_analyzer = Stock(
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            reddit_user_agent=os.getenv("REDDIT_USER_AGENT")
        )

        logger.info("Bot initialized successfully. Starting trading loop...")

        # Main trading loop
        while True:
            try:
                # Check if market is open
                if not is_market_open():
                    logger.info("Market is closed. Waiting for next check...")
                    time.sleep(config["check_interval"])
                    continue

                logger.info(f"Market is open. Starting analysis cycle in {config['mode'].upper()} mode...")

                # Analyze all tickers
                analysis_results = []
                for ticker in config["tickers"]:
                    result = analyze_stock(ticker, config, stock_analyzer)
                    if result:
                        analysis_results.append(result)
                        mode_indicator = "🔴" if config['mode'] == "live" else "📝"
                        logger.info(f"{mode_indicator} {ticker}: {result['recommendation']} (Buy: {result['buy_score']:.3f}, Sell: {result['sell_score']:.3f})")

                # Execute trades based on analysis
                execute_trades(analysis_results, portfolio, executor, config)

                # Log portfolio metrics
                portfolio_tracker.log_metrics()

                # Generate performance report
                report = performance_report.generate_report()
                logger.info(f"Daily Performance: ROI: {report['roi_today']:.2f}%, Trades: {report['trades_executed']}")

                # Wait before next cycle
                logger.info(f"Analysis cycle complete. Waiting {config['check_interval']} seconds...")
                time.sleep(config["check_interval"])

            except KeyboardInterrupt:
                logger.info("Bot shutdown requested by user.")
                break
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(60)  # Wait 1 minute before retrying

        logger.info("Indian Stock Trading Bot shut down successfully.")

    except Exception as e:
        logger.error(f"Critical error in main function: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
