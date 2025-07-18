"""
Stock analysis and sentiment analysis module for the Indian stock trading bot.
Includes the Stock class with methods for sentiment analysis, financial data,
and technical analysis.
"""

import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw
from gnews import GNews
from urllib.parse import quote
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import HTTPError, RequestException

logger = logging.getLogger(__name__)

class Stock:
    COINGECKO_API = "https://api.coingecko.com/api/v3"
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
    STOCKTWITS_BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol"
    FMPC_API_KEY = os.getenv("FMPC_API_KEY")
    MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
    SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
    SANTIMENT_API = "https://api.santiment.net"
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

    def __init__(self, reddit_client_id=None, reddit_client_secret=None, reddit_user_agent=None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        if reddit_client_id and reddit_client_secret and reddit_user_agent:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
        else:
            self.reddit = None
        self.COINGECKO_API = "https://api.coingecko.com/api/v3"
        self.last_rate_fetch = None
        self.rate_cache = None
        self.cache_duration = 300
        self.google_news = GNews()

    def fetch_exchange_rates(self):
        """Fetch real-time exchange rates for INR, EUR, BTC, ETH"""
        if (self.last_rate_fetch and 
            self.rate_cache and 
            (datetime.now() - self.last_rate_fetch).total_seconds() < self.cache_duration):
            return self.rate_cache

        try:
            url = f"{self.COINGECKO_API}/simple/price?ids=bitcoin,ethereum&vs_currencies=usd,inr,eur"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            btc_usd = data.get("bitcoin", {}).get("usd", 1)
            rates = {
                "bitcoin": {"usd": btc_usd},
                "ethereum": {"usd": data.get("ethereum", {}).get("usd", 1)},
                "inr": {"usd": data.get("bitcoin", {}).get("inr", 1) / btc_usd if btc_usd != 0 else 83},
                "eur": {"usd": data.get("bitcoin", {}).get("eur", 1) / btc_usd if btc_usd != 0 else 0.95}
            }

            self.rate_cache = rates
            self.last_rate_fetch = datetime.now()
            return rates
        except Exception as e:
            logger.error(f"Error fetching exchange rates: {e}")
            fallback = {
                "bitcoin": {"usd": 85000},
                "ethereum": {"usd": 1633},
                "inr": {"usd": 86},
                "eur": {"usd": 0.95}
            }
            self.rate_cache = fallback
            self.last_rate_fetch = datetime.now()
            return fallback

    def convert_price(self, price, exchange_rates):
        """Convert price to different currencies including crypto and fiat"""
        if not isinstance(price, (int, float)) or price == "N/A":
            return {"INR": price}
        try:
            return {
                "INR": round(float(price), 2),
                "USD": round(float(price) / exchange_rates.get("inr", {}).get("usd", 1), 2),
                "EUR": round(float(price) / exchange_rates.get("inr", {}).get("usd", 1) * exchange_rates.get("eur", {}).get("usd", 1), 2),
                "BTC": round(float(price) / exchange_rates.get("inr", {}).get("usd", 1) / exchange_rates.get("bitcoin", {}).get("usd", 1), 8),
                "ETH": round(float(price) / exchange_rates.get("inr", {}).get("usd", 1) / exchange_rates.get("ethereum", {}).get("usd", 1), 8)
            }
        except Exception as e:
            logger.error(f"Error converting price: {e}")
            return {
                "INR": round(float(price), 2),
                "USD": price,
                "EUR": price,
                "BTC": price,
                "ETH": price
            }

    def convert_np_types(self, data):
        if isinstance(data, dict):
            return {k: self.convert_np_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_np_types(item) for item in data]
        elif isinstance(data, (np.integer, np.int64)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        elif pd.isna(data):
            return None
        else:
            return data

    def newsapi_sentiment(self, ticker):
        try:
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={self.NEWSAPI_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            if "articles" in data:
                for article in data["articles"][:10]:
                    description = article.get("description", "") or article.get("content", "")[:200]
                    sentiment = self.sentiment_analyzer.polarity_scores(description)
                    if sentiment["compound"] > 0.1:
                        sentiments["positive"] += 1
                    elif sentiment["compound"] < -0.1:
                        sentiments["negative"] += 1
                    else:
                        sentiments["neutral"] += 1
            return sentiments
        except Exception as e:
            logger.error(f"Error fetching NewsAPI sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(HTTPError))
    def _make_request(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response

    def gnews_sentiment(self, ticker):
        try:
            # Remove .NS suffix from ticker
            query = ticker.replace(".NS", "")
            query = quote(query)  # Encode query for URL
            url = f"https://gnews.io/api/v4/search?q={query}&lang=en&country=in&token={self.GNEWS_API_KEY}"
            logger.debug(f"Requesting GNews API: {url}")
            
            response = self._make_request(url)
            data = response.json()
            logger.debug(f"Response status: {response.status_code}, content: {response.text[:200]}")
            
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            if not data or "articles" not in data or not data["articles"]:
                logger.warning(f"No articles found for ticker {ticker}. Response: {response.text[:200]}")
                return sentiments

            for article in data["articles"][:10]:
                description = article.get("description", "") or article.get("content", "")[:200]
                sentiment = self.sentiment_analyzer.polarity_scores(description)
                if sentiment["compound"] > 0.1:
                    sentiments["positive"] += 1
                elif sentiment["compound"] < -0.1:
                    sentiments["negative"] += 1
                else:
                    sentiments["neutral"] += 1
            return sentiments

        except Exception as e:
            logger.error(f"Error fetching GNews sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    def reddit_sentiment(self, ticker):
        if not self.reddit:
            return {"positive": 0, "negative": 0, "neutral": 0}
        try:
            subreddit = self.reddit.subreddit("all")
            query = f"${ticker}"
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            for submission in subreddit.search(query, limit=10):
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:20]:
                    sentiment = self.sentiment_analyzer.polarity_scores(comment.body)
                    if sentiment["compound"] > 0.1:
                        sentiments["positive"] += 1
                    elif sentiment["compound"] < -0.1:
                        sentiments["negative"] += 1
                    else:
                        sentiments["neutral"] += 1
            return sentiments
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    def google_news_sentiment(self, ticker):
        try:
            news = self.google_news.get_news(ticker)
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            for article in news[:10]:
                description = article.get("description", "")
                sentiment = self.sentiment_analyzer.polarity_scores(description)
                if sentiment["compound"] > 0.1:
                    sentiments["positive"] += 1
                elif sentiment["compound"] < -0.1:
                    sentiments["negative"] += 1
                else:
                    sentiments["neutral"] += 1
            return sentiments
        except Exception as e:
            logger.error(f"Error fetching Google News sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    def fetch_combined_sentiment(self, ticker):
        try:
            newsapi_sentiment = self.newsapi_sentiment(ticker)
            gnews_sentiment = self.gnews_sentiment(ticker)
            reddit_sentiment = self.reddit_sentiment(ticker)
            google_sentiment = self.google_news_sentiment(ticker)

            aggregated = {
                "positive": (newsapi_sentiment["positive"] + gnews_sentiment["positive"] +
                            reddit_sentiment["positive"] + google_sentiment["positive"]),
                "negative": (newsapi_sentiment["negative"] + gnews_sentiment["negative"] +
                            reddit_sentiment["negative"] + google_sentiment["negative"]),
                "neutral": (newsapi_sentiment["neutral"] + gnews_sentiment["neutral"] +
                           reddit_sentiment["neutral"] + google_sentiment["neutral"])
            }

            return {
                "newsapi": newsapi_sentiment,
                "gnews": gnews_sentiment,
                "reddit": reddit_sentiment,
                "google_news": google_sentiment,
                "aggregated": aggregated
            }
        except Exception as e:
            logger.error(f"Error fetching combined sentiment: {e}")
            return {
                "newsapi": {"positive": 0, "negative": 0, "neutral": 0},
                "gnews": {"positive": 0, "negative": 0, "neutral": 0},
                "reddit": {"positive": 0, "negative": 0, "neutral": 0},
                "google_news": {"positive": 0, "negative": 0, "neutral": 0},
                "aggregated": {"positive": 0, "negative": 0, "neutral": 0}
            }

    def _generate_detailed_recommendation(self, stock_data, recommendation, buy_score, sell_score,
                                        price_to_sma200, trend_direction, sentiment_score,
                                        volatility, sharpe_ratio):
        explanation = f"Recommendation for {stock_data['name']} ({stock_data['symbol']}): {recommendation}\n"
        explanation += f"Current Price: ₹{stock_data['current_price']['INR']:.2f}\n\n"
        explanation += f"Buy Score: {buy_score:.3f}, Sell Score: {sell_score:.3f}\n\n"

        if recommendation in ["STRONG BUY", "BUY"]:
            explanation += "Bullish Factors:\n"
            if price_to_sma200 < 1:
                explanation += "- Price is below the 200-day SMA, indicating potential undervaluation.\n"
            if trend_direction == "UPTREND":
                explanation += "- Stock is in an uptrend (50-day SMA > 200-day SMA).\n"
            if sentiment_score > 0.6:
                explanation += "- Positive market sentiment from news and social media.\n"
            if sharpe_ratio > 0:
                explanation += "- Positive risk-adjusted return (Sharpe Ratio).\n"
        elif recommendation in ["STRONG SELL", "SELL"]:
            explanation += "Bearish Factors:\n"
            if price_to_sma200 > 1:
                explanation += "- Price is above the 200-day SMA, suggesting potential overvaluation.\n"
            if trend_direction == "DOWNTREND":
                explanation += "- Stock is in a downtrend (50-day SMA < 200-day SMA).\n"
            if sentiment_score < 0.4:
                explanation += "- Negative market sentiment from news and social media.\n"
            if sharpe_ratio < 0:
                explanation += "- Negative risk-adjusted return (Sharpe Ratio).\n"
        else:
            explanation += "Neutral Factors:\n"
            explanation += "- Price is stable relative to moving averages.\n"
            explanation += "- Sentiment is balanced, with no strong bullish or bearish signals.\n"

        explanation += f"\nRisk Assessment:\n"
        explanation += f"- Volatility: {volatility:.4f} (higher values indicate higher risk)\n"
        explanation += f"- Sector: {stock_data.get('sector', 'N/A')}\n"
        explanation += f"- Industry: {stock_data.get('industry', 'N/A')}\n"

        return explanation

    def convert_df_to_dict(self, df):
        if df is None or df.empty:
            return {}
        try:
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.where(pd.notnull(df), None)
            result = df.to_dict(orient='index')
            return {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in result.items()}
        except Exception as e:
            logger.error(f"Error converting DataFrame to dict: {e}")
            return {}

    def income_statement(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            income_stmt = stock.financials
            if income_stmt.empty:
                return {"success": False, "message": f"No income statement data for {ticker}"}
            income_dict = self.convert_df_to_dict(income_stmt)
            return {
                "success": True,
                "income_statement": income_dict
            }
        except Exception as e:
            logger.error(f"Error fetching income statement: {e}")
            return {"success": False, "message": f"Error fetching income statement: {str(e)}"}

    def balance_sheet(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            balance = stock.balance_sheet
            if balance.empty:
                return {"success": False, "message": f"No balance sheet data for {ticker}"}
            balance_dict = self.convert_df_to_dict(balance)
            return {
                "success": True,
                "balance_sheet": balance_dict
            }
        except Exception as e:
            logger.error(f"Error fetching balance sheet: {e}")
            return {"success": False, "message": f"Error fetching balance sheet: {str(e)}"}

    def cash_flow(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            cashflow = stock.cashflow
            if cashflow.empty:
                return {"success": False, "message": f"No cash flow data for {ticker}"}
            cashflow_dict = self.convert_df_to_dict(cashflow)
            return {
                "success": True,
                "cash_flow": cashflow_dict
            }
        except Exception as e:
            logger.error(f"Error fetching cash flow: {e}")
            return {"success": False, "message": f"Error fetching cash flow: {str(e)}"}

    def calculate_mpt_metrics(self, stock_history, benchmark_tickers):
        try:
            stock_returns = stock_history["Close"].pct_change().dropna()
            if stock_returns.empty:
                return {
                    "annual_return": "N/A",
                    "annual_volatility": "N/A",
                    "sharpe_ratio": "N/A",
                    "beta": "N/A",
                    "alpha": "N/A"
                }

            annual_return = stock_returns.mean() * 252
            annual_volatility = stock_returns.std() * np.sqrt(252)
            risk_free_rate = 0.06  # Adjusted for Indian risk-free rate
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else "N/A"

            beta = "N/A"
            alpha = "N/A"
            for benchmark_ticker in benchmark_tickers:
                benchmark = yf.Ticker(benchmark_ticker)
                benchmark_history = benchmark.history(period="2y")
                benchmark_returns = benchmark_history["Close"].pct_change().dropna()

                if not benchmark_returns.empty:
                    aligned_returns = pd.concat([stock_returns, benchmark_returns], axis=1, join='inner')
                    if not aligned_returns.empty:
                        stock_ret = aligned_returns.iloc[:, 0]
                        bench_ret = aligned_returns.iloc[:, 1]
                        covariance = stock_ret.cov(bench_ret)
                        benchmark_variance = bench_ret.var()
                        beta = covariance / benchmark_variance if benchmark_variance != 0 else "N/A"
                        market_return = bench_ret.mean() * 252
                        alpha = annual_return - (risk_free_rate + beta * (market_return - risk_free_rate)) if beta != "N/A" else "N/A"
                        break

            return {
                "annual_return": float(annual_return) if not pd.isna(annual_return) else "N/A",
                "annual_volatility": float(annual_volatility) if not pd.isna(annual_volatility) else "N/A",
                "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio != "N/A" else "N/A",
                "beta": float(beta) if beta != "N/A" else "N/A",
                "alpha": float(alpha) if alpha != "N/A" else "N/A"
            }
        except Exception as e:
            logger.error(f"Error calculating MPT metrics: {e}")
            return {
                "annual_return": "N/A",
                "annual_volatility": "N/A",
                "sharpe_ratio": "N/A",
                "beta": "N/A",
                "alpha": "N/A"
            }

    def train_rl_with_adversarial_events(self, history, ml_predicted_price, current_price,
                                       num_episodes=100, adversarial_freq=0.2, max_event_magnitude=0.1):
        try:
            from .models import AdversarialStockTradingEnv, AdversarialQLearningAgent

            history = history.copy()
            history["SMA_50"] = history["Close"].rolling(window=50).mean()
            history["SMA_200"] = history["Close"].rolling(window=200).mean()

            def calculate_rsi(data, periods=14):
                delta = data.diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.ewm(com=periods-1, adjust=False).mean()
                roll_down = down.ewm(com=periods-1, adjust=False).mean()
                rs = roll_up / roll_down.where(roll_down != 0, 1e-10)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                return rsi.clip(0, 100)

            history["RSI"] = calculate_rsi(history["Close"])

            exp1 = history["Close"].ewm(span=12, adjust=False).mean()
            exp2 = history["Close"].ewm(span=26, adjust=False).mean()
            history["MACD"] = exp1 - exp2

            history["Daily_Return"] = history["Close"].pct_change()
            history["Volatility"] = history["Daily_Return"].rolling(window=30).std()

            history.fillna({
                "SMA_50": history["Close"],
                "SMA_200": history["Close"],
                "RSI": 50,
                "MACD": 0,
                "Volatility": 0
            }, inplace=True)

            env = AdversarialStockTradingEnv(
                history=history,
                current_price=current_price,
                ml_predicted_price=ml_predicted_price,
                adversarial_freq=adversarial_freq,
                max_event_magnitude=max_event_magnitude
            )

            agent = AdversarialQLearningAgent(state_size=11, action_size=3)

            total_rewards = []
            event_counts = []
            epoch_logs = []

            logger.info(f"Training adversarial RL agent...")
            for episode in range(num_episodes):
                state = env.reset()
                total_reward = 0
                done = False
                episode_events = 0

                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                    if state[-1] > 0:
                        episode_events += 1

                total_rewards.append(total_reward)
                event_counts.append(episode_events)
                if (episode + 1) % 10 == 0:
                    logger.info(f"Adversarial Episode {episode + 1}/{num_episodes}, "
                                f"Total Reward: {total_reward:.2f}, Events: {episode_events}")
                    epoch_logs.append({
                        "episode": episode + 1,
                        "total_reward": total_reward,
                        "average_reward": np.mean(total_rewards[-10:]),
                        "events_triggered": episode_events
                    })

            state = env.reset()
            done = False
            actions_taken = []
            net_worth_history = []

            while not done:
                action = agent.get_action(state)
                actions_taken.append(action)
                next_state, reward, done, _ = env.step(action)
                net_worth_history.append(env.net_worth)
                state = next_state

            final_net_worth = net_worth_history[-1]
            performance = (final_net_worth - env.initial_balance) / env.initial_balance * 100

            recommendation = "BUY" if performance > 10 else "SELL" if performance < -5 else "HOLD"

            return {
                "success": True,
                "recommendation": recommendation,
                "performance_pct": float(performance),
                "final_net_worth": float(final_net_worth),
                "average_reward": float(np.mean(total_rewards)),
                "average_events_per_episode": float(np.mean(event_counts)),
                "actions_distribution": {
                    "hold": actions_taken.count(0) / len(actions_taken),
                    "buy": actions_taken.count(1) / len(actions_taken),
                    "sell": actions_taken.count(2) / len(actions_taken)
                },
                "epoch_logs": epoch_logs
            }

        except Exception as e:
            logger.error(f"Error in adversarial RL training: {e}")
            return {
                "success": False,
                "message": f"Error in adversarial RL training: {str(e)}"
            }
