"""
Portfolio management module for the Indian stock trading bot.
Includes classes for data feeds, virtual portfolio, paper execution, 
performance reporting, and portfolio tracking.
"""

import json
import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dhanhq import dhanhq

logger = logging.getLogger(__name__)

class DataFeed:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_live_prices(self):
        """Fetch live prices for specified tickers using yfinance."""
        data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="1d", interval="1m")
                if not df.empty:
                    latest = df.iloc[-1]
                    data[ticker] = {
                        "price": latest["Close"],
                        "volume": latest["Volume"]
                    }
                else:
                    logger.warning(f"No data returned for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **data
        }

class VirtualPortfolio:
    def __init__(self, config):
        self.starting_balance = config["starting_balance"]
        self.cash = self.starting_balance
        self.holdings = {}  # {asset: {qty, avg_price}}
        self.trade_log = []
        self.mode = config.get("mode", "paper")

        # Initialize Dhan API only if we have credentials
        if config.get("dhan_client_id") and config.get("dhan_access_token"):
            self.api = dhanhq(
                client_id=config["dhan_client_id"],
                access_token=config["dhan_access_token"]
            )
        else:
            self.api = None
            logger.warning("Dhan API credentials not provided. Running in simulation mode.")

        self.config = config

        # Set file paths based on mode
        if self.mode == "live":
            self.portfolio_file = "data/portfolio_india_live.json"
            self.trade_log_file = "data/trade_log_india_live.json"
        else:
            self.portfolio_file = "data/portfolio_india_paper.json"
            self.trade_log_file = "data/trade_log_india_paper.json"

        self.initialize_files()

    def initialize_files(self):
        """Initialize portfolio and trade log JSON files if they don't exist."""
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, "w") as f:
                json.dump({"cash": self.cash, "holdings": self.holdings}, f, indent=4)
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, "w") as f:
                json.dump([], f, indent=4)

    def initialize_portfolio(self, balance=None):
        """Reset or initialize portfolio with a given balance."""
        if balance is not None:
            self.starting_balance = balance
        self.cash = self.starting_balance
        self.holdings = {}
        self.trade_log = []
        self.save_portfolio()
        self.save_trade_log()

    def buy(self, asset, qty, price):
        """Execute a buy order in live or paper trading mode."""
        cost = qty * price
        if cost > self.cash:
            logger.warning(f"Insufficient cash for buy order: {asset}, qty: {qty}, price: {price}")
            return False

        try:
            # Only place actual order in live mode
            if self.mode == "live" and self.api:
                order_result = self.api.place_order(
                    security_id=self.get_security_id(asset),
                    exchange_segment="NSE_EQ",
                    transaction_type="BUY",
                    order_type="MARKET",
                    quantity=qty,
                    price=0,  # Market order uses 0 for price
                    validity="DAY"
                )
                logger.info(f"Live order placed: {order_result}")
            else:
                logger.info(f"Paper trade executed: BUY {qty} {asset} at ₹{price}")

            # Update portfolio regardless of mode
            self.cash -= cost
            if asset in self.holdings:
                current_qty = self.holdings[asset]["qty"]
                current_avg_price = self.holdings[asset]["avg_price"]
                new_qty = current_qty + qty
                new_avg_price = ((current_avg_price * current_qty) + (price * qty)) / new_qty
                self.holdings[asset] = {"qty": new_qty, "avg_price": new_avg_price}
            else:
                self.holdings[asset] = {"qty": qty, "avg_price": price}

            self.log_trade({
                "asset": asset,
                "action": "buy",
                "qty": qty,
                "price": price,
                "mode": self.mode,
                "timestamp": str(datetime.now())
            })
            self.save_portfolio()
            return True

        except Exception as e:
            logger.error(f"Error executing buy order for {asset}: {e}")
            return False

    def sell(self, asset, qty, price):
        """Execute a sell order in live or paper trading mode."""
        if asset not in self.holdings or self.holdings[asset]["qty"] < qty:
            logger.warning(f"Insufficient holdings for sell order: {asset}, qty: {qty}")
            return False

        try:
            # Only place actual order in live mode
            if self.mode == "live" and self.api:
                order_result = self.api.place_order(
                    security_id=self.get_security_id(asset),
                    exchange_segment="NSE_EQ",
                    transaction_type="SELL",
                    order_type="MARKET",
                    quantity=qty,
                    price=0,  # Market order uses 0 for price
                    validity="DAY"
                )
                logger.info(f"Live order placed: {order_result}")
            else:
                logger.info(f"Paper trade executed: SELL {qty} {asset} at ₹{price}")

            # Update portfolio regardless of mode
            revenue = qty * price
            self.cash += revenue
            current_qty = self.holdings[asset]["qty"]
            if current_qty == qty:
                del self.holdings[asset]
            else:
                self.holdings[asset]["qty"] -= qty

            self.log_trade({
                "asset": asset,
                "action": "sell",
                "qty": qty,
                "price": price,
                "mode": self.mode,
                "timestamp": str(datetime.now())
            })
            self.save_portfolio()
            return True

        except Exception as e:
            logger.error(f"Error executing sell order for {asset}: {e}")
            return False

    def get_security_id(self, ticker):
        """Fetch security ID for a ticker from Dhan API or a mapping."""
        try:
            # Convert ticker to Dhan format (remove .NS, .BO suffixes)
            dhan_symbol = ticker.split('.')[0] if ticker.endswith(('.NS', '.BO')) else ticker

            # If no API connection, return mock ID for paper trading
            if not self.api:
                logger.info(f"No API connection. Using mock ID for {ticker}")
                return f"MOCK_{dhan_symbol}"

            # For live mode, we need a real security ID
            if self.mode == "live":
                try:
                    # Use fetch_security_list for live trading
                    instruments = self.api.fetch_security_list("compact")

                    # Handle different response formats
                    if isinstance(instruments, list):
                        for inst in instruments:
                            if isinstance(inst, dict) and inst.get("trading_symbol") == dhan_symbol and inst.get("exchange_segment") == "NSE":
                                security_id = inst.get("security_id")
                                logger.info(f"Found security ID {security_id} for {ticker}")
                                return security_id

                    # If we get here, we couldn't find the security ID in live mode
                    logger.error(f"Security ID not found for {ticker} in live mode. Cannot proceed with live trading.")
                    return None

                except Exception as e:
                    logger.error(f"Error fetching security list in live mode: {str(e)}")
                    return None
            else:
                # Paper trading mode - try to get real ID but fall back to mock
                try:
                    instruments = self.api.fetch_security_list("compact")
                    if isinstance(instruments, list):
                        for inst in instruments:
                            if isinstance(inst, dict) and inst.get("trading_symbol") == dhan_symbol and inst.get("exchange_segment") == "NSE":
                                return inst.get("security_id")
                except Exception as e:
                    logger.warning(f"Error fetching security list in paper mode: {str(e)}. Using mock ID.")

                # Fall back to mock ID for paper trading
                logger.info(f"Using mock ID for paper trading: {ticker}")
                return f"MOCK_{dhan_symbol}"

        except Exception as e:
            logger.error(f"Error in get_security_id for {ticker}: {str(e)}")
            # For paper trading, return a mock ID to allow trades to proceed
            if self.mode == "paper":
                return f"MOCK_{ticker.replace('.NS', '')}"
            else:
                return None

    def get_value(self, current_prices):
        """Calculate total portfolio value based on current prices."""
        total_value = self.cash
        for asset, data in self.holdings.items():
            price = current_prices.get(asset, {}).get("price", 0)
            total_value += data["qty"] * price
        return total_value

    def get_metrics(self):
        """Return portfolio metrics including PnL and exposure."""
        current_prices = self.get_current_prices()
        metrics = {
            "cash": self.cash,
            "holdings": self.holdings,
            "total_value": self.get_value(current_prices),
            "current_portfolio_value": self.config.get("current_portfolio_value", 0),
            "current_pnl": self.config.get("current_pnl", 0),
            "realized_pnl": sum(
                (t["price"] - self.holdings.get(t["asset"], {}).get("avg_price", t["price"])) * t["qty"]
                for t in self.trade_log if t["action"] == "sell"
            ),
            "unrealized_pnl": sum(
                (current_prices.get(asset, {}).get("price", 0) - data["avg_price"]) * data["qty"]
                for asset, data in self.holdings.items()
            ),
            "total_exposure": sum(
                data["qty"] * current_prices.get(asset, {}).get("price", 0)
                for asset, data in self.holdings.items()
            )
        }
        return metrics

    def log_trade(self, trade):
        """Log a trade to the trade log file."""
        self.trade_log.append(trade)
        self.save_trade_log()

    def save_portfolio(self):
        """Save portfolio state to JSON file."""
        try:
            with open(self.portfolio_file, "w") as f:
                json.dump({"cash": self.cash, "holdings": self.holdings}, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    def save_trade_log(self):
        """Save trade log to JSON file."""
        try:
            with open(self.trade_log_file, "w") as f:
                json.dump(self.trade_log, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving trade log: {e}")

    def get_current_prices(self):
        """Fetch current prices from Dhan API."""
        prices = {}
        for asset in self.holdings:
            try:
                quote = self.api.get_market_quote(security_id=self.get_security_id(asset))
                if quote and "last_price" in quote:
                    prices[asset] = {"price": quote["last_price"], "volume": quote.get("volume", 0)}
            except Exception as e:
                logger.error(f"Error fetching price for {asset}: {e}")
        return prices


class TradingExecutor:
    def __init__(self, portfolio, config):
        self.portfolio = portfolio
        self.mode = config.get("mode", "paper")
        self.dhanhq = portfolio.api  # Use the dhanhq client from VirtualPortfolio
        self.stop_loss_pct = float(config.get("stop_loss_pct", 0.05))
        self.max_capital_per_trade = float(config.get("max_capital_per_trade", 0.25))
        self.max_trade_limit = int(config.get("max_trade_limit", 10))

    def execute_trade(self, action, ticker, qty, price, stop_loss=None, take_profit=None):
        try:
            # Apply risk management rules
            portfolio_value = self.portfolio.get_value({ticker: {"price": price}})
            max_trade_value = portfolio_value * self.max_capital_per_trade
            trade_value = qty * price

            # Check if trade exceeds maximum capital per trade
            if trade_value > max_trade_value:
                adjusted_qty = int(max_trade_value / price)
                if adjusted_qty < qty:
                    logger.warning(f"Reducing trade size from {qty} to {adjusted_qty} due to capital limits")
                    qty = adjusted_qty

            # Check trade limit
            if len(self.portfolio.trade_log) >= self.max_trade_limit:
                logger.warning(f"Maximum trade limit ({self.max_trade_limit}) reached")
                return {"success": False, "message": "Trade limit exceeded"}

            # Set default stop loss if not provided
            if stop_loss is None:
                if action.upper() == "BUY":
                    stop_loss = price * (1 - self.stop_loss_pct)
                else:
                    stop_loss = price * (1 + self.stop_loss_pct)

            # Execute trade based on mode
            if self.mode == "live" and self.dhanhq:
                # Fetch security ID for live trading
                security_id = self.get_security_id(ticker)
                if security_id is None:
                    error_msg = f"Could not find security ID for {ticker}"
                    logger.error(error_msg)
                    return {"success": False, "message": error_msg}

                # Place live order
                order = self.dhanhq.place_order(
                    security_id=security_id,
                    exchange_segment="NSE_EQ",
                    transaction_type=action.upper(),
                    order_type="MARKET",
                    quantity=int(qty),
                    price=0,  # Market order
                    validity="DAY"
                )
                logger.info(f"Live trade executed: {action} {qty} units of {ticker} at ₹{price}")
            else:
                # Paper trading
                logger.info(f"Paper trade executed: {action} {qty} units of {ticker} at ₹{price}")
                order = {"order_id": f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}

            return {
                "success": True,
                "action": action,
                "ticker": ticker,
                "qty": qty,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "mode": self.mode,
                "order": order
            }
        except Exception as e:
            logger.error(f"Error executing {action} order for {ticker}: {str(e)}")
            return {"success": False, "message": str(e)}

    def convert_ticker_to_dhan_format(self, ticker):
        """Convert yfinance ticker format to Dhan API format."""
        # Remove .NS, .BO suffixes for Indian stocks
        if ticker.endswith('.NS') or ticker.endswith('.BO'):
            return ticker.split('.')[0]
        return ticker

    def get_security_id(self, symbol, exchange="NSE"):
        """Fetch security ID for a ticker from Dhan API."""
        try:
            # Convert ticker to Dhan format
            dhan_symbol = self.convert_ticker_to_dhan_format(symbol)

            # For paper trading, we can use a mock security ID if API fails
            try:
                # Use fetch_security_list instead of get_instruments
                instruments = self.dhanhq.fetch_security_list("compact")

                # Handle different response formats
                if isinstance(instruments, list):
                    for inst in instruments:
                        if isinstance(inst, dict) and inst.get("trading_symbol") == dhan_symbol and inst.get("exchange_segment") == exchange:
                            return inst.get("security_id")
                elif isinstance(instruments, str):
                    # API returned a string instead of list, use mock ID for paper trading
                    logger.warning(f"API returned string instead of list for {symbol}. Using mock ID for paper trading.")
                    return f"MOCK_{dhan_symbol}"
                else:
                    logger.warning(f"Unexpected API response type: {type(instruments)}. Using mock ID for paper trading.")
                    return f"MOCK_{dhan_symbol}"
            except Exception as e:
                logger.warning(f"Error fetching security list: {str(e)}. Using mock ID for paper trading.")
                return f"MOCK_{dhan_symbol}"

            # If we get here, we couldn't find the security ID
            logger.warning(f"Security ID not found for {symbol} (converted to {dhan_symbol}). Using mock ID for paper trading.")
            return f"MOCK_{dhan_symbol}"
        except Exception as e:
            logger.error(f"Error in get_security_id for {symbol}: {str(e)}")
            # For paper trading, return a mock ID to allow trades to proceed
            return f"MOCK_{symbol.replace('.NS', '')}"


class PerformanceReport:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_report(self):
        """Generate a daily performance report."""
        metrics = self.portfolio.get_metrics()
        total_value = metrics["total_value"]
        starting_value = self.portfolio.starting_balance
        daily_roi = ((total_value / starting_value) - 1) * 100
        cumulative_roi = daily_roi  # Simplified for daily report

        returns = [t["price"] for t in self.portfolio.trade_log if t["action"] == "sell"]
        if len(returns) > 1:
            returns = np.array(returns)
            sharpe_ratio = (np.mean(returns) - 0.02) / np.std(returns) if np.std(returns) != 0 else 0
        else:
            sharpe_ratio = 0

        values = [starting_value] + [metrics["total_value"]]
        max_drawdown = 0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "roi_today": daily_roi,
            "cumulative_roi": cumulative_roi,
            "sharpe": sharpe_ratio,
            "drawdown": max_drawdown,
            "trades_executed": len(self.portfolio.trade_log)
        }

        report_file = os.path.join(self.report_dir, f"report_{datetime.now().strftime('%Y%m%d')}.json")
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Saved report to {report_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        return report


class PortfolioTracker:
    def __init__(self, portfolio, config):
        self.portfolio = portfolio
        self.config = config

    def log_metrics(self):
        """Log portfolio metrics once."""
        try:
            metrics = self.portfolio.get_metrics()
            logger.info(f"Portfolio Metrics:")
            logger.info(f"Cash: ₹{metrics['cash']:.2f}")
            logger.info(f"Holdings: {metrics['holdings']}")
            logger.info(f"Total Value: ₹{metrics['total_value']:.2f}")
            logger.info(f"Current Portfolio Value (Dhan): ₹{self.config.get('current_portfolio_value', 0):.2f}")
            logger.info(f"Current PnL (Dhan): ₹{self.config.get('current_pnl', 0):.2f}")
            logger.info(f"Realized PnL: ₹{metrics['realized_pnl']:.2f}")
            logger.info(f"Unrealized PnL: ₹{metrics['unrealized_pnl']:.2f}")
            logger.info(f"Total Exposure: ₹{metrics['total_exposure']:.2f}")
        except Exception as e:
            logger.error(f"Error logging portfolio metrics: {e}")
