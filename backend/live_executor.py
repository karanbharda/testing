#!/usr/bin/env python3
"""
Live Trading Executor for Dhan API Integration
Handles real order execution, portfolio management, and risk controls
Integrates with SQLite database for consistent portfolio calculations
"""

import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dhan_client import DhanAPIClient

# Import database portfolio manager
try:
    from portfolio_manager import DualPortfolioManager
    from db.database import Portfolio, Holding, Trade
except ImportError:
    from .portfolio_manager import DualPortfolioManager
    from .db.database import Portfolio, Holding, Trade

logger = logging.getLogger(__name__)


class LiveTradingExecutor:
    """Execute live trades through Dhan API with database integration and risk management"""

    def __init__(self, portfolio_manager: DualPortfolioManager, config: Dict):
        self.portfolio_manager = portfolio_manager
        # Ensure config is a dictionary
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            logger.warning(
                f"Invalid config type: {type(config)}, converting to empty dict")
            config = {}

        self.config = config
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.05)
        self.max_capital_per_trade = self.config.get(
            "max_capital_per_trade", 0.25)
        self.max_trade_limit = self.config.get("max_trade_limit", 150)

        # Initialize all attributes before any method calls
        # Trading state
        self.pending_orders = {}
        self.executed_orders = {}
        self.daily_trade_count = 0
        self.last_trade_date = None

        # Risk management
        self.max_daily_loss = config.get(
            "max_daily_loss", 0.05)  # 5% of portfolio
        self.daily_pnl = 0.0

        # Rate limiting for sync operations
        self.last_sync_time = 0
        self.min_sync_interval = 60  # Minimum 60 seconds between syncs

        # Global enable flags (env or config)
        self.enable_sell = str(self.config.get("enable_sell", os.getenv(
            "ENABLE_SELL", "true"))).lower() not in ("false", "0", "no", "off")
        self.enable_buy = str(self.config.get("enable_buy", os.getenv(
            "ENABLE_BUY", "true"))).lower() not in ("false", "0", "no", "off")

        # Ensure we're in live mode
        if self.portfolio_manager.current_mode != "live":
            self.portfolio_manager.switch_mode("live")

        # Initialize Dhan client - try config first, then environment variables
        dhan_client_id = config.get(
            "dhan_client_id") or os.getenv("DHAN_CLIENT_ID")
        dhan_access_token = config.get(
            "dhan_access_token") or os.getenv("DHAN_ACCESS_TOKEN")

        if not dhan_client_id or not dhan_access_token:
            logger.error(
                "Dhan credentials not found in config or environment variables")
            raise ValueError("Dhan credentials required for live trading")

        self.dhan_client = DhanAPIClient(
            client_id=dhan_client_id,
            access_token=dhan_access_token,
            config=self.config
        )

        # Sync portfolio with Dhan account on initialization
        try:
            if not self.sync_portfolio_with_dhan():
                logger.warning(
                    "Failed to sync portfolio with Dhan account during initialization, using local database")
            else:
                logger.info("Successfully synced portfolio with Dhan account")
        except Exception as e:
            logger.warning(
                f"Failed to sync portfolio with Dhan: {e}, continuing with local database")

        if not self.enable_sell:
            logger.warning(
                "Sell operations are DISABLED by configuration (ENABLE_SELL=false)")

        if not self.enable_buy:
            logger.warning(
                "Buy operations are DISABLED by configuration (ENABLE_BUY=false)")

        self.enable_shortsell = str(self.config.get("enable_shortsell", os.getenv(
            "ENABLE_SHORTSELL", "false"))).lower() not in ("false", "0", "no", "off")
        if not self.enable_shortsell:
            logger.info("Short-selling is disabled by configuration")

        logger.info("Live Trading Executor initialized")

    @staticmethod
    def _normalize_product_type(product_type: str) -> str:
        """Normalize product type values to Dhan-compatible format"""
        if not product_type:
            return "CNC"
        normalized = product_type.upper()
        if normalized == "INTRADAY":
            return "MIS"
        if normalized == "MARGIN":
            return "NRML"
        if normalized == "NRML":
            return "NRML"
        return "MIS" if normalized == "MIS" else "CNC"

    def get_current_product_type(self) -> str:
        """Return the active product type from the Dhan client or local config."""
        if hasattr(self, 'dhan_client') and self.dhan_client:
            product_type = getattr(self.dhan_client, 'product_type', None)
            if product_type:
                return self._normalize_product_type(product_type)
        return self._normalize_product_type(self.config.get("productType", "CNC"))

    def _has_short_position(self, symbol: str) -> bool:
        """Return True if the portfolio has an open short position for the symbol"""
        holding = self.portfolio_manager.current_holdings_dict.get(symbol)
        return holding is not None and holding.get("qty", 0) < 0

    def _should_cover_short(self, signal_data: Dict) -> bool:
        """Determine whether a BUY signal should cover an existing short position"""
        if not signal_data:
            return False
        direction = str(signal_data.get("direction", "")).upper()
        action = str(signal_data.get("action", "")).upper()
        cover_hint = signal_data.get("cover_short", False)

        return (
            cover_hint
            or direction in {"BUY", "BUY_TO_COVER", "BUY TO COVER", "COVER"}
            or action in {"BUY", "BUY_TO_COVER", "BUY TO COVER", "COVER"}
        )

    def _should_short_sell(self, symbol: str, signal_data: Dict) -> bool:
        """Determine whether a SELL signal should open an intraday short position"""
        if not self.enable_shortsell:
            return False
        product_type = self.config.get("productType", "CNC")
        if self._normalize_product_type(product_type) != "MIS":
            return False

        if not signal_data:
            return False

        direction = str(signal_data.get("direction", "")).upper()
        action = str(signal_data.get("action", "")).upper()
        short_hint = signal_data.get(
            "short_sell", False) or signal_data.get("should_short", False)

        # Allow short-selling when the signal explicitly requests it or when a bearish SELL/SHORT directional signal occurs
        return (
            short_hint
            or direction in {"SELL", "SHORT"}
            or action in {"SELL", "SHORT"}
        )

    def validate_connection(self) -> bool:
        """Validate Dhan API connection"""
        try:
            return self.dhan_client.validate_connection()
        except Exception as e:
            logger.error(f"Failed to validate Dhan connection: {e}")
            return False

    def sync_portfolio_with_dhan(self) -> bool:
        """Sync local portfolio with actual Dhan account using database"""
        try:
            # Rate limiting - avoid excessive sync calls
            current_time = time.time()
            if current_time - self.last_sync_time < self.min_sync_interval:
                logger.debug(
                    f"Sync rate limited - last sync {current_time - self.last_sync_time:.1f}s ago")
                return True  # Return success to avoid errors, but skip actual sync

            logger.debug("Syncing portfolio with Dhan account...")

            # Get account funds
            funds = self.dhan_client.get_funds()
            # Normalize across possible Dhan fund keys

            def _get_cash(d: Dict[str, Any]) -> float:
                for key in ("availablecash", "availabelBalance", "availableBalance", "netAvailableMargin", "netAvailableCash"):
                    if key in d and d.get(key) is not None:
                        try:
                            return float(d.get(key))
                        except Exception:
                            continue
                # Fallback: if there is a numeric field resembling available funds
                for k, v in d.items():
                    if isinstance(v, (int, float)) and "avail" in k.lower() and "cash" in k.lower():
                        return float(v)
                return 0.0

            available_cash = _get_cash(funds)
            logger.debug(f"Dhan Account Balance: Rs.{available_cash:.2f}")

            # Get current holdings from Dhan
            holdings = self.dhan_client.get_holdings()
            logger.debug(f"Dhan Holdings: {len(holdings)} positions")

            # Update database portfolio
            session = None
            try:
                session = self.portfolio_manager.db.Session()
                portfolio = session.query(
                    Portfolio).filter_by(mode='live').first()
                if not portfolio:
                    # Create live portfolio if it doesn't exist
                    portfolio = Portfolio(
                        mode='live',
                        cash=available_cash,
                        starting_balance=available_cash,
                        realized_pnl=0.0,
                        unrealized_pnl=0.0,
                        last_updated=datetime.now()
                    )
                    session.add(portfolio)
                    session.flush()
                else:
                    # Update existing portfolio cash
                    portfolio.cash = available_cash
                    portfolio.last_updated = datetime.now()

                # Clear existing holdings and add current ones from Dhan
                session.query(Holding).filter_by(
                    portfolio_id=portfolio.id).delete()

                total_holdings_value = 0
                for holding in holdings:
                    symbol = holding.get("tradingSymbol", "")
                    quantity = int(holding.get("totalQty", 0))
                    avg_price = float(holding.get("avgCostPrice", 0))
                    # Use lastTradedPrice if available (Dhan API field name)
                    current_price = float(holding.get(
                        "lastTradedPrice", holding.get("ltp", avg_price)))

                    if quantity > 0 and symbol:
                        db_holding = Holding(
                            portfolio_id=portfolio.id,
                            ticker=symbol,
                            quantity=quantity,
                            avg_price=avg_price,
                            last_price=current_price
                        )
                        session.add(db_holding)
                        total_holdings_value += quantity * current_price

                # Update portfolio metrics
                portfolio.unrealized_pnl = total_holdings_value - sum(
                    h.quantity * h.avg_price for h in
                    session.query(Holding).filter_by(
                        portfolio_id=portfolio.id).all()
                )

                session.commit()

                # Update current portfolio reference
                self.portfolio_manager.current_portfolio = portfolio

                # Refresh the in-memory holdings dict from database
                self.portfolio_manager.refresh_holdings_from_database()

                # NEW: Update data service watchlist with current holdings
                self._update_data_service_watchlist(session, portfolio)

                logger.debug(
                    f"Portfolio synced - Cash: Rs.{available_cash:.2f}, Holdings: Rs.{total_holdings_value:.2f}")
                self.last_sync_time = time.time()
                return True

            except Exception as e:
                if session:
                    session.rollback()
                raise e
            finally:
                if session:
                    session.close()

        except Exception as e:
            logger.warning(
                f"Failed to sync portfolio with Dhan: {e}, using local database values")
            return False

    def _update_data_service_watchlist(self, session, portfolio):
        """Update data service watchlist with current portfolio holdings"""
        try:
            # Get current holdings from database
            holdings = session.query(Holding).filter_by(
                portfolio_id=portfolio.id).all()
            holding_symbols = [holding.ticker for holding in holdings]

            # Import data service client
            from data_service_client import get_data_client
            data_client = get_data_client()

            # Update watchlist with current holdings
            if holding_symbols:
                data_client.update_watchlist(holding_symbols)
                logger.info(
                    f"Updated data service watchlist with {len(holding_symbols)} symbols")
            else:
                logger.info("No holdings to update in data service watchlist")

        except Exception as e:
            logger.error(f"Failed to update data service watchlist: {e}")

    def get_real_time_price(self, symbol: str) -> float:
        """Get real-time price using Fyers API for reliable data"""
        try:
            # Use Fyers for real-time price data (more reliable than Dhan quotes)
            try:
                # Import Fyers utilities
                from testindia import get_stock_data_fyers_or_yf

                # Get recent price data from Fyers
                hist = get_stock_data_fyers_or_yf(symbol, period="1d")

                if hist is not None and not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    if current_price > 0:
                        logger.debug(
                            f"Got price from Fyers for {symbol}: Rs.{current_price:.2f}")
                        return current_price
                    else:
                        logger.warning(
                            f"Invalid price from Fyers for {symbol}: {current_price}")
                else:
                    logger.warning(f"No price data from Fyers for {symbol}")

            except Exception as fyers_error:
                logger.warning(
                    f"Fyers price fetch failed for {symbol}: {fyers_error}")

            # Secondary fallback: Try Dhan API if Fyers fails
            try:
                quote = self.dhan_client.get_quote(symbol)
                price = float(quote.get("ltp", 0))

                if price > 0:
                    logger.debug(
                        f"Got fallback price from Dhan API for {symbol}: Rs.{price:.2f}")
                    return price

            except Exception as dhan_error:
                logger.warning(
                    f"Dhan API fallback failed for {symbol}: {dhan_error}")

            logger.error(f"All price sources failed for {symbol}")
            return 0.0

        except Exception as e:
            logger.error(f"Failed to get real-time price for {symbol}: {e}")
            return 0.0

    def calculate_position_size(self, symbol: str, current_price: float,
                                signal_strength: float) -> int:
        """
        Calculate position size based on user configuration from live_config.json
        Respects max_capital_per_trade setting (user input)
        Includes INTRADAY LEVERAGE calculation (5x for MIS)
        """
        session = None
        try:
            # Get portfolio summary from database
            session = self.portfolio_manager.db.Session()
            portfolio = session.query(Portfolio).filter_by(mode='live').first()
            if not portfolio:
                logger.warning(f"No live portfolio found for position sizing")
                return 0

            available_cash = portfolio.cash
            if available_cash <= 0:
                logger.warning(
                    f"No available cash for {symbol}: Rs.{available_cash:.2f}")
                return 0

            # Load user configuration from live_config.json
            max_capital_per_trade = 0.25  # Default fallback
            try:
                import json
                import os
                config_path = os.path.join(os.path.dirname(
                    __file__), '..', 'data', 'live_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        live_config = json.load(f)
                    max_capital_per_trade = float(
                        live_config.get("max_capital_per_trade", 0.25))
                    logger.info(
                        f"Loaded max_capital_per_trade from config: {max_capital_per_trade:.1%}")
            except Exception as config_error:
                logger.warning(
                    f"Failed to load config: {config_error}, using default 25%")

            # CRITICAL: Get product type to determine leverage
            # Dhan provides 5x leverage for MIS (intraday) positions
            product_type = self.get_current_product_type()
            leverage = 5.0 if product_type == 'MIS' else 1.0

            logger.info(f"Product Type: {product_type}, Leverage: {leverage}x")
            logger.info(f"Available Cash: Rs.{available_cash:.2f}")

            # Apply leverage for intraday positions
            effective_buying_power = available_cash * leverage
            logger.info(
                f"Effective Buying Power with {leverage}x leverage: Rs.{effective_buying_power:.2f}")

            # Calculate position size based on user's configured allocation
            # Use leveraged buying power for intraday
            position_value = effective_buying_power * max_capital_per_trade
            quantity = int(position_value / current_price)

            # Ensure at least 1 share if we have enough cash for it
            if quantity < 1 and available_cash >= current_price:
                quantity = 1

            logger.info(f"Position sizing for {symbol}:")
            logger.info(f"  Available Cash: Rs.{available_cash:.2f}")
            logger.info(f"  Product Type: {product_type}")
            logger.info(f"  Leverage: {leverage}x")
            logger.info(
                f"  Effective Buying Power: Rs.{effective_buying_power:.2f}")
            logger.info(
                f"  Max Capital Per Trade (user config): {max_capital_per_trade:.1%}")
            logger.info(f"  Allocated Position Value: Rs.{position_value:.2f}")
            logger.info(f"  Current Price: Rs.{current_price:.2f}")
            logger.info(
                f"  Calculated Quantity: {quantity} shares (Value: Rs.{quantity * current_price:.2f})")

            return quantity

        except Exception as e:
            logger.error(
                f"Failed to calculate position size for {symbol}: {e}")
            return 0
        finally:
            if session:
                session.close()

    def _adjust_quantity_based_on_funds(self, symbol: str, price: float, requested_quantity: int) -> int:
        """Adjust quantity based on available funds, using database as fallback when Dhan API fails"""
        try:
            # First, try to get funds from Dhan API
            try:
                funds = self.dhan_client.get_funds()
                logger.debug(f"Raw Dhan funds response: {funds}")

                # Try multiple possible keys that Dhan API might return for available cash
                possible_keys = [
                    'availableBalance', 'availablebalance', 'available_balance',
                    'sodLimit', 'sodlimit', 'sod_limit',
                    'netBalance', 'netbalance', 'net_balance',
                    'availablecash', 'available_cash', 'availableCash',
                    'netAvailableMargin', 'netAvailableCash',
                    'usableCash', 'usable_cash', 'UsableCash',
                    'cash', 'Cash'
                ]

                # Helper to parse numeric values robustly
                def _parse_numeric(v):
                    try:
                        if v is None:
                            return None
                        # If already numeric
                        if isinstance(v, (int, float)):
                            return float(v)
                        # If string, strip common formatting
                        s = str(v).strip()
                        # Remove currency symbols and commas
                        s = s.replace(',', '').replace(
                            'Rs.', '').replace('₹', '').replace('INR', '')
                        # Remove parentheses and plus signs
                        s = s.replace('(', '').replace(')',
                                                       '').replace('+', '')
                        # If empty after cleanup
                        if s == '':
                            return None
                        return float(s)
                    except Exception:
                        return None

                available_cash = 0.0
                # Try known keys first
                for key in possible_keys:
                    if key in funds and funds[key] is not None:
                        parsed = _parse_numeric(funds[key])
                        if parsed is not None:
                            available_cash = parsed
                            logger.info(
                                f"✅ Found available cash in Dhan response key '{key}': Rs.{available_cash:.2f}")
                            break

                # If still zero, search any nested or similarly named keys
                if available_cash == 0.0:
                    # Flatten nested dicts if present
                    def _flatten(d):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                for kk, vv in _flatten(v):
                                    yield kk, vv
                            else:
                                yield k, v

                    for key, value in _flatten(funds):
                        if value is None:
                            continue
                        parsed = _parse_numeric(value)
                        if parsed is not None:
                            available_cash = parsed
                            logger.info(
                                f"✅ Found available cash in nested key '{key}': Rs.{available_cash:.2f}")
                            break

                if available_cash > 0:
                    logger.info(
                        f"🏦 Dhan API funds available: Rs.{available_cash:.2f}")
                else:
                    logger.warning(
                        f"⚠️  Dhan API returned no parseable fund value. Raw response keys: {list(funds.keys())}")
            except Exception as api_error:
                logger.warning(
                    f"Dhan API funds fetch failed: {api_error}. Falling back to database.")
                # Fall back to database funds
                available_cash = 0.0
                session = self.portfolio_manager.db.Session()
                try:
                    portfolio = session.query(
                        Portfolio).filter_by(mode='live').first()
                    if portfolio:
                        available_cash = float(portfolio.cash or 0.0)
                        logger.debug(
                            f"Database funds available: Rs.{available_cash:.2f}")
                except Exception as db_error:
                    logger.error(
                        f"Failed to get funds from database: {db_error}")
                finally:
                    if session:
                        session.close()

            # If available cash looks zero, try to force a sync with Dhan once (may refresh values)
            if available_cash == 0.0:
                try:
                    logger.warning(
                        "⚠️  Available cash appears zero, attempting forced sync with Dhan...")
                    if self.sync_portfolio_with_dhan():
                        # Read from database after successful sync
                        session = self.portfolio_manager.db.Session()
                        try:
                            portfolio = session.query(
                                Portfolio).filter_by(mode='live').first()
                            if portfolio:
                                available_cash = float(portfolio.cash or 0.0)
                                logger.info(
                                    f"✅ Post-sync database funds: Rs.{available_cash:.2f}")
                        finally:
                            if session:
                                session.close()
                    else:
                        logger.warning(
                            "Sync returned False - broker account may have no funds or connection issue")
                except Exception as sync_err:
                    logger.error(f"Forced sync failed: {sync_err}")

            # Calculate required amount
            required_amount = price * requested_quantity

            # CRITICAL: Apply intraday leverage if product type is MIS
            product_type = self.get_current_product_type()
            leverage = 5.0 if product_type == 'MIS' else 1.0

            # For intraday (MIS), we can use 5x leverage
            effective_available_cash = available_cash * leverage

            logger.info(f"Funds check for {symbol}:")
            logger.info(f"  Product Type: {product_type}")
            logger.info(f"  Leverage: {leverage}x")
            logger.info(f"  Base Available Cash: Rs.{available_cash:.2f}")
            logger.info(
                f"  Effective Available Cash with leverage: Rs.{effective_available_cash:.2f}")
            logger.info(
                f"  Required: Rs.{required_amount:.2f} for {requested_quantity} shares")

            # If we have enough funds (with leverage), return the requested quantity
            if effective_available_cash >= required_amount:
                if leverage > 1.0:
                    logger.info(
                        f"✅ Sufficient funds with {leverage}x leverage - returning {requested_quantity} shares")
                else:
                    logger.info(
                        f"✅ Sufficient funds available - returning {requested_quantity} shares")
                return requested_quantity

            # Otherwise, calculate maximum affordable quantity (with leverage)
            max_affordable_quantity = int(effective_available_cash // price)

            # Ensure we return at least 1 share if funds allow for it
            if max_affordable_quantity >= 1:
                if leverage > 1.0:
                    logger.warning(
                        f"⚠️  Insufficient funds - adjusted quantity from {requested_quantity} to {max_affordable_quantity} "
                        f"(Base Cash: Rs.{available_cash:.2f}, Effective with {leverage}x leverage: Rs.{effective_available_cash:.2f}, "
                        f"Required: Rs.{required_amount:.2f})")
                else:
                    logger.warning(
                        f"⚠️  Insufficient funds - adjusted quantity from {requested_quantity} to {max_affordable_quantity} "
                        f"(Available: Rs.{available_cash:.2f}, Required: Rs.{required_amount:.2f})")
                return max_affordable_quantity
            else:
                logger.error(
                    f"❌ Insufficient funds to buy even 1 share of {symbol} (Price: Rs.{price:.2f}, Available: Rs.{available_cash:.2f})")
                return 0

        except Exception as e:
            logger.error(
                f"Error adjusting quantity based on funds for {symbol}: {e}")
            # In case of error, return the requested quantity (let the API handle the failure)
            return requested_quantity

    def execute_buy_order(self, symbol: str, signal_data: Dict) -> Dict:
        """Execute a buy order through Dhan API with database recording"""
        try:
            # Enforce global buy disable flag
            if not self.enable_buy:
                logger.info(
                    "Buy disabled by configuration (ENABLE_BUY=false). Skipping buy for %s", symbol)
                return {"success": False, "message": "Buy disabled by configuration"}

            # Check daily trade limit
            if self._check_daily_limits():
                return {"success": False, "message": "Daily trade limit exceeded"}

            # Check market status
            if not self.dhan_client.is_market_open():
                return {"success": False, "message": "Market is closed"}

            # Get current price using Fyers (reliable) with fallback
            current_price = self.get_real_time_price(symbol)
            if current_price <= 0:
                # Use the price from trading signal as final fallback
                current_price = signal_data.get("current_price", 0)
                if current_price <= 0:
                    return {"success": False, "message": "Unable to get current price from any source"}
                logger.info(
                    f"Using trading signal price for {symbol}: Rs.{current_price:.2f}")

            # Use provided quantity if available, otherwise calculate
            requested_quantity = signal_data.get("quantity", 0)
            if requested_quantity <= 0:
                # Calculate position size if no quantity provided
                signal_strength = signal_data.get("confidence", 0.5)
                requested_quantity = self.calculate_position_size(
                    symbol, current_price, signal_strength)

            if requested_quantity <= 0:
                return {"success": False, "message": "Insufficient funds or invalid quantity"}

            # Check available funds and adjust quantity if needed
            adjusted_quantity = self._adjust_quantity_based_on_funds(
                symbol, current_price, requested_quantity)

            if adjusted_quantity <= 0:
                return {"success": False, "message": "Insufficient funds for purchase"}

            # If we have an open short position, treat this buy as a cover request first
            if self._has_short_position(symbol) and self._should_cover_short(signal_data):
                logger.info(
                    f"Detected open short position for {symbol}; executing buy-to-cover instead of new long entry")
                return self.execute_buy_to_cover_order(symbol, adjusted_quantity, current_price, signal_data)

            # Place market buy order with product type
            # Get product type from signal_data or current execution configuration
            product_type = self._normalize_product_type(
                signal_data.get('product_type') or self.get_current_product_type())

            logger.info(f"Placing BUY order with product type: {product_type}")

            order_response = self.dhan_client.place_order(
                symbol=symbol,
                quantity=adjusted_quantity,
                order_type="MARKET",
                side="BUY",
                product_type=product_type  # Pass product type dynamically
            )

            order_id = order_response.get("orderId")
            if order_id:
                # Track pending order
                self.pending_orders[order_id] = {
                    "symbol": symbol,
                    "quantity": adjusted_quantity,
                    "side": "BUY",
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                    "signal_data": signal_data
                }

                # Check order status immediately after placing to see if it's already executed
                try:
                    self.check_and_update_orders()
                except Exception as status_error:
                    logger.warning(
                        f"Failed to check order status immediately after placement: {status_error}")

                # NOTE: We shouldn't record the trade immediately after placing the order.
                # The order may still fail at the broker level. The actual recording should happen
                # in _update_portfolio_after_execution when we confirm the order was executed.
                # However, since the system currently relies on immediate recording, we'll keep this
                # but add better handling for cases where the order status changes later.
                try:
                    # Check if the order was immediately executed by checking the status
                    order_details = self.dhan_client.get_order_by_id(order_id)
                    order_status = order_details.get(
                        "orderStatus", "").upper() if order_details else "UNKNOWN"

                    # Dhan API order statuses: PENDING, OPEN, TRADED, CANCELLED, REJECTED, MODIFYING, MODIFIED
                    executed_statuses = ["TRADED", "FILLED", "COMPLETE"]
                    pending_statuses = ["PENDING", "OPEN", "TRIGGER PENDING"]
                    failed_statuses = ["CANCELLED",
                                       "REJECTED", "REJECTED BY EXCHANGE"]

                    if order_status in executed_statuses:
                        # Order was immediately executed, record it in database
                        # Get actual execution price from order details if available
                        executed_price = float(
                            order_details.get("price", current_price))
                        executed_qty = int(order_details.get(
                            "tradedQuantity", adjusted_quantity))

                        logger.info(
                            f"✅ Order EXECUTED: {order_status} | Price: {executed_price} | Qty: {executed_qty}")

                        self.portfolio_manager.record_trade(
                            ticker=symbol,
                            action="buy",
                            quantity=executed_qty,
                            price=executed_price,
                            stop_loss=signal_data.get("stop_loss"),
                            take_profit=signal_data.get("take_profit"),
                            product_type=product_type
                        )
                        logger.info(
                            f"✅ Buy trade executed and recorded: {executed_qty} {symbol} at Rs.{executed_price:.2f}")

                        # Also update the portfolio after execution
                        self._update_portfolio_after_execution(
                            symbol, "BUY", executed_qty, executed_price
                        )
                    elif order_status in pending_statuses:
                        # Order is pending, don't record in database yet - wait for execution
                        logger.info(
                            f"⏳ Buy order placed but PENDING execution: {adjusted_quantity} {symbol} at Rs.{current_price:.2f}")
                        logger.info(f"   Order ID: {order_id}")
                        logger.info(f"   Current status: {order_status}")
                    elif order_status in failed_statuses:
                        # Order failed immediately
                        logger.error(
                            f"❌ Buy order FAILED: {order_status} | Order ID: {order_id}")
                        logger.error(
                            f"   Symbol: {symbol} | Quantity: {adjusted_quantity}")
                        # Don't record in database - order failed
                    else:
                        # Unexpected status - log warning and don't record
                        logger.warning(
                            f"⚠️ Unexpected order status: {order_status}")
                        logger.info(f"   Order ID: {order_id}")
                        logger.info(f"   Full order details: {order_details}")

                except Exception as db_error:
                    logger.error(
                        f"❌ Error checking immediate execution or recording trade: {db_error}")
                    logger.error(f"   Order ID: {order_id}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")

                    # Don't auto-record on error - wait for proper confirmation
                    # This prevents recording failed orders
                    logger.warning(
                        f"⚠️ Skipping trade recording due to error - will sync on next check")

                # Update trade count
                self._update_daily_trade_count()

                logger.info(
                    f"Buy order placed: {adjusted_quantity} {symbol} at Rs.{current_price:.2f}")
                logger.info(f"   Order ID: {order_id}")

                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": adjusted_quantity,
                    "price": current_price,
                    "message": f"Buy order placed for {adjusted_quantity} shares of {symbol}"
                }
            else:
                logger.error(
                    f"❌ Unexpected order response for buy order: {order_response}")
                return {"success": False, "message": "Order placement failed - unexpected response"}

        except ValueError as ve:
            # Handle security ID validation errors specifically
            logger.error(f"❌ Security ID validation failed for {symbol}: {ve}")
            return {"success": False, "message": f"Security ID validation failed for {symbol}"}
        except Exception as e:
            logger.error(f"❌ Failed to execute buy order for {symbol}: {e}")
            return {"success": False, "message": f"Buy order failed: {str(e)}"}

    def execute_short_sell_order(self, symbol: str, signal_data: Dict) -> Dict:
        """Execute an intraday short sell position"""
        try:
            if not self.enable_sell or not self.enable_shortsell:
                logger.info(
                    "Short sell disabled by configuration. Skipping short sell for %s", symbol)
                return {"success": False, "message": "Short sell disabled by configuration"}

            if self._check_daily_limits():
                return {"success": False, "message": "Daily trade limit exceeded"}

            if not self.dhan_client.is_market_open():
                return {"success": False, "message": "Market is closed"}

            current_price = self.get_real_time_price(symbol)
            if current_price <= 0:
                current_price = signal_data.get("current_price", 0)
                if current_price <= 0:
                    return {"success": False, "message": "Unable to get current price from any source"}
                logger.info(
                    f"Using trading signal price for short sell {symbol}: Rs.{current_price:.2f}")

            requested_quantity = signal_data.get("quantity", 0)
            if requested_quantity <= 0:
                signal_strength = signal_data.get("confidence", 0.5)
                requested_quantity = self.calculate_position_size(
                    symbol, current_price, signal_strength)

            if requested_quantity <= 0:
                return {"success": False, "message": "Insufficient funds or invalid quantity"}

            adjusted_quantity = self._adjust_quantity_based_on_funds(
                symbol, current_price, requested_quantity)
            if adjusted_quantity <= 0:
                return {"success": False, "message": "Insufficient funds for short sell"}

            product_type = self._normalize_product_type(
                signal_data.get('product_type') or self.get_current_product_type())
            logger.info(
                f"Placing SHORT SELL order with product type: {product_type}")

            order_response = self.dhan_client.place_order(
                symbol=symbol,
                quantity=adjusted_quantity,
                order_type="MARKET",
                side="SELL",
                product_type=product_type
            )

            order_id = order_response.get("orderId")
            if order_id:
                self.pending_orders[order_id] = {
                    "symbol": symbol,
                    "quantity": adjusted_quantity,
                    "side": "SELL",
                    "is_short_sell": True,
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                    "signal_data": signal_data
                }

                try:
                    self.check_and_update_orders()
                except Exception as status_error:
                    logger.warning(
                        f"Failed to check order status immediately after placement: {status_error}")

                try:
                    order_details = self.dhan_client.get_order_by_id(order_id)
                    order_status = order_details.get(
                        "orderStatus", "").upper() if order_details else "UNKNOWN"

                    executed_statuses = ["TRADED", "FILLED", "COMPLETE"]
                    pending_statuses = ["PENDING", "OPEN", "TRIGGER PENDING"]
                    failed_statuses = ["CANCELLED",
                                       "REJECTED", "REJECTED BY EXCHANGE"]

                    if order_status in executed_statuses:
                        executed_price = float(
                            order_details.get("price", current_price))
                        executed_qty = int(order_details.get(
                            "tradedQuantity", adjusted_quantity))

                        logger.info(
                            f"✅ Short sell EXECUTED: {order_status} | Price: {executed_price} | Qty: {executed_qty}")

                        self.portfolio_manager.record_trade(
                            ticker=symbol,
                            action="sell",
                            quantity=executed_qty,
                            price=executed_price,
                            stop_loss=signal_data.get("stop_loss"),
                            take_profit=signal_data.get("take_profit"),
                            product_type=product_type,
                            is_short_sell=True
                        )
                        logger.info(
                            f"✅ Short sell executed and recorded: {executed_qty} {symbol} at Rs.{executed_price:.2f}")
                        self._update_portfolio_after_execution(
                            symbol, "SELL", executed_qty, executed_price,
                            is_short_sell=True
                        )
                    elif order_status in pending_statuses:
                        logger.info(
                            f"⏳ Short sell order PENDING: {adjusted_quantity} {symbol} at Rs.{current_price:.2f}")
                        logger.info(
                            f"   Order ID: {order_id} | Status: {order_status}")
                    elif order_status in failed_statuses:
                        logger.error(
                            f"❌ Short sell order FAILED: {order_status} | Order ID: {order_id}")
                    else:
                        logger.warning(
                            f"⚠️ Unexpected short sell order status: {order_status}")
                        logger.info(f"   Order details: {order_details}")

                except Exception as db_error:
                    logger.error(
                        f"❌ Error checking short sell execution: {db_error}")
                    logger.warning(
                        f"⚠️ Skipping trade recording - will sync on next check")

                self._update_daily_trade_count()
                logger.info(
                    f"Short sell order placed: {adjusted_quantity} {symbol} at Rs.{current_price:.2f}")
                logger.info(f"   Order ID: {order_id}")

                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": adjusted_quantity,
                    "price": current_price,
                    "message": f"Short sell order placed for {adjusted_quantity} shares of {symbol}"
                }
            else:
                logger.error(
                    f"❌ Unexpected order response for short sell order: {order_response}")
                return {"success": False, "message": "Short sell order placement failed - unexpected response"}

        except Exception as e:
            logger.error(
                f"❌ Failed to execute short sell order for {symbol}: {e}")
            return {"success": False, "message": f"Short sell order failed: {str(e)}"}

    def execute_buy_to_cover_order(self, symbol: str, quantity: int, price: float, signal_data: Dict = None) -> Dict:
        """Buy to cover an open short intraday position"""
        try:
            if not self.enable_buy:
                logger.info(
                    "Buy disabled by configuration (ENABLE_BUY=false). Skipping cover buy for %s", symbol)
                return {"success": False, "message": "Buy disabled by configuration"}

            if self._check_daily_limits():
                return {"success": False, "message": "Daily trade limit exceeded"}

            if not self.dhan_client.is_market_open():
                return {"success": False, "message": "Market is closed"}

            if quantity <= 0:
                return {"success": False, "message": "Invalid quantity for cover buy"}

            if not self._has_short_position(symbol):
                return {"success": False, "message": "No short position to cover"}

            product_type = self._normalize_product_type(
                signal_data.get('product_type') or self.get_current_product_type()) if signal_data else self.get_current_product_type()
            logger.info(
                f"Placing BUY TO COVER for {quantity} {symbol} with product type: {product_type}")

            order_response = self.dhan_client.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type="MARKET",
                side="BUY",
                product_type=product_type
            )

            order_id = order_response.get("orderId")
            if order_id:
                self.pending_orders[order_id] = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": "BUY",
                    "is_cover_short": True,
                    "price": price,
                    "timestamp": datetime.now().isoformat(),
                    "signal_data": signal_data
                }

                try:
                    self.check_and_update_orders()
                except Exception as status_error:
                    logger.warning(
                        f"Failed to check order status immediately after placement: {status_error}")

                try:
                    order_details = self.dhan_client.get_order_by_id(order_id)
                    order_status = order_details.get(
                        "orderStatus", "").upper() if order_details else "UNKNOWN"

                    executed_statuses = ["TRADED", "FILLED", "COMPLETE"]
                    pending_statuses = ["PENDING", "OPEN", "TRIGGER PENDING"]
                    failed_statuses = ["CANCELLED",
                                       "REJECTED", "REJECTED BY EXCHANGE"]

                    if order_status in executed_statuses:
                        executed_price = float(
                            order_details.get("price", price))
                        executed_qty = int(order_details.get(
                            "tradedQuantity", quantity))

                        logger.info(
                            f"✅ Buy-to-cover EXECUTED: {order_status} | Price: {executed_price} | Qty: {executed_qty}")

                        self.portfolio_manager.record_trade(
                            ticker=symbol,
                            action="buy",
                            quantity=executed_qty,
                            price=executed_price,
                            pnl=0.0,
                            stop_loss=signal_data.get("stop_loss"),
                            take_profit=signal_data.get("take_profit"),
                            product_type=product_type,
                            is_cover_short=True
                        )
                        logger.info(
                            f"✅ Cover buy executed and recorded: {executed_qty} {symbol} at Rs.{executed_price:.2f}")
                        self._update_portfolio_after_execution(
                            symbol, "BUY", executed_qty, executed_price,
                            is_cover_short=True
                        )
                    elif order_status in pending_statuses:
                        logger.info(
                            f"⏳ Buy-to-cover order PENDING: {quantity} {symbol} at Rs.{price:.2f}")
                        logger.info(
                            f"   Order ID: {order_id} | Status: {order_status}")
                    elif order_status in failed_statuses:
                        logger.error(
                            f"❌ Buy-to-cover order FAILED: {order_status} | Order ID: {order_id}")
                    else:
                        logger.warning(
                            f"⚠️ Unexpected buy-to-cover order status: {order_status}")

                except Exception as db_error:
                    logger.error(
                        f"❌ Error checking buy-to-cover execution: {db_error}")
                    logger.warning(
                        f"⚠️ Skipping trade recording - will sync on next check")

                self._update_daily_trade_count()
                logger.info(
                    f"Buy to cover order placed: {quantity} {symbol} at Rs.{price:.2f}")
                logger.info(f"   Order ID: {order_id}")

                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "message": f"Buy to cover order placed for {quantity} shares of {symbol}"
                }
            else:
                logger.error(
                    f"❌ Unexpected order response for buy to cover order: {order_response}")
                return {"success": False, "message": "Buy to cover order placement failed - unexpected response"}

        except Exception as e:
            logger.error(
                f"❌ Failed to execute buy to cover order for {symbol}: {e}")
            return {"success": False, "message": f"Buy to cover order failed: {str(e)}"}

    def execute_sell_order(self, symbol: str, signal_data: Dict = None) -> Dict:
        """Execute a sell order with comprehensive validation"""
        try:
            logger.info(
                f"Using database-integrated live executor for SELL {symbol}")

            if not self.enable_sell:
                logger.info(
                    "Sell disabled by configuration (ENABLE_SELL=false). Skipping sell for %s", symbol)
                return {"success": False, "message": "Sell disabled by configuration"}

            # Get current price
            current_price = self.get_real_time_price(symbol)
            if not current_price:
                return {"success": False, "message": "Failed to get current price"}

            # Get current holding from database
            session = None
            try:
                session = self.portfolio_manager.db.Session()
                portfolio = session.query(
                    Portfolio).filter_by(mode='live').first()
                if not portfolio:
                    return {"success": False, "message": "No live portfolio found"}

                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id,
                    ticker=symbol
                ).first()

                if (not holding or holding.quantity <= 0) and self._should_short_sell(symbol, signal_data):
                    logger.info(
                        f"No long holdings found for {symbol}; executing intraday short sell")
                    return self.execute_short_sell_order(symbol, signal_data or {})

                if not holding or holding.quantity <= 0:
                    return {"success": False, "message": "No position to sell"}

                # Calculate profit/loss
                pnl = (current_price - holding.avg_price) * holding.quantity

                # Get product type from signal_data or config and normalize for Dhan
                product_type = self._normalize_product_type(
                    signal_data.get('product_type') or self.get_current_product_type()) if signal_data else self.get_current_product_type()

                # Place order through Dhan API with comprehensive validation
                logger.info(
                    f"Attempting to place sell order for {holding.quantity} {symbol} (Product Type: {product_type})")
                order_response = self.dhan_client.place_order(
                    symbol=symbol,
                    quantity=holding.quantity,
                    order_type="MARKET",
                    side="SELL",
                    product_type=product_type  # Pass product type dynamically
                )

                order_id = order_response.get("orderId")
                if order_id:
                    # Track pending order
                    self.pending_orders[order_id] = {
                        "symbol": symbol,
                        "quantity": holding.quantity,
                        "side": "SELL",
                        "price": current_price,
                        "timestamp": datetime.now().isoformat(),
                        "signal_data": signal_data
                    }

                    # Check order status immediately after placing to see if it's already executed
                    try:
                        self.check_and_update_orders()
                    except Exception as status_error:
                        logger.warning(
                            f"Failed to check order status immediately after placement: {status_error}")

                    # Check if the order was immediately executed by checking the status
                    try:
                        order_details = self.dhan_client.get_order_by_id(
                            order_id)
                        if order_details and order_details.get("orderStatus", "").upper() == "TRADED":
                            # Order was immediately executed, record it in database
                            self.portfolio_manager.record_trade(
                                ticker=symbol,
                                action="sell",
                                quantity=holding.quantity,
                                price=current_price,
                                pnl=pnl,
                                stop_loss=signal_data.get("stop_loss"),
                                take_profit=signal_data.get("take_profit"),
                                product_type=product_type  # Pass product type for storage
                            )
                            logger.info(
                                f"✅ Sell trade executed and recorded: {holding.quantity} {symbol} at Rs.{current_price:.2f}")

                            # Also update the portfolio after execution
                            self._update_portfolio_after_execution(
                                symbol, "SELL", holding.quantity, current_price
                            )
                        else:
                            # Order is pending, don't record in database yet - wait for execution
                            logger.info(
                                f"⏳ Sell order placed but pending execution: {holding.quantity} {symbol} at Rs.{current_price:.2f}")
                            logger.info(f"   Order ID: {order_id}")
                            logger.info(
                                f"   Current status: {order_details.get('orderStatus', 'UNKNOWN') if order_details else 'UNABLE_TO_FETCH_STATUS'}")
                    except Exception as db_error:
                        logger.error(
                            f"❌ Error checking immediate execution or recording trade: {db_error}")
                        # As fallback, record the trade anyway to maintain consistency
                        try:
                            self.portfolio_manager.record_trade(
                                ticker=symbol,
                                action="sell",
                                quantity=holding.quantity,
                                price=current_price,
                                pnl=pnl,
                                stop_loss=signal_data.get("stop_loss"),
                                take_profit=signal_data.get("take_profit"),
                                product_type=product_type  # Pass product type for storage
                            )
                            logger.info(
                                f"✅ Sell trade recorded after error (fallback): {holding.quantity} {symbol} at Rs.{current_price:.2f}")
                        except Exception as fallback_error:
                            logger.error(
                                f"❌ Fallback recording also failed: {fallback_error}")

                    return {
                        "success": True,
                        "order_id": order_id,
                        "quantity": holding.quantity,
                        "price": current_price,
                        "pnl": pnl,
                        "message": f"Sell order placed for {holding.quantity} shares of {symbol}"
                    }
                else:
                    logger.error(
                        f"❌ Unexpected order response for sell order: {order_response}")
                    return {"success": False, "message": "Sell order placement failed - unexpected response"}

            finally:
                if session:
                    session.close()

        except ValueError as ve:
            # Handle security ID validation errors specifically
            logger.error(f"❌ Security ID validation failed for {symbol}: {ve}")
            return {"success": False, "message": f"Security ID validation failed for {symbol}"}
        except Exception as e:
            logger.error(f"❌ Error executing sell order for {symbol}: {e}")
            return {"success": False, "message": f"Sell order failed: {str(e)}"}

    def check_and_update_orders(self) -> List[Dict]:
        """Check status of pending orders and update portfolio"""
        updated_orders = []

        try:
            # Get all orders from Dhan
            all_orders = self.dhan_client.get_orders()

            for order_id, pending_order in list(self.pending_orders.items()):
                # Find matching order in Dhan response
                dhan_order = next(
                    (o for o in all_orders if o.get("orderId") == order_id), None)

                if dhan_order:
                    order_status = dhan_order.get("orderStatus", "").upper()

                    if order_status == "TRADED":
                        # Order executed
                        executed_price = float(dhan_order.get(
                            "price", pending_order["price"]))
                        executed_qty = int(dhan_order.get(
                            "quantity", pending_order["quantity"]))

                        # Update portfolio
                        self._update_portfolio_after_execution(
                            pending_order["symbol"],
                            pending_order["side"],
                            executed_qty,
                            executed_price,
                            is_short_sell=pending_order.get(
                                "is_short_sell", False),
                            is_cover_short=pending_order.get(
                                "is_cover_short", False)
                        )

                        # Move to executed orders
                        self.executed_orders[order_id] = {
                            **pending_order,
                            "executed_price": executed_price,
                            "executed_quantity": executed_qty,
                            "execution_time": datetime.now().isoformat(),
                            "status": "EXECUTED"
                        }

                        updated_orders.append(self.executed_orders[order_id])
                        del self.pending_orders[order_id]

                        logger.info(
                            f"Order executed: {pending_order['side']} {executed_qty} {pending_order['symbol']} at Rs.{executed_price:.2f}")

                    elif order_status in ["CANCELLED", "REJECTED"]:
                        # Order failed
                        self.executed_orders[order_id] = {
                            **pending_order,
                            "status": order_status,
                            "failure_time": datetime.now().isoformat()
                        }

                        updated_orders.append(self.executed_orders[order_id])
                        del self.pending_orders[order_id]

                        logger.warning(
                            f"Order {order_status.lower()}: {pending_order['side']} {pending_order['symbol']}")

            return updated_orders

        except Exception as e:
            logger.error(f"Failed to check order status: {e}")
            return []

    def _update_portfolio_after_execution(self, symbol: str, side: str,
                                          quantity: int, price: float,
                                          is_short_sell: bool = False,
                                          is_cover_short: bool = False):
        """Update portfolio after order execution using database"""
        session = None
        try:
            session = self.portfolio_manager.db.Session()
            portfolio = session.query(Portfolio).filter_by(mode='live').first()
            if not portfolio:
                logger.error("No live portfolio found for execution update")
                return

            if side == "BUY" and not is_cover_short:
                # Standard long buy
                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id,
                    ticker=symbol
                ).first()

                if holding:
                    # Average down the price
                    total_qty = holding.quantity + quantity
                    total_cost = (holding.quantity *
                                  holding.avg_price) + (quantity * price)
                    avg_price = total_cost / total_qty

                    holding.quantity = total_qty
                    holding.avg_price = avg_price
                    holding.last_price = price
                else:
                    holding = Holding(
                        portfolio_id=portfolio.id,
                        ticker=symbol,
                        quantity=quantity,
                        avg_price=price,
                        last_price=price
                    )
                    session.add(holding)

                portfolio.cash -= quantity * price

                if symbol in self.portfolio_manager.current_holdings_dict:
                    existing = self.portfolio_manager.current_holdings_dict[symbol]
                    existing_qty = existing["qty"] + quantity
                    total_cost = (existing["qty"] * existing["avg_price"]) + (
                        quantity * price)
                    existing["qty"] = existing_qty
                    existing["avg_price"] = total_cost / existing_qty
                    existing["last_price"] = price
                else:
                    self.portfolio_manager.current_holdings_dict[symbol] = {
                        "qty": quantity,
                        "avg_price": price,
                        "last_price": price
                    }

            elif side == "BUY" and is_cover_short:
                # Cover a short position
                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id,
                    ticker=symbol
                ).first()

                if holding and holding.quantity < 0:
                    entry_price = holding.avg_price
                    pnl = (entry_price - price) * quantity
                    self.daily_pnl += pnl

                    remaining_qty = holding.quantity + quantity
                    if remaining_qty == 0:
                        session.delete(holding)
                        self.portfolio_manager.current_holdings_dict.pop(
                            symbol, None)
                    else:
                        holding.quantity = remaining_qty
                        holding.last_price = price
                        self.portfolio_manager.current_holdings_dict[symbol] = {
                            "qty": remaining_qty,
                            "avg_price": holding.avg_price,
                            "last_price": price
                        }

                    portfolio.cash -= quantity * price
                    portfolio.realized_pnl += pnl
                    logger.info(
                        f"Short cover P&L: Rs.{pnl:.2f} for {quantity} {symbol}")
                else:
                    logger.warning(
                        f"No short position found for cover buy of {symbol}, treating as normal buy")
                    self._update_portfolio_after_execution(
                        symbol, "BUY", quantity, price)
                    return

            elif side == "SELL" and is_short_sell:
                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id,
                    ticker=symbol
                ).first()

                if holding:
                    if holding.quantity < 0:
                        total_short_qty = abs(holding.quantity) + quantity
                        average_price = ((abs(holding.quantity) * holding.avg_price) +
                                         (quantity * price)) / total_short_qty
                        holding.quantity -= quantity
                        holding.avg_price = average_price
                        holding.last_price = price
                    else:
                        # Convert a long holding into a short position
                        net_qty = holding.quantity - quantity
                        holding.quantity = net_qty
                        holding.avg_price = price
                        holding.last_price = price
                else:
                    holding = Holding(
                        portfolio_id=portfolio.id,
                        ticker=symbol,
                        quantity=-quantity,
                        avg_price=price,
                        last_price=price
                    )
                    session.add(holding)

                portfolio.cash += quantity * price
                self.portfolio_manager.current_holdings_dict[symbol] = {
                    "qty": holding.quantity,
                    "avg_price": holding.avg_price,
                    "last_price": price
                }
                logger.info(
                    f"Opened short position: {quantity} {symbol} at Rs.{price:.2f}")

            elif side == "SELL":
                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id,
                    ticker=symbol
                ).first()

                if holding and holding.quantity >= quantity:
                    pnl = quantity * (price - holding.avg_price)
                    self.daily_pnl += pnl

                    remaining_qty = holding.quantity - quantity
                    if remaining_qty > 0:
                        holding.quantity = remaining_qty
                        holding.last_price = price
                        if symbol in self.portfolio_manager.current_holdings_dict:
                            self.portfolio_manager.current_holdings_dict[symbol]["qty"] = remaining_qty
                            self.portfolio_manager.current_holdings_dict[symbol]["last_price"] = price
                    else:
                        if symbol in self.portfolio_manager.current_holdings_dict:
                            del self.portfolio_manager.current_holdings_dict[symbol]
                else:
                    logger.warning(
                        f"Attempted to sell {quantity} {symbol} but holdings are insufficient or missing")

                portfolio.cash += quantity * price
                portfolio.realized_pnl += pnl
                logger.info(f"Trade P&L: Rs.{pnl:.2f} for {quantity} {symbol}")

            portfolio.last_updated = datetime.now()
            session.commit()

            self.portfolio_manager.current_portfolio = portfolio
            self.portfolio_manager.refresh_holdings_from_database()

        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Failed to update portfolio after execution: {e}")
            raise
        finally:
            if session:
                session.close()

    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits are exceeded using database"""
        current_date = datetime.now().date()

        # Reset daily counters if new day
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.daily_pnl = 0.0
            self.last_trade_date = current_date

        # Check trade count limit
        if self.daily_trade_count >= self.max_trade_limit:
            logger.warning(
                f"Daily trade limit reached: {self.daily_trade_count}")
            return True

        # Check daily loss limit using fresh database session
        session = None
        try:
            session = self.portfolio_manager.db.Session()
            portfolio = session.query(Portfolio).filter_by(mode='live').first()
            if portfolio:
                total_value = portfolio.cash  # Simple calculation
                max_loss = total_value * self.max_daily_loss

                if self.daily_pnl < -max_loss:
                    logger.warning(
                        f"Daily loss limit reached: Rs.{self.daily_pnl:.2f}")
                    return True
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
        finally:
            if session:
                session.close()

        return False

    def _update_daily_trade_count(self):
        """Update daily trade counter"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 1
            self.last_trade_date = current_date
        else:
            self.daily_trade_count += 1
