#!/usr/bin/env python3
"""
Dhan Account Sync Service - Real-time background sync with Dhan account
"""

import os
import json
import sqlite3
import asyncio
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DhanSyncService:
    """Background service to sync portfolio with Dhan account in real-time"""

    def __init__(self, sync_interval: int = 300):  # Increased to 5 minutes
        """
        Initialize Dhan sync service

        Args:
            sync_interval: Sync interval in seconds (default: 30 seconds)
        """
        self.sync_interval = sync_interval
        self.client_id = os.getenv("DHAN_CLIENT_ID")
        self.access_token = os.getenv("DHAN_ACCESS_TOKEN")
        self.is_running = False
        self.last_sync_time = None
        self.last_known_balance = 0.0

        # Validate credentials
        if not self.client_id or not self.access_token:
            logger.error("Dhan credentials not found in environment variables")
            raise ValueError("Dhan credentials not configured")

        logger.debug(
            f"Dhan Sync Service initialized with {sync_interval}s interval")

    def get_dhan_funds(self) -> Optional[Dict[str, Any]]:
        """Get current funds from Dhan API"""
        try:
            headers = {
                'access-token': self.access_token,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }

            # First try fundlimit endpoint as it provides accurate real-time balance with zero parameters
            try:
                response = requests.get(
                    "https://api.dhan.co/v2/fundlimit",
                    headers=headers,
                    timeout=10
                )

                if response.status_code == 200:
                    funds_data = response.json()
                    logger.debug(
                        f"Fetched funds data from fundlimit: {funds_data}")
                    return funds_data
                else:
                    logger.warning(
                        f"Fundlimit endpoint returned status {response.status_code}: {response.text}")
            except Exception as fundlimit_error:
                logger.error(
                    f"Fundlimit endpoint failed: {fundlimit_error}")

            # Fallback to profile endpoint
            try:
                response = requests.get(
                    "https://api.dhan.co/v2/profile",
                    headers=headers,
                    timeout=10
                )

                if response.status_code == 200:
                    profile_data = response.json()
                    logger.debug(f"Fetched profile data: {profile_data}")
                    # Convert profile data to funds-like structure
                    return {
                        "availableBalance": profile_data.get("availableBalance", 0.0),
                        "marginUsed": profile_data.get("marginUsed", 0.0),
                        "totalBalance": profile_data.get("totalBalance", 0.0),
                        "clientName": profile_data.get("clientName", "Unknown"),
                        "clientId": profile_data.get("clientId", self.client_id)
                    }
            except Exception as profile_error:
                logger.error(
                    f"Profile endpoint failed for funds: {profile_error}")

            return None

        except Exception as e:
            logger.error(f"Error fetching funds from Dhan API: {e}")
            return None

    def get_dhan_holdings(self) -> Optional[list]:
        """Get current holdings from Dhan API"""
        try:
            headers = {
                'access-token': self.access_token,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }

            try:
                response = requests.get(
                    "https://api.dhan.co/v2/holdings",
                    headers=headers,
                    timeout=10
                )
            except Exception as holdings_error:
                logger.debug(f"Holdings endpoint failed: {holdings_error}")
                # Return empty list for any holdings error
                return []

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(
                    f"Dhan holdings API returned status {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching holdings from Dhan API: {e}")
            return []

    def update_database(self, cash_amount: float, holdings: list) -> bool:
        """Update the database with new portfolio data"""
        try:
            # Get the correct path to the database
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            db_path = os.path.join(project_root, 'data', 'trading.db')

            if not os.path.exists(db_path):
                logger.error(f"Database not found at {db_path}")
                return False

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Only update cash if it's a valid, non-zero amount
            # This prevents resetting balance to 0 when API fails
            effective_cash = cash_amount if cash_amount > 0 else None
            
            if effective_cash is not None:
                cursor.execute("""
                    UPDATE portfolios 
                    SET cash = ?, last_updated = ?
                    WHERE mode = 'live'
                """, (effective_cash, datetime.now().isoformat()))
            else:
                # Don't update cash if it's zero or invalid - keep existing value
                cursor.execute("""
                    UPDATE portfolios 
                    SET last_updated = ?
                    WHERE mode = 'live'
                """, (datetime.now().isoformat(),))

            if cursor.rowcount == 0:
                # Insert new live portfolio if it doesn't exist
                # Only use cash_amount if it's valid and non-zero, otherwise use default
                effective_cash = cash_amount if cash_amount > 0 else 50000.0  # Default starting balance
                effective_starting_balance = cash_amount if cash_amount > 0 else 50000.0  # Default starting balance
                cursor.execute("""
                    INSERT INTO portfolios (mode, cash, starting_balance, realized_pnl, unrealized_pnl, last_updated)
                    VALUES ('live', ?, ?, 0.0, 0.0, ?)
                """, (effective_cash, effective_starting_balance, datetime.now().isoformat()))
                if cash_amount <= 0:
                    logger.info(f"Created new live portfolio with default cash (API returned invalid balance: Rs.{cash_amount:.2f})")
                else:
                    logger.info(f"Created new live portfolio with cash from Dhan: Rs.{effective_cash:.2f}")

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Database update failed: {e}")
            return False

    def update_json_files(self, cash_amount: float, holdings: list) -> bool:
        """Update JSON files with new portfolio data"""
        try:
            # Prepare holdings data
            holdings_dict = {}
            for holding in holdings:
                symbol = holding.get("tradingSymbol", "")
                quantity = int(holding.get("totalQty", 0))
                avg_price = float(holding.get("avgCostPrice", 0))

                if quantity > 0:
                    holdings_dict[symbol] = {
                        "qty": quantity,
                        "avg_price": avg_price,
                        "last_price": avg_price
                    }

            # Calculate total holdings value
            total_holdings_value = sum(
                h["qty"] * h["avg_price"] for h in holdings_dict.values()
            )

            # Get the correct path to data directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            data_dir = os.path.join(project_root, 'data')

            # Update portfolio_india_live.json
            # Only use cash_amount if it's valid and non-zero, otherwise keep existing cash
            effective_cash = cash_amount if cash_amount > 0 else self.last_known_balance
            effective_starting_balance = max(cash_amount, self.last_known_balance) if cash_amount > 0 else self.last_known_balance
            portfolio_data = {
                "cash": effective_cash,
                "holdings": holdings_dict,
                "starting_balance": effective_starting_balance,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "last_updated": datetime.now().isoformat()
            }

            portfolio_file = os.path.join(
                data_dir, 'portfolio_india_live.json')
            with open(portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=4)

            # Update live_portfolio.json
            # Only use cash_amount if it's valid and non-zero, otherwise keep existing cash
            effective_cash = cash_amount if cash_amount > 0 else self.last_known_balance
            effective_total_value = (cash_amount + total_holdings_value) if cash_amount > 0 else (self.last_known_balance + total_holdings_value)
            effective_starting_balance = max(cash_amount, self.last_known_balance) if cash_amount > 0 else self.last_known_balance
            live_portfolio_data = {
                "cash": effective_cash,
                "total_value": effective_total_value,
                "starting_balance": effective_starting_balance,
                "holdings": holdings_dict,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "last_updated": datetime.now().isoformat()
            }

            live_portfolio_file = os.path.join(data_dir, 'live_portfolio.json')
            with open(live_portfolio_file, 'w') as f:
                json.dump(live_portfolio_data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"JSON update failed: {e}")
            return False

    def sync_once(self) -> bool:
        """Perform a single sync operation"""
        try:
            # Get data from Dhan
            funds_data = self.get_dhan_funds()
            holdings_data = self.get_dhan_holdings()

            if funds_data is None:
                logger.warning("Failed to fetch funds data from Dhan")
                return False

            if holdings_data is None:
                holdings_data = []

            # Extract cash balance (tolerant to different key names / typos)
            current_balance = 0.0
            try:
                logger.debug(
                    f"Dhan funds data keys: {list(funds_data.keys()) if isinstance(funds_data, dict) else 'Not a dict'}")

                # Common keys returned by Dhan endpoints (with more comprehensive list)
                possible_keys = [
                    'availableBalance', 'availablebalance', 'available_balance',
                    'sodLimit', 'sodlimit', 'sod_limit',
                    'netBalance', 'netbalance', 'net_balance',
                    'availablecash', 'available_cash', 'availableCash',
                    'netAvailableMargin', 'netAvailableCash',
                    'usableCash', 'usable_cash', 'UsableCash',
                    'cash', 'Cash'
                ]

                for key in possible_keys:
                    if isinstance(funds_data, dict) and key in funds_data:
                        value = funds_data[key]
                        if value is not None:
                            try:
                                current_balance = float(value)
                                logger.debug(
                                    f"Found balance in key '{key}': ₹{current_balance}")
                                break
                            except (ValueError, TypeError):
                                continue

                # If still no balance found, try to find the first numeric value in the dict
                if current_balance == 0.0 and isinstance(funds_data, dict):
                    for k, v in funds_data.items():
                        if isinstance(v, (int, float)) and 'balance' in k.lower():
                            current_balance = float(v)
                            logger.debug(
                                f"Found balance in '{k}' field: ₹{current_balance}")
                            break

                if current_balance == 0.0 and isinstance(funds_data, dict):
                    for k, v in funds_data.items():
                        if isinstance(v, (int, float)) and v > 100:  # Likely a balance amount
                            current_balance = float(v)
                            logger.debug(
                                f"Using first large numeric value from '{k}' field: ₹{current_balance}")
                            break

            except Exception as e:
                logger.error(f"Error extracting balance from funds data: {e}")
                current_balance = 0.0

            # Check if balance changed
            balance_changed = abs(
                current_balance - self.last_known_balance) > 0.01

            if balance_changed:
                logger.info(
                    f"💰 Balance changed: ₹{self.last_known_balance} → ₹{current_balance}")

                # Update database and JSON files
                db_success = self.update_database(
                    current_balance, holdings_data)
                json_success = self.update_json_files(
                    current_balance, holdings_data)

                if db_success and json_success:
                    self.last_known_balance = current_balance
                    self.last_sync_time = datetime.now()
                    logger.debug(
                        f"✅ Portfolio synced successfully: ₹{current_balance}")
                    return True
                else:
                    logger.error("Failed to update portfolio data")
                    return False
            else:
                logger.debug(f"No balance change detected: ₹{current_balance}")
                self.last_sync_time = datetime.now()
                return True

        except Exception as e:
            logger.error(f"Sync operation failed: {e}")
            return False

    async def start_background_sync(self):
        """Start the background sync service"""
        self.is_running = True
        logger.info(
            f"🚀 Starting Dhan sync service (every {self.sync_interval}s)")

        # Initial sync
        self.sync_once()

        while self.is_running:
            try:
                await asyncio.sleep(self.sync_interval)
                if self.is_running:
                    self.sync_once()

            except Exception as e:
                logger.error(f"Error in background sync loop: {e}")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying

    def stop(self):
        """Stop the background sync service"""
        self.is_running = False
        logger.info("🛑 Dhan sync service stopped")


# Global sync service instance
_sync_service = None


def get_sync_service() -> Optional[DhanSyncService]:
    """Get the global sync service instance"""
    return _sync_service


def start_sync_service(sync_interval: int = 300) -> DhanSyncService:  # 5 minutes default
    """Start the global sync service"""
    global _sync_service

    if _sync_service is None:
        try:
            _sync_service = DhanSyncService(sync_interval)
            # Start the background task
            asyncio.create_task(_sync_service.start_background_sync())
            logger.info("Dhan sync service started successfully")
        except Exception as e:
            logger.error(f"Failed to start sync service: {e}")
            _sync_service = None

    return _sync_service


def stop_sync_service():
    """Stop the global sync service"""
    global _sync_service

    if _sync_service:
        _sync_service.stop()
        _sync_service = None
        logger.info("Dhan sync service stopped")


if __name__ == "__main__":
    # Test the sync service
    async def test_sync():
        service = DhanSyncService(sync_interval=10)  # 10 seconds for testing
        await service.start_background_sync()

    asyncio.run(test_sync())
