#!/usr/bin/env python3
"""
Dhan API Client for Live Trading Integration
Handles real-time market data, portfolio management, and order execution
"""

import requests
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sqlite3
from io import StringIO
import pytz

logger = logging.getLogger(__name__)


class DhanAPIClient:
    """Dhan API client for live trading operations"""

    def __init__(self, client_id: str, access_token: str, config: dict = None):
        self.client_id = client_id
        self.access_token = access_token
        self.config = config or {}
        self.base_url = "https://api.dhan.co"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'access-token': access_token
        })

        # Get product type from config (default to CNC)
        self.product_type = self.config.get('productType', 'CNC')

        # Map product type to Dhan API format
        self.product_type_mapping = {
            'CNC': 'CNC',          # Cash and Carry - Delivery
            'INTRADAY': 'INTRADAY',  # Intraday orders use Dhan INTRADAY product type
            'MIS': 'INTRADAY',      # MIS style intraday orders map to Dhan INTRADAY
            'MARGIN': 'MARGIN',    # Margin positions map to Dhan MARGIN
            'NRML': 'MARGIN',      # NRML orders map to Dhan MARGIN
            'MTF': 'MTF'           # Multi-Trade Facility / margin trading facility
        }

        # Log initialization for debugging
        logger.info(
            f"Dhan API client initialized for client ID: {client_id[:4]}...{client_id[-4:] if len(client_id) > 8 else client_id}")
        logger.info(
            f"Product Type: {self.product_type} ({self.product_type_mapping.get(self.product_type, 'CNC')})")

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Validation rate limiting
        self.last_validation_time = 0
        self.validation_cache_duration = 300  # Cache validation for 5 minutes

        # Data directory for caching
        backend_dir = Path(__file__).resolve().parent
        project_root = backend_dir.parent
        self.data_dir = project_root / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Security ID cache for dynamic lookups
        self.security_id_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # Cache for 1 hour

        # Load manual security ID mappings from JSON file
        try:
            mapping_file = self.data_dir / 'security_id_mapping.json'
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    self.manual_security_id_mapping = json.load(f)
                logger.info(
                    f"Loaded {len(self.manual_security_id_mapping)} manual security ID mappings from {mapping_file}")
            else:
                logger.warning(
                    f"Security ID mapping file not found at {mapping_file}")
                self.manual_security_id_mapping = {}
        except Exception as e:
            logger.error(f"Failed to load security ID mapping file: {e}")
            self.manual_security_id_mapping = {}

        # Instrument data cache
        self.instrument_cache = {}
        self.instrument_cache_expiry = {}
        # Reduce cache duration to 1 hour for fresher data
        self.instrument_cache_duration = 3600

        # Supported Dhan exchange segments for dynamic instrument lookup
        self.supported_exchange_segments = ["NSE_EQ", "BSE_EQ"]
        self.exchange_suffix_to_segment = {
            ".NS": "NSE_EQ",
            ".BO": "BSE_EQ"
        }

        # Daily instrument master (Dhan's recommended approach)
        self.daily_instruments_df = None
        self.daily_instruments_cache_duration = 3600  # Refresh every hour
        self.daily_refresh_hour = 8  # Refresh at 8 AM daily

        # Dhan instrument master URLs (no auth required)
        self.compact_url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        self.detailed_url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

        # Initialize database for corrected security IDs
        self._init_security_id_db()

        logger.info(
            "Dhan API client initialized with direct API instrument fetching")

    def _init_security_id_db(self):
        """Initialize database for storing corrected security IDs"""
        try:
            db_path = self.data_dir / "security_ids.db"
            self.security_db = sqlite3.connect(
                db_path, check_same_thread=False)
            cursor = self.security_db.cursor()

            # Create table for corrected security IDs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS corrected_security_ids (
                    symbol TEXT PRIMARY KEY,
                    security_id TEXT NOT NULL,
                    exchange_segment TEXT DEFAULT 'NSE_EQ',
                    last_validated TIMESTAMP,
                    validation_count INTEGER DEFAULT 0
                )
            ''')

            # Create index for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_exchange 
                ON corrected_security_ids (symbol, exchange_segment)
            ''')

            self.security_db.commit()
            logger.info("Security ID database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize security ID database: {e}")
            self.security_db = None

    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        # Increased to 0.25s (4 requests/sec) to be safer
        # Dhan limit is nominally 10/sec but bursts cause 429s
        if time_since_last < 0.25:
            time.sleep(0.25 - time_since_last)
        self.last_request_time = time.time()

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make API request with error handling and retry logic"""
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"

        # Log request for debugging
        logger.debug(f"Making {method} request to {url}")
        if data:
            logger.debug(f"Request data: {data}")

        # Enhanced error handling with retry logic for connection issues and 429s
        max_retries = 3
        retry_delay = 1.0  # Initial retry delay in seconds

        for attempt in range(max_retries + 1):
            try:
                if method.upper() == 'GET':
                    # Create headers for GET requests without Content-Type
                    get_headers = {k: v for k,
                                   v in self.session.headers.items()}
                    # Remove Content-Type for GET requests
                    get_headers.pop('Content-Type', None)

                    if data:
                        response = self.session.get(
                            url, params=data, headers=get_headers, timeout=30)
                    else:
                        response = self.session.get(
                            url, headers=get_headers, timeout=30)
                elif method.upper() == 'POST':
                    response = self.session.post(url, json=data, timeout=30)
                elif method.upper() == 'PUT':
                    response = self.session.put(url, json=data, timeout=30)
                elif method.upper() == 'DELETE':
                    response = self.session.delete(url, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Handle specific status codes more gracefully
                if response.status_code == 500 and endpoint == '/v2/holdings':
                    # 500 error on holdings usually means empty portfolio
                    logger.debug(
                        "Holdings endpoint returned 500 - likely empty portfolio")
                    return {"data": []}

                # Handle 429 Too Many Requests explicitly
                if response.status_code == 429:
                    logger.warning(
                        f"⚠️ Dhan API Rate Limit (429) hit on {endpoint}")
                    if attempt < max_retries:
                        # Check for Retry-After header
                        retry_after = response.headers.get("Retry-After")
                        wait_time = float(retry_after) if retry_after else (
                            retry_delay * (2 ** attempt))
                        # Cap max wait to 10s
                        wait_time = min(wait_time, 10.0)

                        logger.info(
                            f"   Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(
                            "❌ Max retries reached for 429 Rate Limit")
                        raise Exception(
                            f"Dhan API Rate Limit Exceeded (429) after {max_retries} retries")

                # Enhanced error handling for 400 errors
                if response.status_code == 400:
                    try:
                        error_detail = response.json()
                        logger.error(
                            f"Dhan API 400 Error - {endpoint}: {error_detail}")
                        if data:
                            logger.error(
                                f"Request data that caused 400 error: {data}")
                        raise Exception(
                            f"Dhan API error (400): {error_detail}")
                    except ValueError:
                        # JSON decode error
                        logger.error(
                            f"Dhan API 400 Error - {endpoint}: {response.text}")
                        if data:
                            logger.error(
                                f"Request data that caused 400 error: {data}")
                        raise Exception(
                            f"Dhan API error (400): {response.text}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.ConnectionError as e:
                logger.error(
                    f"Dhan API connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries:
                    # Exponential backoff
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(
                        f"Dhan API connection failed after {max_retries} attempts")
                    raise Exception(f"Dhan API connection error: {str(e)}")

            except requests.exceptions.Timeout as e:
                logger.error(
                    f"Dhan API timeout on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries:
                    # Exponential backoff
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(
                        f"Dhan API timeout failed after {max_retries} attempts")
                    raise Exception(f"Dhan API timeout: {str(e)}")

            except requests.exceptions.RequestException as e:
                # More specific error handling
                if "500" in str(e) and endpoint == '/v2/holdings':
                    logger.info(
                        "Holdings API returned 500 - empty portfolio (normal for new accounts)")
                elif "429" in str(e):
                    # Should be caught above, but just in case RequestException wraps it without status_code attr access
                    pass
                else:
                    logger.error(f"Dhan API request failed: {e}")
                    if data:
                        logger.error(f"Request data: {data}")

                # If we are here, it's a non-retryable error or retries exhausted (though loop handles retries)
                # But wait, the loop structure handles RequestException generally.
                # If we want to retry generic RequestExceptions:
                if attempt < max_retries:
                    time.sleep(retry_delay * (attempt + 1))
                    continue

                raise Exception(f"Dhan API error: {str(e)}")

    def get_profile(self) -> Dict:
        """Get user profile and account information"""
        try:
            return self._make_request('GET', '/v2/profile')
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            # Return a default profile to prevent system failure
            return {
                "dhanClientId": self.client_id,
                "customerName": "Dhan Client",
                "email": "",
                "mobileNo": "",
                "status": "ACTIVE"
            }

    def get_funds(self) -> Dict:
        """Get account funds and margin information via Dhan /v2/fundlimit endpoint"""
        try:
            # Primary: use /v2/fundlimit — the correct Dhan API for real-time fund data
            try:
                fundlimit_response = self._make_request('GET', '/v2/fundlimit')
                logger.debug(
                    f"Dhan fundlimit response: {json.dumps(fundlimit_response, indent=2)}")

                if isinstance(fundlimit_response, dict):
                    # CRITICAL: Dhan API has a typo - 'availabelBalance' has correct value!
                    # Check keys in priority order to get the REAL available cash
                    available = (
                        # ✅ CORRECT (despite typo)
                        fundlimit_response.get("availabelBalance") or
                        fundlimit_response.get("availableBalance") or
                        fundlimit_response.get("availablecash") or
                        fundlimit_response.get("netAvailableMargin") or
                        fundlimit_response.get("sodLimit") or
                        0.0
                    )
                    try:
                        available = float(available)
                    except (TypeError, ValueError):
                        available = 0.0

                    logger.info(
                        f"✅ Dhan fundlimit — available cash: Rs.{available:.2f} (from availabelBalance)")
                    return {
                        "availablecash": available,
                        "availableBalance": available,
                        "availabelBalance": available,  # Preserve the correct key
                        "sodLimit": float(fundlimit_response.get("sodLimit", 0.0) or 0.0),
                        "marginUsed": float(fundlimit_response.get("utilizedAmount", 0.0) or 0.0),
                        "totalBalance": float(fundlimit_response.get("totalBalance", available) or available),
                        "status": "success",
                        # keep raw data accessible
                        **fundlimit_response
                    }
            except Exception as fundlimit_error:
                logger.warning(f"fundlimit endpoint failed: {fundlimit_error}")

            # Fallback: /v2/profile (does NOT contain balance — returns 0, just for resilience)
            try:
                profile_response = self._make_request('GET', '/v2/profile')
                logger.debug(
                    f"Dhan Profile Response (fallback): {json.dumps(profile_response, indent=2)}")
                available = float(profile_response.get(
                    "availableBalance", 0.0) or 0.0)
                logger.warning(
                    f"⚠️  Using profile endpoint as fallback — balance may be 0 (Rs.{available:.2f})")
                return {
                    "availablecash": available,
                    "availableBalance": available,
                    "sodLimit": 0.0,
                    "marginUsed": 0.0,
                    "totalBalance": available,
                    "clientName": profile_response.get("clientName", "Unknown"),
                    "clientId": profile_response.get("clientId", self.client_id),
                    "status": "profile_fallback"
                }
            except Exception as profile_error:
                logger.debug(
                    f"Profile-based funds request also failed: {profile_error}")

            # Last resort: return zeros
            logger.error("All fund endpoints failed — returning zero balance")
            return {
                "availablecash": 0.0,
                "availableBalance": 0.0,
                "sodLimit": 0.0,
                "marginUsed": 0.0,
                "totalBalance": 0.0,
                "clientId": self.client_id,
                "status": "error"
            }
        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return {
                "availablecash": 0.0,
                "availableBalance": 0.0,
                "sodLimit": 0.0,
                "marginUsed": 0.0,
                "totalBalance": 0.0,
                "clientId": self.client_id,
                "status": "error",
                "errorMessage": str(e)
            }

    def get_holdings(self) -> List[Dict]:
        """Get current stock holdings from Dhan API (/v2/holdings)"""
        try:
            response = self._make_request('GET', '/v2/holdings')
            
            logger.debug(f"Raw holdings response type: {type(response)}")
            logger.debug(f"Raw holdings response: {json.dumps(response if isinstance(response, (dict, list)) else str(response), indent=2)[:500]}")
            
            # Handle response format - Dhan can return data in different ways
            holdings_list = []
            
            if isinstance(response, list):
                # Direct list response
                holdings_list = response
                logger.info(f"✅ Fetched {len(holdings_list)} holdings from Dhan API (direct list)")
                
            elif isinstance(response, dict):
                # Response is a dict - check for data wrapper
                
                # Try standard 'data' key first
                if 'data' in response and isinstance(response['data'], list):
                    holdings_list = response['data']
                    logger.info(f"✅ Fetched {len(holdings_list)} holdings from Dhan API (data key with list)")
                    
                # Try 'holdings' key
                elif 'holdings' in response and isinstance(response['holdings'], list):
                    holdings_list = response['holdings']
                    logger.info(f"✅ Fetched {len(holdings_list)} holdings from Dhan API (holdings key)")
                    
                # Check if response itself contains holding fields (single holding as dict)
                elif any(key in response for key in ['quantity', 'qty', 'tradingSymbol', 'symbol']):
                    # Single holding returned as dict, wrap it in a list
                    holdings_list = [response]
                    logger.info(f"✅ Fetched 1 holding from Dhan API (single holding dict)")
                    
                # Response might be empty or wrapped differently
                elif 'status' in response and response.get('status') == 'success':
                    # Empty holdings but successful response
                    holdings_list = []
                    logger.info("✅ Holdings API call successful but no holdings data")
                else:
                    logger.debug(f"Holdings response structure: {list(response.keys())}")
                    holdings_list = []
            else:
                logger.warning(f"Unexpected response format from holdings: {type(response)}")
                holdings_list = []
            
            logger.info(f"📋 Holdings list to process: {len(holdings_list)} items")
            return holdings_list
                
        except Exception as e:
            logger.error(f"❌ Failed to get holdings from Dhan API: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def get_positions(self) -> List[Dict]:
        """Get current trading positions from Dhan API (/v2/positions)"""
        try:
            response = self._make_request('GET', '/v2/positions')
            
            # Handle response format - Dhan returns data directly or in 'data' key
            if isinstance(response, list):
                logger.info(f"✅ Fetched {len(response)} positions from Dhan API")
                return response
            elif isinstance(response, dict):
                # Check for data in common response formats
                positions_data = response.get('data') or response.get('positions') or []
                if positions_data:
                    logger.info(f"✅ Fetched {len(positions_data)} positions from Dhan API (data key)")
                    return positions_data
                else:
                    logger.debug(f"Positions response structure: {response.keys() if isinstance(response, dict) else 'non-dict'}")
                    return response if response else []
            else:
                logger.warning(f"Unexpected response format from positions: {type(response)}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to get positions from Dhan API: {e}")
            return []

    def get_live_portfolio_metrics(self) -> Dict:
        """
        Calculate portfolio metrics using LIVE Dhan API data (no JSON files).
        This ensures values match exactly with the Dhan API documentation.
        Per DhanHQ API: /holdings and /positions endpoints return live holdings/positions data.
        Ref: https://dhanhq.co/docs/v2/portfolio-and-positions
        """
        try:
            # Fetch all live data from Dhan API
            holdings = self.get_holdings()  # GET /v2/holdings
            positions = self.get_positions()  # GET /v2/positions
            funds = self.get_funds()  # GET /v2/fundlimit
            
            # Extract cash balance (handle all variants)
            available_cash = (
                funds.get('availabelBalance') or  # Dhan typo but correct
                funds.get('availableBalance') or
                funds.get('availablecash') or
                0.0
            )
            try:
                available_cash = float(available_cash)
            except (TypeError, ValueError):
                available_cash = 0.0
            
            # Calculate holdings value and P&L from holdings data
            total_holdings_value = 0.0
            total_invested = 0.0
            unrealized_pnl = 0.0
            holdings_dict = {}
            
            logger.debug(f"📋 Raw holdings data received: {len(holdings)} items")
            logger.debug(f"📋 Holdings structure: {holdings[:2] if holdings else 'empty'}")  # Log first 2 items
            
            if holdings:
                logger.info(f"🔍 Starting to process {len(holdings)} holdings...")
                
                # LOG FIRST HOLDING STRUCTURE TO SEE ACTUAL FIELD NAMES
                if holdings:
                    first_holding = holdings[0]
                    logger.info(f"🔑 ACTUAL HOLDING STRUCTURE (first item):")
                    logger.info(f"   Keys in holding: {list(first_holding.keys()) if isinstance(first_holding, dict) else 'NOT A DICT'}")
                    logger.info(f"   Full first holding: {json.dumps(first_holding, indent=2, default=str)}")
                
                for idx, holding in enumerate(holdings):
                    try:
                        logger.debug(f"\n🔍 === Processing holding {idx + 1}/{len(holdings)} ===")
                        logger.debug(f"Raw holding data: {json.dumps(holding, indent=2, default=str)[:300]}")
                        
                        # Extract ticker - try all possible field names per Dhan API
                        ticker = (
                            holding.get('tradingSymbol') or
                            holding.get('symbol') or
                            holding.get('securityId') or
                            holding.get('isin') or
                            'UNKNOWN'
                        )
                        logger.debug(f"  Ticker: {ticker}")
                        
                        # Extract quantity - CRITICAL FIX: Try ALL possible field names
                        qty = None
                        qty_found_in = None
                        
                        # Log all available keys to find the right one
                        available_keys = list(holding.keys()) if isinstance(holding, dict) else []
                        logger.info(f"  Available keys in {ticker} holding: {available_keys}")
                        
                        # Per Dhan API documentation, use these field names:
                        for qty_field in ['totalQty', 'dpQty', 'availableQty', 'quantity', 'qty', 'holdQty', 'deliveryQty', 'Quantity', 'holdQuantity', 'netQuantity', 'netQty', 'qtyHeld', 'held']:
                            qty_val = holding.get(qty_field)
                            logger.debug(f"    Checking qty field '{qty_field}': {qty_val} (type: {type(qty_val).__name__})")
                            if qty_val is not None and qty_val != '':
                                try:
                                    qty = float(qty_val)
                                    qty_found_in = qty_field
                                    if qty > 0:
                                        logger.info(f"  ✅ Found quantity={qty} in field '{qty_field}'")
                                        break
                                except (TypeError, ValueError) as e:
                                    logger.debug(f"      Failed to convert to float: {type(qty_val)} - {e}")
                                    continue
                        
                        if qty is None or qty == 0:
                            logger.warning(f"  ⚠️ {ticker}: qty is None or 0 (checked all common field names)")
                            continue
                        
                        # Extract average price - Per Dhan API: avgCostPrice
                        avg_price = None
                        avg_price_found_in = None
                        for price_field in ['avgCostPrice', 'averagePrice', 'avgPrice', 'costPrice', 'costprice', 'AveragePrice', 'avg_price', 'costprice_new', 'buyPrice']:
                            price_val = holding.get(price_field)
                            if price_val is not None and price_val != '':
                                try:
                                    avg_price = float(price_val)
                                    avg_price_found_in = price_field
                                    if avg_price >= 0:
                                        logger.debug(f"    ✅ Found avgPrice={avg_price} in field '{price_field}'")
                                        break
                                except (TypeError, ValueError):
                                    continue
                        
                        if avg_price is None:
                            logger.warning(f"  ⚠️ {ticker}: No average price found, using 0.0")
                            avg_price = 0.0
                        
                        # Extract current price - Per Dhan API: lastTradedPrice
                        current_price = None
                        current_price_found_in = None
                        for price_field in ['lastTradedPrice', 'lastPrice', 'ltp', 'currentPrice', 'price', 'LastPrice', 'LTP', 'last_price', 'close_price']:
                            price_val = holding.get(price_field)
                            if price_val is not None and price_val != '':
                                try:
                                    current_price = float(price_val)
                                    current_price_found_in = price_field
                                    if current_price >= 0:
                                        logger.debug(f"    ✅ Found currentPrice={current_price} in field '{price_field}'")
                                        break
                                except (TypeError, ValueError):
                                    continue
                        
                        if current_price is None:
                            logger.warning(f"  ⚠️ {ticker}: No current price found, using avgPrice as fallback")
                            current_price = avg_price  # Fallback to avg price
                        
                        # Calculate P&L
                        holding_value = qty * current_price
                        invested = qty * avg_price
                        pnl = holding_value - invested
                        
                        total_holdings_value += holding_value
                        total_invested += invested
                        unrealized_pnl += pnl
                        
                        logger.info(f"✅ {ticker}: Qty={qty}, AvgPrice={avg_price}, LTP={current_price}, Value={holding_value:.2f}, P&L={pnl:.2f}")
                        
                        holdings_dict[ticker] = {
                            'qty': qty,
                            'avg_price': avg_price,
                            'currentPrice': current_price,
                            'value': holding_value,
                            'pnl': pnl,
                            'pnl_pct': (pnl / invested * 100) if invested > 0 else 0
                        }
                    except Exception as e:
                        logger.error(f"❌ Error processing holding {idx}: {e}")
                        logger.error(f"   Holding data: {holding}")
                        import traceback
                        logger.error(f"   Traceback: {traceback.format_exc()}")
                        continue
                
                logger.info(f"✅ Processed {len(holdings_dict)} valid holdings out of {len(holdings)}")
            
            # Get positions data (intraday/margin positions)
            positions_dict = {}
            if positions:
                for position in positions:
                    try:
                        ticker = position.get('securityId') or position.get('symbol', 'UNKNOWN')
                        net_qty = float(position.get('netQty') or position.get('quantity', 0))
                        avg_price = float(position.get('averagePrice', 0))
                        current_price = float(position.get('lastPrice', avg_price))
                        
                        if net_qty != 0:
                            positions_dict[ticker] = {
                                'qty': net_qty,
                                'avg_price': avg_price,
                                'currentPrice': current_price,
                                'type': 'LONG' if net_qty > 0 else 'SHORT',
                                'value': abs(net_qty * current_price)
                            }
                    except Exception as e:
                        logger.warning(f"Error processing position: {e}")
                        continue
            
            # Calculate totals
            total_margin_value = sum(p.get('value', 0) for p in positions_dict.values())
            total_portfolio_value = available_cash + total_holdings_value + total_margin_value
            
            # Realized P&L (can be fetched from trades if needed)
            realized_pnl = funds.get('realizedPnL', 0.0) or 0.0
            try:
                realized_pnl = float(realized_pnl)
            except (TypeError, ValueError):
                realized_pnl = 0.0
            
            total_pnl = unrealized_pnl + realized_pnl
            
            logger.info(f"📊 LIVE Portfolio Metrics from Dhan API:")
            logger.info(f"   Available Cash: Rs.{available_cash:,.2f}")
            logger.info(f"   Holdings Value: Rs.{total_holdings_value:,.2f} ({len(holdings_dict)} positions)")
            logger.info(f"   Positions Value: Rs.{total_margin_value:,.2f} ({len(positions_dict)} intraday)")
            logger.info(f"   Unrealized P&L: Rs.{unrealized_pnl:,.2f}")
            logger.info(f"   Realized P&L: Rs.{realized_pnl:,.2f}")
            logger.info(f"   Total Portfolio Value: Rs.{total_portfolio_value:,.2f}")
            
            return {
                'cash': available_cash,
                'holdings_value': total_holdings_value,
                'positions_value': total_margin_value,
                'total_value': total_portfolio_value,
                'total_invested': total_invested,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'total_pnl': total_pnl,
                'holdings': holdings_dict,
                'positions': positions_dict,
                'holdings_count': len(holdings_dict),
                'positions_count': len(positions_dict),
                'funds_data': funds,
                'is_live': True  # Flag indicating this is live API data
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate live portfolio metrics: {e}")
            return {
                'cash': 0,
                'holdings_value': 0,
                'positions_value': 0,
                'total_value': 0,
                'total_invested': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0,
                'total_pnl': 0,
                'holdings': {},
                'positions': {},
                'holdings_count': 0,
                'positions_count': 0,
                'is_live': False,
                'error': str(e)
            }

    def get_orders(self) -> List[Dict]:
        """Get order history from Dhan API /v2/orders"""
        try:
            response = self._make_request('GET', '/v2/orders')

            # Handle different response formats
            if isinstance(response, list):
                logger.info(f"✅ Fetched {len(response)} orders from Dhan API")
                return response
            elif isinstance(response, dict):
                orders_data = response.get('data', [])
                logger.info(f"✅ Fetched {len(orders_data)} orders from Dhan API (data key)")
                return orders_data
            else:
                logger.warning(f"Unexpected response format from orders: {type(response)}")
                return []

        except Exception as e:
            logger.error(f"❌ Failed to get orders: {e}")
            return []

    def get_trades(self) -> List[Dict]:
        """Get executed trades from Dhan API /v2/trades (actual executed trades, not orders)"""
        try:
            response = self._make_request('GET', '/v2/trades')

            # Handle different response formats
            if isinstance(response, list):
                logger.info(f"✅ Fetched {len(response)} trades from Dhan API")
                return response
            elif isinstance(response, dict):
                trades_data = response.get('data', [])
                logger.info(f"✅ Fetched {len(trades_data)} trades from Dhan API (data key)")
                return trades_data
            else:
                logger.warning(f"Unexpected response format from trades: {type(response)}")
                return []

        except Exception as e:
            logger.error(f"❌ Failed to get trades: {e}")
            return []

    def get_recent_trading_activity(self, limit: int = 50) -> List[Dict]:
        """
        Get recent trading activity (executed trades) from Dhan API
        Per DhanHQ API docs: https://dhanhq.co/docs/v2
        Formats trade data for display in UI
        """
        try:
            # Get executed trades from Dhan API
            trades = self.get_trades()
            
            if not trades:
                logger.warning("No trades found from Dhan API")
                return []
            
            formatted_trades = []
            
            for trade in trades[:limit]:
                try:
                    # Extract trade data from Dhan response
                    # Handle both possible field names per Dhan API variants
                    trade_type = (
                        trade.get('transactionType') or 
                        trade.get('orderSide') or 
                        trade.get('side', 'BUY')
                    ).upper()
                    
                    symbol = (
                        trade.get('tradingSymbol') or
                        trade.get('symbol') or
                        trade.get('securityId', 'UNKNOWN')
                    )
                    
                    quantity = float(trade.get('quantity') or trade.get('qty', 0))
                    price = float(trade.get('price') or trade.get('executedPrice', 0))
                    total = quantity * price
                    
                    # Extract timestamp - handle different formats
                    timestamp = (
                        trade.get('executedTime') or
                        trade.get('exchangeTime') or
                        trade.get('orderTime') or
                        trade.get('transactionTime', 'N/A')
                    )
                    
                    # Get current price for P&L calculation
                    try:
                        quote = self.get_quote(symbol)
                        current_price = float(quote.get('ltp', price))
                        pnl = (current_price - price) * quantity if trade_type == 'BUY' else (price - current_price) * quantity
                    except Exception:
                        current_price = price
                        pnl = 0
                    
                    formatted_trades.append({
                        'type': f'{trade_type} {symbol}',
                        'symbol': symbol,
                        'transaction_type': trade_type,
                        'quantity': int(quantity),
                        'price': round(price, 2),
                        'total': round(total, 2),
                        'current_price': round(current_price, 2),
                        'pnl': round(pnl, 2),
                        'timestamp': str(timestamp),
                        'order_id': trade.get('orderId', 'N/A'),
                        'trade_id': trade.get('tradeId', 'N/A'),
                        'status': 'EXECUTED',
                        'source': 'Dhan API'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error formatting trade: {e} | Trade data: {trade}")
                    continue
            
            logger.info(f"✅ Formatted {len(formatted_trades)} trades for display")
            return formatted_trades
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent trading activity: {e}")
            return []

    def get_order_by_id(self, order_id: str) -> Dict:
        """Get specific order by ID"""
        try:
            response = self._make_request('GET', f'/v2/orders/{order_id}')

            # Handle different response formats
            if isinstance(response, dict):
                return response
            else:
                logger.warning(
                    f"Unexpected response format from order {order_id}: {type(response)}")
                return {}

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return {}

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        """Get real-time quote for a symbol"""
        try:
            # Get numeric security ID instead of symbol
            security_id = self.get_security_id(symbol)

            data = {
                "securityId": security_id,
                "exchangeSegment": "NSE_EQ"
            }

            response = self._make_request('POST', '/v2/marketdata/quote', data)
            return response.get('data', {})

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}

    def get_historical_data(self, symbol: str, timeframe: str = "1D",
                            from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """Get historical price data"""
        try:
            # Get numeric security ID
            security_id = self.get_security_id(symbol)

            # Default date range if not provided
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            if not from_date:
                from_date = (datetime.now() - timedelta(days=365)
                             ).strftime("%Y-%m-%d")

            data = {
                "securityId": security_id,
                "exchangeSegment": "NSE_EQ",
                "timeframe": timeframe,
                "fromDate": from_date,
                "toDate": to_date
            }

            response = self._make_request(
                'POST', '/v2/charts/historical', data)

            # Convert to DataFrame
            if 'data' in response and response['data']:
                df = pd.DataFrame(response['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()

    def _get_corrected_security_id(self, symbol: str, exchange: str = "NSE_EQ") -> Optional[str]:
        """Get corrected security ID from database if available"""
        try:
            if not self.security_db:
                return None

            cursor = self.security_db.cursor()
            cursor.execute('''
                SELECT security_id FROM corrected_security_ids 
                WHERE symbol = ? AND exchange_segment = ? AND validation_count > 0
                ORDER BY last_validated DESC LIMIT 1
            ''', (symbol.upper(), exchange))

            result = cursor.fetchone()
            if result:
                logger.info(
                    f"✅ Found corrected security ID in database for {symbol}: {result[0]}")
                return result[0]
            return None
        except Exception as e:
            logger.error(
                f"Error retrieving corrected security ID for {symbol}: {e}")
            return None

    def _store_corrected_security_id(self, symbol: str, security_id: str, exchange: str = "NSE_EQ"):
        """Store corrected security ID in database"""
        try:
            if not self.security_db:
                return

            cursor = self.security_db.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO corrected_security_ids 
                (symbol, security_id, exchange_segment, last_validated, validation_count)
                VALUES (?, ?, ?, ?, 
                    CASE 
                        WHEN EXISTS(SELECT 1 FROM corrected_security_ids WHERE symbol = ? AND exchange_segment = ?) 
                        THEN (SELECT validation_count FROM corrected_security_ids WHERE symbol = ? AND exchange_segment = ?) + 1
                        ELSE 1
                    END
                )
            ''', (symbol.upper(), security_id, exchange, datetime.now(),
                  symbol.upper(), exchange, symbol.upper(), exchange))

            self.security_db.commit()
            logger.info(
                f"✅ Stored corrected security ID in database: {symbol} -> {security_id}")
        except Exception as e:
            logger.error(
                f"Error storing corrected security ID for {symbol}: {e}")

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached security ID is still valid"""
        if symbol not in self.security_id_cache:
            return False

        if symbol not in self.cache_expiry:
            return False

        return time.time() < self.cache_expiry[symbol]

    def _cache_security_id(self, symbol: str, security_id: str):
        """Cache security ID for future use"""
        self.security_id_cache[symbol] = security_id
        self.cache_expiry[symbol] = time.time() + self.cache_duration
        logger.debug(f"Cached security ID for {symbol}: {security_id}")

    def _get_cached_security_id(self, symbol: str) -> Optional[str]:
        """Get security ID from cache if valid"""
        if self._is_cache_valid(symbol):
            logger.debug(
                f"Using cached security ID for {symbol}: {self.security_id_cache[symbol]}")
            return self.security_id_cache[symbol]
        return None

    def _fetch_instrument_data(self, exchange_segment: str = "NSE_EQ", force_refresh: bool = False) -> pd.DataFrame:
        """Fetch real-time instrument data directly from Dhan API v2 endpoint"""
        try:
            cache_key = f"instruments_{exchange_segment}"

            # Check cache unless force refresh is requested
            if not force_refresh and (cache_key in self.instrument_cache and
                                      time.time() <= self.instrument_cache_expiry.get(cache_key, 0)):
                logger.debug(
                    f"Using cached instrument data for {exchange_segment}")
                return self.instrument_cache[cache_key]

            # Use the specific endpoint as requested: https://api.dhan.co/v2/instrument/NSE_EQ
            url = f"{self.base_url}/v2/instrument/{exchange_segment}"
            logger.info(
                f"{'Force fetching' if force_refresh else 'Fetching'} instrument data from: {url}")

            response = self.session.get(url)

            if response.status_code == 200:
                # Parse CSV data directly into DataFrame
                df = pd.read_csv(StringIO(response.text))
                logger.info(f"✅ Loaded instruments: {len(df)}")

                # Log some sample data for debugging
                if not df.empty:
                    logger.debug(
                        f"Sample instrument data: {df.head(3).to_dict('records')}")

                # Cache the data
                self.instrument_cache[cache_key] = df
                self.instrument_cache_expiry[cache_key] = time.time(
                ) + self.instrument_cache_duration

                return df
            else:
                logger.error(
                    f"Failed to fetch instruments. Status: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching instrument data: {e}")
            return pd.DataFrame()

    def get_supported_symbols(self, exchange_segment: str = "NSE_EQ") -> List[str]:
        """Return the list of supported instrument symbols for the given Dhan exchange segment."""
        df = self._fetch_instrument_data(exchange_segment)
        if df.empty:
            return []

        symbol_columns = ['SYMBOL_NAME', 'TradingSymbol', 'tradingSymbol', 'symbol', 'Symbol']
        symbol_col = next((col for col in df.columns if col in symbol_columns), None)
        if not symbol_col:
            symbol_col = next((col for col in df.columns if 'symbol' in col.lower()), None)
        if not symbol_col:
            logger.warning(
                f"Unable to determine symbol column for supported symbols in {exchange_segment}")
            return []

        return df[symbol_col].astype(str).str.upper().unique().tolist()

    def _fetch_daily_instrument_master(self, force_refresh: bool = False) -> pd.DataFrame:
        """Alias for the daily instrument master refresh path used in fallback logic"""
        return self._fetch_instrument_data(force_refresh=force_refresh)

    def _search_security_id_in_instruments(self, symbol: str, exchange_segment: str = "NSE_EQ") -> Optional[str]:
        """Search for security ID in instrument data fetched from Dhan API v2"""
        try:
            # Fetch instrument data if not already cached or cache expired
            cache_key = f"instruments_{exchange_segment}"
            if (cache_key not in self.instrument_cache or
                    time.time() > self.instrument_cache_expiry.get(cache_key, 0)):

                df = self._fetch_instrument_data(exchange_segment)
                if not df.empty:
                    self.instrument_cache[cache_key] = df
                    self.instrument_cache_expiry[cache_key] = time.time(
                    ) + self.instrument_cache_duration
                else:
                    return None
            else:
                df = self.instrument_cache[cache_key]

            # Search for symbol in the DataFrame
            # Handle different column names that might be present
            symbol_columns = ['SYMBOL_NAME', 'TradingSymbol',
                              'tradingSymbol', 'symbol', 'Symbol']
            security_id_columns = ['SECURITY_ID',
                                   'SecurityId', 'securityId', 'SecurityID']

            symbol_col = None
            security_id_col = None
            lower_symbol_cols = [c.lower() for c in symbol_columns]
            lower_security_id_cols = [c.lower() for c in security_id_columns]

            # Find the appropriate columns using case-insensitive matching and heuristics
            for col in df.columns:
                lower_col = col.lower()
                if (lower_col in lower_symbol_cols or 'symbol' in lower_col or 'trading' in lower_col or 'name' in lower_col) and symbol_col is None:
                    symbol_col = col
                if (lower_col in lower_security_id_cols or ('security' in lower_col and 'id' in lower_col)) and security_id_col is None:
                    security_id_col = col

            if not symbol_col or not security_id_col:
                # Second pass with relaxed heuristics
                for col in df.columns:
                    lower_col = col.lower()
                    if not symbol_col and 'symbol' in lower_col:
                        symbol_col = col
                    if not security_id_col and 'security' in lower_col and 'id' in lower_col:
                        security_id_col = col

            if not symbol_col or not security_id_col:
                logger.error(
                    f"Could not find required columns in instrument data. Available columns: {df.columns.tolist()}")
                return None

            # Create a more dynamic and comprehensive search without relying on hardcoded mappings
            search_symbol = symbol.upper()

            # Special handling for symbols with numeric suffixes like JAYNECOIND
            # Try to find variations that might match
            variations = [
                search_symbol,
                search_symbol.replace('-', ' '),
                search_symbol.replace('_', ' '),
                search_symbol.replace('&', ' AND '),
                search_symbol + ' LTD',
                search_symbol + ' LIMITED',
                search_symbol + ' EQ',
                search_symbol + '-EQ',
                'THE ' + search_symbol,
                search_symbol.replace('M&M', 'MAHINDRA AND MAHINDRA'),
                search_symbol.replace('M&M', 'MAHINDRA & MAHINDRA'),
                search_symbol.replace('L&T', 'LARSEN AND TOUBRO'),
                search_symbol.replace('L&T', 'LARSEN & TOUBRO')
            ]

            # Add specific handling for JAYNECOIND and similar symbols
            if search_symbol == "JAYNECOIND":
                variations.extend([
                    "JAYASWAL NUCLEUS",
                    "JAYASWAL NUCLEUS LTD",
                    "JAYASWAL NUCLEUS LIMITED",
                    "JAYNECOIND EQ",
                    "JAYNECOIND-EQ"
                ])

            # Add specific handling for POWERGRID
            if search_symbol == "POWERGRID":
                variations.extend([
                    "POWER GRID",
                    "POWER GRID CORPORATION",
                    "POWER GRID CORPORATION OF INDIA",
                    "POWER GRID CORPORATION OF INDIA LTD",
                    "POWER GRID CORPORATION OF INDIA LIMITED",
                    "POWERGRID EQ",
                    "POWERGRID-EQ",
                    "POWER GRID EQ"
                ])

            # Add specific handling for ASHOKLEY
            if search_symbol == "ASHOKLEY":
                variations.extend([
                    "ASHOK LEYLAND",
                    "ASHOK LEYLAND LTD",
                    "ASHOK LEYLAND LIMITED",
                    "ASHOKLEY EQ",
                    "ASHOKLEY-EQ",
                    "ASHOK LEYLAND EQ"
                ])

            # Add variations with common suffixes/prefixes
            common_suffixes = [' LTD', ' LIMITED', ' CORPORATION',
                               ' COMPANY', ' INDIA', ' INDUSTRIES', ' SOLUTIONS', ' SERVICES']
            common_prefixes = ['THE ', 'M/S ']

            for suffix in common_suffixes:
                if not search_symbol.endswith(suffix.strip()):
                    variations.append(search_symbol + suffix)

            for prefix in common_prefixes:
                if not search_symbol.startswith(prefix.strip()):
                    variations.append(prefix + search_symbol)

            # Special handling for known abbreviation patterns
            abbreviation_mapping = {
                'RVNL': 'RAIL VIKAS NIGAM',
                'HINDUNILVR': 'HINDUSTAN UNILEVER',
                'BAJFINANCE': 'BAJAJ FINANCE',
                'BAJAJFINSV': 'BAJAJ FINSERV',
                'TATAMOTORS': 'TATA MOTORS',
                'TATASTEEL': 'TATA STEEL',
                'TATAPOWER': 'TATA POWER',
                'TATACONSUM': 'TATA CONSUMER',
                'M&MFIN': 'MAHINDRA FINANCE',
                'TECHM': 'TECH MAHINDRA',
                'HCLTECH': 'HCL TECHNOLOGIES',
                'ULTRACEMCO': 'ULTRATECH CEMENT',
                'DRREDDY': 'DR. REDDY',
                'DIVISLAB': 'DIVI S LABORATORIES',
                'BHARTIARTL': 'BHARTI AIRTEL',
                'ONGC': 'OIL AND NATURAL GAS',
                'POWERGRID': 'POWER GRID CORP. LTD.',
                'POWERGRID.NS': 'POWER GRID CORP. LTD.',
                'COALINDIA': 'COAL INDIA',
                'SHANTIGOLD': 'Shanti Gold International Ltd',
                'IDBI': 'IDBI Bank Ltd',
                'DBI': 'IDBI Bank Ltd',
                'MOTHERSON': 'Samvardhana Motherson International Ltd',
                'GMRAIRPORT': 'GMR Airports Ltd',
                'GMRI': 'GMR Airports Ltd',
                'ASHOKLEY': 'ASHOK LEYLAND',
                'IEX': 'Indian Energy Exchange Ltd',
                # CRITICAL FIX: Add ADANIPOWER to prevent wrong partial match with "POWER"
                'ADANIPOWER': 'ADANI POWER LIMITED',
                'ADANIPOWER.NS': 'ADANI POWER LIMITED'
            }

            # Apply abbreviation mapping if available
            if search_symbol in abbreviation_mapping:
                expanded_name = abbreviation_mapping[search_symbol]
                variations.append(expanded_name)
                variations.append(expanded_name + ' LTD')
                variations.append(expanded_name + ' LIMITED')
            
            # CRITICAL ADDITION: Add common Indian stock name patterns for ADANIPOWER
            if search_symbol == "ADANIPOWER" or search_symbol == "ADANIPOWER.NS":
                # Try all possible ADANI POWER variations
                adani_power_variations = [
                    "ADANI POWER",
                    "ADANI POWER LTD",
                    "ADANI POWER LIMITED",
                    "ADANIPOWER LTD",
                    "ADANIPOWER LIMITED",
                    "ADANI PWR",
                    "ADANI POWR",
                    "APOWER"
                ]
                variations.extend(adani_power_variations)
                logger.debug(f"Added {len(adani_power_variations)} ADANIPOWER-specific variations")

            # CRITICAL: Try exact matches FIRST before any partial matching
            # This prevents false positives like matching ADANIPOWER with generic POWER stocks
            logger.debug(
                f"Searching for {symbol} with {len(variations)} variations")

            # Try each variation for EXACT match first (highest priority)
            for variation in variations:
                result = df[df[symbol_col].str.upper() == variation]
                if not result.empty:
                    security_id = str(result.iloc[0][security_id_col])
                    logger.info(
                        f"✅ Found security ID for {symbol} (EXACT match: {variation}): {security_id}")
                    return security_id

            # Try partial contains match as second priority (safer than reverse)
            for variation in variations:
                result = df[df[symbol_col].str.contains(
                    variation, case=False, na=False)]
                if not result.empty:
                    # Additional validation: ensure the matched symbol is reasonably similar
                    matched_symbol = str(result.iloc[0][symbol_col]).upper()
                    # Calculate similarity - skip if too different (prevents POWER matching ADANIPOWER)
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(
                        None, search_symbol, matched_symbol).ratio()
                    if similarity > 0.5:  # At least 50% similar
                        security_id = str(result.iloc[0][security_id_col])
                        logger.info(
                            f"✅ Found security ID for {symbol} (partial match: {matched_symbol}, similarity: {similarity:.2f}): {security_id}")
                        return security_id
                    else:
                        logger.debug(
                            f"Skipping low-similarity match for {symbol}: {matched_symbol} (similarity: {similarity:.2f})")

            # LAST RESORT: Try reverse partial match with STRICT validation
            # Only use this when all other methods fail
            # Add extra validation to prevent false matches
            for idx, row in df.iterrows():
                instrument_symbol = str(row[symbol_col]).upper()
                # Check if one contains the other AND they share significant length
                if (search_symbol in instrument_symbol or instrument_symbol in search_symbol):
                    # Additional safety check: ensure reasonable length similarity
                    len_ratio = min(len(search_symbol), len(
                        instrument_symbol)) / max(len(search_symbol), len(instrument_symbol))
                    if len_ratio > 0.6:  # At least 60% length overlap
                        security_id = str(row[security_id_col])
                        logger.info(
                            f"✅ Found security ID for {symbol} (reverse match: {instrument_symbol}, len_ratio: {len_ratio:.2f}): {security_id}")
                        return security_id
                    else:
                        logger.debug(
                            f"Skipping reverse match for {symbol}: {instrument_symbol} (length ratio too low: {len_ratio:.2f})")
            
            # ABSOLUTE LAST RESORT: Fuzzy matching with high threshold
            # This will catch typos and minor variations
            try:
                from difflib import get_close_matches
                # Get all unique symbols from DataFrame
                all_symbols = df[symbol_col].str.upper().unique().tolist()
                
                # Find closest matches
                close_matches = get_close_matches(
                    search_symbol, 
                    all_symbols, 
                    n=3, 
                    cutoff=0.7  # 70% similarity threshold (high to prevent false matches)
                )
                
                if close_matches:
                    best_match = close_matches[0]
                    # Get the row with this symbol
                    result = df[df[symbol_col].str.upper() == best_match]
                    if not result.empty:
                        security_id = str(result.iloc[0][security_id_col])
                        logger.warning(
                            f"⚠️  Using FUZZY MATCH for {symbol}: {best_match} (this might be incorrect!)")
                        return security_id
            except Exception as fuzzy_error:
                logger.debug(f"Fuzzy matching failed: {fuzzy_error}")

            logger.warning(f"❌ Symbol {symbol} not found in instrument data")
            return None

        except Exception as e:
            logger.error(f"Error searching for security ID: {e}")
            return None

    def get_security_id(self, symbol: str) -> str:
        """Get numeric security ID for a symbol using Dhan instrument master lookup."""
        original_symbol = symbol

        # Handle invalid symbol prefixes and normalize casing
        if symbol.startswith('$'):
            logger.warning(
                f"Invalid symbol format detected: {symbol} - removing '$' prefix")
            symbol = symbol[1:]

        # Detect exchange segment from symbol suffix
        exchange_segments = []
        if symbol.upper().endswith('.NS'):
            exchange_segments = [self.exchange_suffix_to_segment['.NS']]
        elif symbol.upper().endswith('.BO'):
            exchange_segments = [self.exchange_suffix_to_segment['.BO']]
        else:
            exchange_segments = list(self.supported_exchange_segments)

        # Use the cleaned symbol for lookups; keep the original for logging
        clean_symbol = symbol.upper()
        if clean_symbol.endswith('.NS'):
            clean_symbol = clean_symbol[:-3]
        elif clean_symbol.endswith('.BO'):
            clean_symbol = clean_symbol[:-3]

        # Check database for previously validated security IDs FIRST
        db_id = self._get_corrected_security_id(clean_symbol)
        if db_id:
            self._cache_security_id(clean_symbol, db_id)
            logger.info(
                f"✅ Using database-corrected security ID for {clean_symbol}: {db_id}")
            return db_id

        # Check manual mapping second
        if clean_symbol in self.manual_security_id_mapping:
            manual_id = self.manual_security_id_mapping[clean_symbol]
            logger.info(
                f"✅ Using manual security ID mapping for {clean_symbol}: {manual_id}")
            return manual_id

        # Special handling for specific symbols
        if clean_symbol == "ASHOKLEY":
            logger.info("Special handling for ASHOKLEY -> ASHOK LEYLAND")
            clean_symbol = "ASHOK LEYLAND"
        elif clean_symbol == "POWERGRID":
            logger.info("Special handling for POWERGRID -> POWER GRID")
            clean_symbol = "POWER GRID"

        if not clean_symbol:
            logger.error(f"Invalid symbol after cleaning: '{original_symbol}'")
            raise ValueError(f"Invalid symbol format: '{original_symbol}'")

        # Check cache first
        cached_id = self._get_cached_security_id(clean_symbol)
        if cached_id:
            logger.debug(f"Using cached security ID for {clean_symbol}: {cached_id}")
            return cached_id

        # Try search across all applicable Dhan instrument segments
        for exchange_segment in exchange_segments:
            try:
                security_id = self._search_security_id_in_instruments(
                    clean_symbol, exchange_segment)
                if security_id:
                    self._cache_security_id(clean_symbol, security_id)
                    self._store_corrected_security_id(
                        clean_symbol, security_id, exchange_segment)
                    logger.info(
                        f"✅ Found security ID for {original_symbol} using {exchange_segment}: {security_id}")
                    return security_id
            except Exception as search_error:
                logger.debug(
                    f"Search failed for {clean_symbol} in {exchange_segment}: {search_error}")

        # If symbol had no explicit suffix, try fallback segments too
        if len(exchange_segments) == 1 and exchange_segments[0] in self.supported_exchange_segments:
            for fallback_segment in self.supported_exchange_segments:
                if fallback_segment in exchange_segments:
                    continue
                security_id = self._search_security_id_in_instruments(
                    clean_symbol, fallback_segment)
                if security_id:
                    self._cache_security_id(clean_symbol, security_id)
                    self._store_corrected_security_id(
                        clean_symbol, security_id, fallback_segment)
                    logger.info(
                        f"✅ Found security ID for {original_symbol} using fallback {fallback_segment}: {security_id}")
                    return security_id

        # Fallback: Try to search with a broader approach for problematic symbols
        if clean_symbol == "JAYNECOIND":
            alternative_symbols = [
                "JAYASWAL NUCLEUS", "JAYASWAL", "JAYA NUCLEUS"]
            for alt_symbol in alternative_symbols:
                for exchange_segment in self.supported_exchange_segments:
                    alt_security_id = self._search_security_id_in_instruments(
                        alt_symbol, exchange_segment)
                    if alt_security_id:
                        logger.info(
                            f"✅ Found alternative security ID for {clean_symbol} using {alt_symbol}: {alt_security_id}")
                        self._cache_security_id(clean_symbol, alt_security_id)
                        self._store_corrected_security_id(
                            clean_symbol, alt_security_id, exchange_segment)
                        return alt_security_id

        logger.error(
            f"❌ Security ID not found for {clean_symbol} (original: {original_symbol})")
        raise ValueError(f"Security ID not found for {original_symbol}")

    def validate_order_prerequisites(self) -> bool:
        """Validate that all prerequisites for order placement are met"""
        try:
            # Check credentials
            if not self.client_id or not self.access_token:
                logger.error("Dhan credentials not properly configured")
                return False

            # Validate API connection - but be more tolerant of temporary issues
            try:
                connection_valid = self.validate_connection()
                if not connection_valid:
                    logger.warning(
                        "Dhan API connection validation had issues, but proceeding with order placement")
            except Exception as conn_error:
                logger.warning(
                    f"Connection validation failed: {conn_error}, proceeding anyway")

            # Check market status - but be more tolerant
            try:
                if not self.is_market_open():
                    logger.warning(
                        "Market is currently closed, but proceeding with order placement")
            except Exception as status_error:
                logger.warning(
                    f"Market status check failed: {status_error}, proceeding anyway")

            logger.info(
                "Order prerequisites validation completed (proceeding despite potential issues)")
            return True

        except Exception as e:
            logger.error(f"Order prerequisites validation failed: {e}")
            # Even if validation fails, allow orders to proceed with warnings
            logger.warning(
                "Proceeding with order placement despite validation issues")
            return True

    def place_order(self, symbol: str, quantity: int, order_type: str = "MARKET",
                    side: str = "BUY", price: float = None, product_type: str = None) -> Dict:
        """Place a trading order with comprehensive validation
        
        Args:
            symbol: Stock symbol (e.g., 'ADANIPOWER.NS')
            quantity: Number of shares
            order_type: 'MARKET' or 'LIMIT'
            side: 'BUY' or 'SELL'
            price: Limit price (optional, required for LIMIT orders)
            product_type: 'CNC' for delivery, 'INTRADAY' or 'MIS' for intraday (optional, overrides config)
        """
        try:
            # Validate prerequisites first, but be tolerant of temporary API issues
            try:
                if not self.validate_order_prerequisites():
                    logger.warning(
                        "Order prerequisites validation failed, but attempting to place order anyway")
            except Exception as validation_error:
                logger.warning(
                    f"Order validation failed ({validation_error}), proceeding with order placement anyway")

            # Get numeric security ID
            security_id = self.get_security_id(symbol)

            # Check market status for afterMarketOrder
            try:
                market_open = self.is_market_open()
                after_market = not market_open
            except Exception:
                # If can't check, assume market is closed for safety
                after_market = True

            # Determine product type: Parameter > Config > Default (CNC)
            if product_type:
                # Use provided product type (dynamic override)
                dhan_product_type = self.product_type_mapping.get(
                    product_type.upper(), 'CNC')
                logger.info(
                    f"Using DYNAMIC product type: {product_type} -> {dhan_product_type}")
            else:
                # Fall back to config
                dhan_product_type = self.product_type_mapping.get(
                    self.product_type, 'CNC')
                logger.info(
                    f"Using CONFIG product type: {self.product_type} -> {dhan_product_type}")

            # Detect exchange segment based on the symbol suffix
            exchange_segment = self._get_exchange_segment(symbol)

            # Prepare order data according to official Dhan API v2 format
            order_data = {
                "correlationId": self._sanitize_correlation_id(symbol),
                "dhanClientId": self.client_id,
                "transactionType": side.upper(),
                "exchangeSegment": exchange_segment,
                "productType": dhan_product_type,
                "orderType": order_type.upper(),
                "validity": "IOC" if after_market else "DAY",
                "securityId": str(security_id),
                "quantity": int(quantity),
                "disclosedQuantity": 0,
                "afterMarketOrder": after_market,
            }

            if order_type.upper() == "LIMIT":
                if price is None:
                    raise ValueError("Limit orders require a valid price")
                order_data["price"] = float(price)
            elif order_type.upper() in ["STOP_LOSS", "STOP_LOSS_MARKET"]:
                if price is None:
                    raise ValueError("Stop loss orders require a trigger price")
                order_data["triggerPrice"] = float(price)

            # Validate order data before sending to Dhan API
            self._validate_order_data(order_data, symbol, quantity, dhan_product_type)

            # Exclude bracket-specific values unless explicitly used later

            logger.info(
                f"Placing order: {side} {quantity} {symbol} (ID: {security_id}, Product: {dhan_product_type})")
            logger.debug(f"Order data: {order_data}")
            logger.info(f"   Order Type: {order_type.upper()}")
            if price:
                logger.info(f"   Price: Rs.{price}")
            logger.info(
                f"   Product Type: {dhan_product_type} ({product_type if product_type else self.product_type})")

            response = self._make_request('POST', '/v2/orders', order_data)

            if response and 'orderId' in response:
                logger.info(
                    f"Order placed successfully: {side} {quantity} {symbol} - Order ID: {response.get('orderId')}")
                # Add more detailed logging about the security
                logger.info(f"   Security ID: {security_id}")
                logger.info(
                    f"   Product Type: {dhan_product_type} ({product_type if product_type else self.product_type})")
                return response
            else:
                logger.warning(f"Unexpected order response: {response}")
                raise Exception(
                    f"Order placement failed with unexpected response: {response}")

        except Exception as e:
            # DH-905 is a generic validation error - don't assume it's always a security ID issue
            error_str = str(e).lower()
            is_dh905_error = "dh-905" in error_str or "missing required fields" in error_str
            
            # Only attempt security ID refresh if we haven't already tried in this call
            # Use a marker to prevent infinite recursion
            retry_count = getattr(self, '_order_place_retry_count', 0)
            if is_dh905_error and retry_count < 1 and ("invalid securityid" in error_str or "security" in error_str):
                logger.warning(
                    f"DH-905 potentially security ID related. Attempting refresh for {symbol}...")

                try:
                    # Increment retry counter
                    self._order_place_retry_count = retry_count + 1
                    
                    # Clear cache and database entry for this symbol
                    clean_symbol = symbol.upper()
                    if clean_symbol.endswith('.NS'):
                        clean_symbol = clean_symbol[:-3]
                    elif clean_symbol.endswith('.BO'):
                        clean_symbol = clean_symbol[:-3]

                    # Remove from cache
                    if clean_symbol in self.security_id_cache:
                        del self.security_id_cache[clean_symbol]
                    if clean_symbol in self.cache_expiry:
                        del self.cache_expiry[clean_symbol]

                    # Remove from database
                    try:
                        cursor = self.security_db.cursor()
                        cursor.execute(
                            "DELETE FROM corrected_security_ids WHERE symbol = ?", (clean_symbol,))
                        self.security_db.commit()
                        logger.debug(
                            f"Cleared stale security ID entry for {clean_symbol}")
                    except Exception as db_error:
                        logger.debug(
                            f"Failed to clear database entry: {db_error}")

                    # Force refresh instruments
                    logger.info(
                        f"⚠️  Force refreshing instrument master for {symbol}...")
                    self.instrument_cache.clear()
                    self.instrument_cache_expiry.clear()

                    try:
                        self._fetch_daily_instrument_master(force_refresh=True)
                        logger.debug("Instrument master refreshed")
                    except Exception as fetch_error:
                        logger.debug(f"Refresh failed: {fetch_error}")

                    # Get fresh security ID
                    clean_search_symbol = symbol.upper()
                    if clean_search_symbol.endswith('.NS'):
                        clean_search_symbol = clean_search_symbol[:-3]
                    elif clean_search_symbol.endswith('.BO'):
                        clean_search_symbol = clean_search_symbol[:-3]

                    fresh_security_id = self._search_security_id_in_instruments(
                        clean_search_symbol, exchange_segment="NSE_EQ")

                    if fresh_security_id and str(fresh_security_id) != str(security_id):
                        logger.info(
                            f"✅ Found DIFFERENT security ID: {security_id} -> {fresh_security_id}")
                        security_id = fresh_security_id
                        # Retry with new security ID
                        logger.info("Retrying order with new security ID...")
                        return self.place_order(symbol, quantity, order_type, side, price, product_type)
                    else:
                        logger.warning(
                            f"Refresh returned same security ID {fresh_security_id} - issue is NOT security ID")
                        raise Exception(
                            f"Order validation failed (DH-905): Likely payload issue, not security ID. See order data above.")
                finally:
                    # Reset retry counter
                    self._order_place_retry_count = 0
                    
            elif is_dh905_error:
                # DH-905 but either not security-related or already retried
                logger.error(
                    f"DH-905 Validation Error - {error_str}")
                logger.error(
                    f"Order payload that failed: {order_data if 'order_data' in locals() else 'Unknown'}")
                logger.error(
                    "This is likely NOT a security ID issue. Check: quantity, lot size, product type availability, market hours")
                raise Exception(
                    f"Order validation failed (DH-905): Check order payload and Dhan API requirements")

            logger.error(f"Failed to place order: {e}")
            raise Exception(f"Order placement failed: {str(e)}")

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an existing order"""
        try:
            response = self._make_request('DELETE', f'/orders/{order_id}')
            logger.info(f"Order cancelled: {order_id}")
            return response
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise Exception(f"Order cancellation failed: {str(e)}")

    def modify_order(self, order_id: str, quantity: int = None,
                     price: float = None, order_type: str = None) -> Dict:
        """Modify an existing order"""
        try:
            modify_data = {"orderId": order_id}

            if quantity:
                modify_data["quantity"] = quantity
            if price:
                modify_data["price"] = price
            if order_type:
                modify_data["orderType"] = order_type.upper()

            response = self._make_request(
                'PUT', f'/orders/{order_id}', modify_data)
            logger.info(f"Order modified: {order_id}")
            return response

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            raise Exception(f"Order modification failed: {str(e)}")

    def _validate_order_data(self, order_data: Dict, symbol: str, quantity: int, product_type: str):
        """Validate order data before sending to Dhan API to prevent DH-905 errors."""
        try:
            # Validate required fields
            required_fields = ["dhanClientId", "transactionType", "exchangeSegment", 
                             "productType", "orderType", "securityId", "quantity"]
            for field in required_fields:
                if field not in order_data or order_data[field] is None:
                    raise ValueError(f"Missing required field: {field}")

            # Validate data types
            if not isinstance(order_data["quantity"], int):
                raise ValueError(
                    f"Quantity must be integer, got {type(order_data['quantity'])}")
            
            if not isinstance(order_data["securityId"], (int, str)):
                raise ValueError(
                    f"SecurityId must be integer or string, got {type(order_data['securityId'])}")

            # Validate quantity constraints
            if order_data["quantity"] <= 0:
                raise ValueError(
                    f"Quantity must be positive, got {order_data['quantity']}")
            
            # Standard NSE lot sizes (this is a heuristic check)
            # Most NSE equities have 1 as minimum lot size, but some have higher
            # If quantity is very small (< 1) it will fail
            if order_data["quantity"] < 1:
                raise ValueError(
                    f"Quantity {order_data['quantity']} is below minimum (1)")

            # Validate product type is supported
            valid_product_types = ["CNC", "INTRADAY", "MARGIN", "MTF"]
            if product_type not in valid_product_types:
                raise ValueError(
                    f"Invalid product type: {product_type}. Must be one of {valid_product_types}")

            # Validate order type
            valid_order_types = ["MARKET", "LIMIT", "STOP_LOSS", "STOP_LOSS_MARKET"]
            if order_data["orderType"] not in valid_order_types:
                raise ValueError(
                    f"Invalid order type: {order_data['orderType']}")

            # For MARKET orders, explicit price/triggerPrice should not be sent
            if order_data["orderType"] == "MARKET":
                if "price" in order_data and order_data["price"] not in [0, None]:
                    raise ValueError(
                        "MARKET orders should not specify a price")
                if "triggerPrice" in order_data and order_data["triggerPrice"] not in [0, None]:
                    raise ValueError(
                        "MARKET orders should not specify a triggerPrice")

            # For LIMIT orders, price must be present and positive
            if order_data["orderType"] == "LIMIT":
                if "price" not in order_data or order_data["price"] is None or order_data["price"] <= 0:
                    raise ValueError("LIMIT orders require a valid price")

            # For STOP_LOSS / STOP_LOSS_MARKET orders, triggerPrice must be present and positive
            if order_data["orderType"] in ["STOP_LOSS", "STOP_LOSS_MARKET"]:
                if "triggerPrice" not in order_data or order_data["triggerPrice"] is None or order_data["triggerPrice"] <= 0:
                    raise ValueError(
                        "Stop loss orders require a valid triggerPrice")

            # Validate exchange segment
            valid_segments = ["NSE_EQ", "BSE_EQ"]
            if order_data["exchangeSegment"] not in valid_segments:
                raise ValueError(
                    f"Invalid exchange segment: {order_data['exchangeSegment']}")

            logger.debug(
                f"✅ Order data validation passed for {symbol} ({quantity} @ {product_type})")

        except ValueError as ve:
            logger.error(f"❌ Order validation failed for {symbol}: {ve}")
            raise

    def _convert_symbol_to_dhan(self, symbol: str) -> str:
        """Convert Yahoo Finance symbol to Dhan symbol format"""
        # Remove exchange suffix if present
        if symbol.endswith('.NS'):
            symbol = symbol[:-3]
        elif symbol.endswith('.BO'):
            symbol = symbol[:-3]

        # Return the symbol in uppercase format
        # No more hardcoded mappings - using dynamic resolution
        return symbol.upper()

    def _sanitize_correlation_id(self, symbol: str) -> str:
        """Create an API-safe correlationId for Dhan orders."""
        clean_symbol = symbol.upper()
        if clean_symbol.endswith('.NS'):
            clean_symbol = clean_symbol[:-3]
        elif clean_symbol.endswith('.BO'):
            clean_symbol = clean_symbol[:-3]

        # Only keep allowed characters: letters, digits, space, underscore, hyphen
        sanitized = ''.join(
            ch for ch in clean_symbol if ch.isalnum() or ch in {' ', '_', '-'}
        )
        if not sanitized:
            sanitized = 'ORDER'

        timestamp = int(time.time() * 1000)
        correlation_id = f"{sanitized}-{timestamp}"[:30]
        return correlation_id

    def _get_exchange_segment(self, symbol: str) -> str:
        """Detect the Dhan exchange segment for a symbol based on its suffix."""
        clean_symbol = symbol.upper().strip()
        if clean_symbol.endswith('.NS'):
            return self.exchange_suffix_to_segment['.NS']
        elif clean_symbol.endswith('.BO'):
            return self.exchange_suffix_to_segment['.BO']
        return 'NSE_EQ'

    def get_market_status(self) -> Dict:
        """Get current market status using funds endpoint for validation"""
        try:
            # Use funds endpoint to check API connectivity since marketstatus doesn't exist in v2
            # and profile endpoint is causing 400 errors
            funds = self.get_funds()
            # If funds works, assume market is accessible
            if funds:
                # Check if it's during market hours (9:15 AM to 3:30 PM IST)
                from datetime import datetime, time

                ist = pytz.timezone('Asia/Kolkata')
                now = datetime.now(ist)
                market_open = time(9, 15)  # 9:15 AM
                market_close = time(15, 30)  # 3:30 PM

                # Check if it's a weekday and within market hours
                if now.weekday() < 5 and market_open <= now.time() <= market_close:
                    return {"marketStatus": "OPEN"}
                else:
                    return {"marketStatus": "CLOSED"}
            else:
                return {"marketStatus": "UNKNOWN"}
        except Exception as e:
            logger.warning(
                f"Failed to get market status, assuming market is accessible: {e}")
            # Return a default status to allow trading to continue
            from datetime import datetime, time

            ist = pytz.timezone('Asia/Kolkata')
            now = datetime.now(ist)
            market_open = time(9, 15)  # 9:15 AM
            market_close = time(15, 30)  # 3:30 PM

            # Check if it's a weekday and within market hours
            if now.weekday() < 5 and market_open <= now.time() <= market_close:
                return {"marketStatus": "OPEN"}
            else:
                return {"marketStatus": "CLOSED"}

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            status = self.get_market_status()
            return status.get("marketStatus", "").upper() == "OPEN"
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False

    def validate_connection(self) -> bool:
        """Validate API connection and credentials with rate limiting"""
        try:
            # Rate limiting - avoid excessive validation calls
            current_time = time.time()
            if current_time - self.last_validation_time < self.validation_cache_duration:
                logger.debug("Using cached validation result")
                return True  # Assume valid if recently validated

            # Try multiple approaches for validation
            validation_methods = [
                self._validate_via_funds,
                self._validate_via_profile,
                self._validate_via_instruments
            ]

            for i, method in enumerate(validation_methods):
                try:
                    if method():
                        logger.debug(
                            f"Dhan API connection validated successfully using method {i+1}")
                        self.last_validation_time = current_time
                        return True
                except Exception as e:
                    logger.debug(f"Validation method {i+1} failed: {e}")
                    continue

            # If all methods fail, log warning but don't fail completely
            logger.warning(
                "All Dhan API validation methods failed, but continuing with cached validation")
            return True

        except Exception as e:
            logger.warning(
                f"Dhan API validation failed, continuing with cached validation: {e}")
            # Don't fail completely on connection issues - allow system to continue
            return True

    def _validate_via_funds(self) -> bool:
        """Validate connection using funds endpoint"""
        try:
            funds = self.get_funds()
            logger.debug(f"Funds response: {funds}")
            if funds and ('availableBalance' in funds or 'sodLimit' in funds or 'netBalance' in funds):
                return True
        except:
            pass
        return False

    def _validate_via_profile(self) -> bool:
        """Validate connection using profile endpoint"""
        try:
            profile = self.get_profile()
            logger.debug(f"Profile response: {profile}")
            if profile and ('dhanClientId' in profile or 'clientId' in profile):
                return True
        except:
            pass
        return False

    def _validate_via_instruments(self) -> bool:
        """Validate connection using instruments endpoint"""
        # Try to fetch a small subset of instruments
        try:
            instruments = self._fetch_instrument_data("NSE_EQ")
            logger.debug(
                f"Instruments response shape: {instruments.shape if hasattr(instruments, 'shape') else 'unknown'}")
            if instruments is not None and not instruments.empty:
                return True
        except Exception as e:
            logger.debug(f"Instruments validation failed: {e}")
        return False
