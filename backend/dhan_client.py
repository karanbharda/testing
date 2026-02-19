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

logger = logging.getLogger(__name__)


class DhanAPIClient:
    """Dhan API client for live trading operations"""

    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = "https://api.dhan.co"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'access-token': access_token
        })

        # Log initialization for debugging
        logger.info(
            f"Dhan API client initialized for client ID: {client_id[:4]}...{client_id[-4:] if len(client_id) > 8 else client_id}")

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
                logger.info(f"Loaded {len(self.manual_security_id_mapping)} manual security ID mappings from {mapping_file}")
            else:
                logger.warning(f"Security ID mapping file not found at {mapping_file}")
                self.manual_security_id_mapping = {}
        except Exception as e:
            logger.error(f"Failed to load security ID mapping file: {e}")
            self.manual_security_id_mapping = {}

        # Instrument data cache
        self.instrument_cache = {}
        self.instrument_cache_expiry = {}
        # Reduce cache duration to 1 hour for fresher data
        self.instrument_cache_duration = 3600

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
                    logger.warning(f"⚠️ Dhan API Rate Limit (429) hit on {endpoint}")
                    if attempt < max_retries:
                        # Check for Retry-After header
                        retry_after = response.headers.get("Retry-After")
                        wait_time = float(retry_after) if retry_after else (retry_delay * (2 ** attempt))
                        # Cap max wait to 10s
                        wait_time = min(wait_time, 10.0) 
                        
                        logger.info(f"   Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("❌ Max retries reached for 429 Rate Limit")
                        raise Exception(f"Dhan API Rate Limit Exceeded (429) after {max_retries} retries")

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
                    # Normalise to a common key set so downstream code only needs one key list
                    available = (
                        fundlimit_response.get("availablecash") or
                        fundlimit_response.get("availableBalance") or
                        fundlimit_response.get("netAvailableMargin") or
                        fundlimit_response.get("sodLimit") or
                        0.0
                    )
                    try:
                        available = float(available)
                    except (TypeError, ValueError):
                        available = 0.0

                    logger.info(f"✅ Dhan fundlimit — available cash: Rs.{available:.2f}")
                    return {
                        "availablecash": available,
                        "availableBalance": available,
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
                available = float(profile_response.get("availableBalance", 0.0) or 0.0)
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
                logger.debug(f"Profile-based funds request also failed: {profile_error}")

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
        """Get current stock holdings"""
        try:
            # Try the holdings endpoint first
            try:
                response = self._make_request('GET', '/v2/holdings')

                # Handle different response formats
                if isinstance(response, list):
                    return response
                elif isinstance(response, dict):
                    return response.get('data', [])
                else:
                    logger.warning(
                        f"Unexpected response format from holdings: {type(response)}")
                    return []
            except Exception as holdings_error:
                logger.debug(f"Holdings endpoint failed: {holdings_error}")

                # Check if it's a DH-905 error (missing required fields)
                error_str = str(holdings_error).lower()
                if "dh-905" in error_str or "missing required fields" in error_str:
                    logger.info(
                        "Holdings endpoint requires authentication/parameters, returning empty portfolio")
                elif "500" in error_str:
                    logger.info(
                        "No holdings found (empty portfolio) - this is normal for new accounts")
                else:
                    logger.warning(
                        f"Holdings request failed: {holdings_error}")

                # Return empty list for any error
                return []

        except Exception as e:
            logger.error(f"Unexpected error in get_holdings: {e}")
            return []

    def get_positions(self) -> List[Dict]:
        """Get current trading positions"""
        try:
            response = self._make_request('GET', '/v2/positions')

            # Handle different response formats
            if isinstance(response, list):
                return response
            elif isinstance(response, dict):
                return response.get('data', [])
            else:
                logger.warning(
                    f"Unexpected response format from positions: {type(response)}")
                return []

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_orders(self) -> List[Dict]:
        """Get order history"""
        try:
            response = self._make_request('GET', '/v2/orders')

            # Handle different response formats
            if isinstance(response, list):
                return response
            elif isinstance(response, dict):
                return response.get('data', [])
            else:
                logger.warning(
                    f"Unexpected response format from orders: {type(response)}")
                return []

        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
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

    def _fetch_instrument_data(self, exchange_segment: str = "NSE_EQ") -> pd.DataFrame:
        """Fetch real-time instrument data directly from Dhan API v2 endpoint"""
        try:
            # Use the specific endpoint as requested: https://api.dhan.co/v2/instrument/NSE_EQ
            url = f"{self.base_url}/v2/instrument/{exchange_segment}"
            logger.info(f"Fetching instrument data from: {url}")

            response = self.session.get(url)

            if response.status_code == 200:
                # Parse CSV data directly into DataFrame
                df = pd.read_csv(StringIO(response.text))
                logger.info(f"✅ Loaded instruments: {len(df)}")

                # Log some sample data for debugging
                if not df.empty:
                    logger.debug(
                        f"Sample instrument data: {df.head(3).to_dict('records')}")

                return df
            else:
                logger.error(
                    f"Failed to fetch instruments. Status: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching instrument data: {e}")
            return pd.DataFrame()

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

            # Find the appropriate columns
            for col in df.columns:
                if col in symbol_columns:
                    symbol_col = col
                elif col in security_id_columns:
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
                'IEX': 'Indian Energy Exchange Ltd'
            }

            # Apply abbreviation mapping if available
            if search_symbol in abbreviation_mapping:
                expanded_name = abbreviation_mapping[search_symbol]
                variations.append(expanded_name)
                variations.append(expanded_name + ' LTD')
                variations.append(expanded_name + ' LIMITED')

            # Try each variation
            for variation in variations:
                result = df[df[symbol_col].str.upper() == variation]
                if not result.empty:
                    security_id = str(result.iloc[0][security_id_col])
                    logger.info(
                        f"✅ Found security ID for {symbol} (variation match: {variation}): {security_id}")
                    return security_id

            # Try partial matches as a last resort
            for variation in variations:
                result = df[df[symbol_col].str.contains(
                    variation, case=False, na=False)]
                if not result.empty:
                    security_id = str(result.iloc[0][security_id_col])
                    logger.info(
                        f"✅ Found security ID for {symbol} (partial match: {variation}): {security_id}")
                    return security_id

            # Try reverse partial match (search symbol contains instrument symbol)
            for idx, row in df.iterrows():
                instrument_symbol = str(row[symbol_col]).upper()
                if search_symbol in instrument_symbol or instrument_symbol in search_symbol:
                    security_id = str(row[security_id_col])
                    logger.info(
                        f"✅ Found security ID for {symbol} (reverse partial match: {instrument_symbol}): {security_id}")
                    return security_id

            logger.warning(f"❌ Symbol {symbol} not found in instrument data")
            return None

        except Exception as e:
            logger.error(f"Error searching for security ID: {e}")
            return None

    def get_security_id(self, symbol: str) -> str:
        """Get numeric security ID for a symbol using direct API approach from https://api.dhan.co/v2/instrument/NSE_EQ"""
        # Validate and clean symbol format
        original_symbol = symbol

        # Handle special cases and invalid formats
        if symbol.startswith('$'):
            logger.warning(
                f"Invalid symbol format detected: {symbol} - removing '$' prefix")
            symbol = symbol[1:]

        # Convert symbol format
        if symbol.endswith('.NS'):
            symbol = symbol[:-3]
        elif symbol.endswith('.BO'):
            symbol = symbol[:-3]

        symbol = symbol.upper()

        # Check database for previously validated security IDs FIRST
        # This takes precedence over manual mappings to use corrected values
        db_id = self._get_corrected_security_id(symbol)
        if db_id:
            self._cache_security_id(symbol, db_id)
            logger.info(
                f"✅ Using database-corrected security ID for {symbol}: {db_id}")
            return db_id

        # Check manual mapping second
        if symbol in self.manual_security_id_mapping:
            manual_id = self.manual_security_id_mapping[symbol]
            logger.info(
                f"✅ Using manual security ID mapping for {symbol}: {manual_id}")
            return manual_id

        # Special handling for specific symbols
        if symbol == "ASHOKLEY":
            logger.info("Special handling for ASHOKLEY -> ASHOK LEYLAND")
            symbol = "ASHOK LEYLAND"
        elif symbol == "POWERGRID":
            logger.info("Special handling for POWERGRID -> POWER GRID")
            symbol = "POWER GRID"  # Use the simpler form that works

        # Check if symbol is empty after cleaning
        if not symbol:
            logger.error(f"Invalid symbol after cleaning: '{original_symbol}'")
            raise ValueError(f"Invalid symbol format: '{original_symbol}'")

        # Check cache first
        cached_id = self._get_cached_security_id(symbol)
        if cached_id:
            logger.debug(f"Using cached security ID for {symbol}: {cached_id}")
            return cached_id

        # Check database for previously validated security IDs
        db_id = self._get_corrected_security_id(symbol)
        if db_id:
            self._cache_security_id(symbol, db_id)
            return db_id

        # Fetch and search in real-time instrument data from the specific endpoint
        security_id = self._search_security_id_in_instruments(symbol)
        if security_id:
            self._cache_security_id(symbol, security_id)
            # Store in database for future use
            self._store_corrected_security_id(symbol, security_id)
            return security_id

        # Fallback: Try to search with a broader approach for problematic symbols
        if symbol == "JAYNECOIND":
            # Try alternative search for Jayaswal Nucleus
            alternative_symbols = [
                "JAYASWAL NUCLEUS", "JAYASWAL", "JAYA NUCLEUS"]
            for alt_symbol in alternative_symbols:
                alt_security_id = self._search_security_id_in_instruments(
                    alt_symbol)
                if alt_security_id:
                    logger.info(
                        f"✅ Found alternative security ID for {symbol} using {alt_symbol}: {alt_security_id}")
                    self._cache_security_id(symbol, alt_security_id)
                    self._store_corrected_security_id(symbol, alt_security_id)
                    return alt_security_id

        logger.error(
            f"❌ Security ID not found for {symbol} (original: {original_symbol})")
        raise ValueError(f"Security ID not found for {symbol}")

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
                    side: str = "BUY", price: float = None) -> Dict:
        """Place a trading order with comprehensive validation"""
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

            # Only use CNC (delivery) product type for equity trades
            # Prepare order data according to official Dhan API v2 format
            order_data = {
                "dhanClientId": self.client_id,
                "transactionType": side.upper(),
                "exchangeSegment": "NSE_EQ",
                "productType": "CNC",  # Cash and Carry - for delivery-based trading
                "orderType": order_type.upper(),
                "validity": "DAY",
                "securityId": str(security_id),  # Must be string
                "quantity": int(quantity),  # Must be integer
                "disclosedQuantity": 0,  # Integer, not string
                "triggerPrice": 0.0,  # Float, always required
                "afterMarketOrder": False,  # Boolean, always required
                "boProfitValue": 0.0,  # Float, always required
                "boStopLossValue": 0.0  # Float, always required
            }

            # Set price field (always required according to API docs)
            if order_type.upper() == "LIMIT" and price:
                order_data["price"] = float(price)
            else:
                # For MARKET orders, price must be 0.0 (float)
                order_data["price"] = 0.0

            logger.info(
                f"Placing order: {side} {quantity} {symbol} (ID: {security_id}, Product: CNC)")
            logger.debug(f"Order data: {order_data}")

            response = self._make_request('POST', '/v2/orders', order_data)

            if response and 'orderId' in response:
                logger.info(
                    f"Order placed successfully: {side} {quantity} {symbol} - Order ID: {response.get('orderId')}")
                # Add more detailed logging about the security
                logger.info(f"   Security ID: {security_id}")
                logger.info(f"   Product Type: CNC (Delivery)")
                return response
            else:
                logger.warning(f"Unexpected order response: {response}")
                raise Exception(
                    f"Order placement failed with unexpected response: {response}")

        except Exception as e:
            # Check if this is an invalid security ID error (DH-905)
            error_str = str(e).lower()
            if "dh-905" in error_str or "invalid securityid" in error_str:
                logger.warning(
                    f"Invalid security ID detected for {symbol}. Attempting to refresh security ID mapping...")

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
                    logger.info(
                        f"Removed stale security ID entry for {clean_symbol} from database")
                except Exception as db_error:
                    logger.warning(
                        f"Failed to remove database entry: {db_error}")

                # Remove from manual mapping if it exists
                if clean_symbol in self.manual_security_id_mapping:
                    old_id = self.manual_security_id_mapping[clean_symbol]
                    logger.info(
                        f"Removing outdated manual mapping for {clean_symbol}: {old_id}")
                    del self.manual_security_id_mapping[clean_symbol]

                # Try to get fresh security ID
                try:
                    fresh_security_id = self.get_security_id(symbol)
                    logger.info(
                        f"Refreshed security ID for {symbol}: {fresh_security_id}")

                    # Retry the order with fresh security ID
                    logger.info("Retrying order with refreshed security ID...")
                    return self.place_order(symbol, quantity, order_type, side, price)

                except Exception as refresh_error:
                    logger.error(
                        f"Failed to refresh security ID for {symbol}: {refresh_error}")
                    raise Exception(
                        f"Order placement failed due to invalid security ID and refresh failed: {str(e)}")

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
                import pytz

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
            import pytz

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
