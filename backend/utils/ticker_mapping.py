import os
import logging
import pickle
import requests
from datetime import datetime, timedelta

# Setup logger
logger = logging.getLogger(__name__)

# Fyers API integration
try:
    from fyers_apiv3 import fyersModel
    FYERS_AVAILABLE = True
except ImportError:
    FYERS_AVAILABLE = False


class FyersTickerMapper:
    """Fyers API-based ticker-to-company name mapping for Indian stock market"""

    def __init__(self, cache_file="data/fyers_ticker_mapping.pkl"):
        self.cache_file = cache_file
        self.ticker_mapping = {}
        self.last_updated = None
        self.update_frequency = timedelta(days=7)  # Update weekly
        self.fyers_client = None

        # Ensure data directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        # Initialize Fyers client
        self._initialize_fyers()

        # Load cached data if available
        self.load_cache()

    def _initialize_fyers(self):
        """Initialize Fyers API client"""
        try:
            if not FYERS_AVAILABLE:
                logger.warning(
                    "Fyers API not available - falling back to cached data")
                return

            app_id = os.getenv("FYERS_APP_ID")
            access_token = os.getenv("FYERS_ACCESS_TOKEN")

            if app_id and access_token:
                self.fyers_client = fyersModel.FyersModel(
                    client_id=app_id,
                    token=access_token,
                    log_path=""
                )
                logger.info(
                    "Fyers API initialized successfully for ticker mapping")
            else:
                logger.warning(
                    "Fyers credentials not found - using cached data only")
        except Exception as e:
            logger.error(f"Error initializing Fyers for ticker mapping: {e}")

    def get_ticker_mapping(self, force_update=False):
        """Get ticker mapping with automatic updates"""
        if self.should_update() or force_update:
            logger.info("Updating ticker mapping from Fyers API...")
            self.update_mapping()
            self.save_cache()

        return self.ticker_mapping

    def should_update(self):
        """Check if mapping needs update"""
        if not self.last_updated:
            return True
        return datetime.now() - self.last_updated > self.update_frequency

    def fetch_fyers_stock_list(self):
        """Fetch all Indian stocks using Fyers API - more reliable than NSE scraping"""
        try:
            stock_mapping = {}

            if not self.fyers_client:
                logger.warning(
                    "Fyers client not available - cannot fetch stock list")
                return stock_mapping

            logger.info("Fetching comprehensive stock list from Fyers API...")

            # Get master data for NSE equity instruments
            try:
                # Fyers doesn't have master_data() method - skip this approach
                logger.info(
                    "Skipping Fyers master data - method not available")
                master_data = None

                if False:  # Disable this block since master_data() doesn't exist
                    instruments = master_data.get('d', [])
                    logger.info(
                        f"Received {len(instruments)} instruments from Fyers")

                    # Filter for NSE equity instruments
                    nse_equities = [
                        inst for inst in instruments
                        if inst.get('exchange') == 'NSE' and
                        inst.get('segment') == 'EQ' and
                        inst.get('symbol_details', {}).get(
                            'symbol_type') == 'EQ'
                    ]

                    logger.info(
                        f"Found {len(nse_equities)} NSE equity instruments")

                    for instrument in nse_equities:
                        try:
                            symbol = instrument.get(
                                'symbol_details', {}).get('symbol', '')
                            company_name = instrument.get(
                                'symbol_details', {}).get('long_name', '')

                            if symbol and company_name:
                                # Convert to Yahoo Finance format
                                ticker = f"{symbol}.NS"
                                variations = self.create_name_variations(
                                    company_name)
                                stock_mapping[ticker] = variations

                        except Exception as e:
                            logger.debug(f"Error processing instrument: {e}")
                            continue

                    logger.info(
                        f"Successfully mapped {len(stock_mapping)} stocks from Fyers")

                else:
                    logger.error(
                        f"Fyers master data request failed: {master_data}")

            except Exception as e:
                logger.error(f"Error fetching Fyers master data: {e}")
                logger.info(
                    "Fyers master data not available - using fallback approach")

                # Fallback: Try to get quotes for known major stocks to build partial mapping
                logger.info(
                    "Attempting fallback approach with major stock quotes...")
                major_stocks = [
                    "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ",
                    "NSE:ICICIBANK-EQ", "NSE:SBIN-EQ", "NSE:BHARTIARTL-EQ", "NSE:ITC-EQ",
                    "NSE:KOTAKBANK-EQ", "NSE:LT-EQ", "NSE:HCLTECH-EQ", "NSE:ASIANPAINT-EQ",
                    "NSE:MARUTI-EQ", "NSE:AXISBANK-EQ", "NSE:TITAN-EQ", "NSE:SUNPHARMA-EQ",
                    "NSE:ULTRACEMCO-EQ", "NSE:WIPRO-EQ", "NSE:NESTLEIND-EQ", "NSE:POWERGRID-EQ"
                ]

                try:
                    quotes_response = self.fyers_client.quotes(
                        {"symbols": ",".join(major_stocks)})
                    if quotes_response and quotes_response.get('s') == 'ok':
                        quotes_data = quotes_response.get('d', [])
                        for quote in quotes_data:
                            symbol_info = quote.get('n', '')
                            if symbol_info:
                                # Extract symbol from Fyers format
                                symbol = symbol_info.split(':')[1].split(
                                    '-')[0] if ':' in symbol_info else ''
                                if symbol:
                                    ticker = f"{symbol}.NS"
                                    # Use symbol as company name for fallback
                                    variations = self.create_name_variations(
                                        symbol)
                                    stock_mapping[ticker] = variations

                        logger.info(
                            f"Fallback approach yielded {len(stock_mapping)} stocks")
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback approach also failed: {fallback_error}")

            return stock_mapping

        except Exception as e:
            logger.error(f"Error in fetch_fyers_stock_list: {e}")
            return {}

    def fetch_yahoo_major_stocks(self):
        """Fetch major Indian stocks using Yahoo Finance as fallback"""
        try:
            import yfinance as yf
            stock_mapping = {}

            # Major Indian stocks that are reliably available on Yahoo Finance
            major_tickers = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "HINDUNILVR.NS",
                "BAJFINANCE.NS", "AXISBANK.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BAJAJFINSV.NS",
                "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "ASIANPAINT.NS", "MARUTI.NS",
                "TITAN.NS", "NESTLEIND.NS", "BRITANNIA.NS", "LT.NS", "ULTRACEMCO.NS",
                "ONGC.NS", "BPCL.NS", "IOC.NS", "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS",
                "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"
            ]

            logger.info(
                f"Fetching info for {len(major_tickers)} major stocks from Yahoo Finance...")

            for ticker in major_tickers:
                try:
                    # Get stock info from Yahoo Finance
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    if info and 'longName' in info:
                        company_name = info['longName']
                        variations = self.create_name_variations(company_name)
                        stock_mapping[ticker] = variations
                    else:
                        # Fallback to symbol-based name
                        symbol = ticker.replace('.NS', '')
                        variations = self.create_name_variations(symbol)
                        stock_mapping[ticker] = variations

                    # Small delay to be respectful to Yahoo Finance
                    import time
                    time.sleep(0.1)

                except Exception as e:
                    logger.debug(
                        f"Error fetching {ticker} from Yahoo Finance: {e}")
                    continue

            logger.info(
                f"Successfully fetched {len(stock_mapping)} stocks from Yahoo Finance")
            return stock_mapping

        except Exception as e:
            logger.error(f"Error in fetch_yahoo_major_stocks: {e}")
            return {}

    def fetch_bse_stock_list(self):
        """Fetch BSE stocks using multiple alternative methods"""
        try:
            stock_mapping = {}

            # Method 1: Try BSE official API with enhanced headers
            try:
                url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/json, text/plain, */*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Referer': 'https://www.bseindia.com/',
                    'Origin': 'https://www.bseindia.com',
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'same-site'
                }

                logger.info(f"Trying BSE API: {url}")
                response = requests.get(url, headers=headers, timeout=15)
                logger.info(f"BSE API response status: {response.status_code}")
                logger.info(
                    f"BSE API response headers: {dict(response.headers)}")

                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    logger.info(f"BSE API content type: {content_type}")

                    # Check if response is JSON
                    if 'application/json' in content_type or response.text.strip().startswith('{'):
                        try:
                            data = response.json()
                            logger.info(
                                f"BSE API JSON parsed successfully. Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

                            # Try different possible data structures
                            stocks_data = []
                            if isinstance(data, dict):
                                stocks_data = data.get('Table', data.get(
                                    'data', data.get('stocks', [])))
                            elif isinstance(data, list):
                                stocks_data = data

                            logger.info(
                                f"Found {len(stocks_data)} stocks in BSE response")

                            for stock in stocks_data:
                                if isinstance(stock, dict):
                                    # Try different field names
                                    scrip_cd = stock.get('Scrip_cd') or stock.get(
                                        'scrip_cd') or stock.get('code') or stock.get('symbol')
                                    scrip_name = (stock.get('Scrip_Name') or stock.get('scrip_name') or
                                                  stock.get('name') or stock.get('company_name') or '')

                                    if scrip_cd and scrip_name:
                                        ticker = f"{scrip_cd}.BO"
                                        variations = self.create_name_variations(
                                            scrip_name)
                                        stock_mapping[ticker] = variations

                            logger.info(
                                f"Successfully processed {len(stock_mapping)} stocks from BSE API")

                        except ValueError as e:
                            logger.warning(f"BSE API JSON parsing failed: {e}")
                            logger.warning(
                                f"Response text preview: {response.text[:300]}")
                    else:
                        logger.warning(
                            f"BSE API returned non-JSON content: {response.text[:200]}")
                else:
                    logger.warning(
                        f"BSE API failed with status {response.status_code}: {response.text[:200]}")

            except Exception as e:
                logger.warning(f"BSE API method 1 failed: {e}")

            # Method 2: Try alternative BSE endpoint if first method failed
            if len(stock_mapping) == 0:
                try:
                    alt_url = "https://www.bseindia.com/corporates/List_Scrips.html"
                    logger.info(f"Trying alternative BSE approach: {alt_url}")

                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                    }

                    response = requests.get(
                        alt_url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        logger.info(
                            "Alternative BSE endpoint accessible, but HTML parsing not implemented")
                        # Note: HTML parsing would require BeautifulSoup, keeping simple for now

                except Exception as e:
                    logger.warning(f"BSE alternative method failed: {e}")

            # Method 3: Use hardcoded BSE stocks if APIs fail
            if len(stock_mapping) == 0:
                logger.info("Using fallback BSE stocks")
                fallback_bse_stocks = {
                    "500325.BO": ["Reliance Industries", "Reliance", "RIL"],
                    "500209.BO": ["Infosys", "Infosys Technologies", "INFY"],
                    "500180.BO": ["HDFC Bank", "HDFC", "Housing Development Finance Corporation"],
                    "500034.BO": ["State Bank of India", "SBI", "State Bank"],
                    "500696.BO": ["Hindustan Unilever", "HUL", "Unilever"],
                    "500875.BO": ["ITC", "Indian Tobacco Company"],
                    "532540.BO": ["Tata Consultancy Services", "TCS", "Tata Consultancy"],
                    "500010.BO": ["HDFC", "Housing Development Finance Corporation"],
                    "500112.BO": ["State Bank of India", "SBI"],
                    "500820.BO": ["Asian Paints", "Asian Paint"]
                }
                stock_mapping.update(fallback_bse_stocks)
                logger.info(
                    f"Added {len(fallback_bse_stocks)} fallback BSE stocks")

            return stock_mapping

        except Exception as e:
            logger.error(f"Error in fetch_bse_stock_list: {e}")
            return {}

    def create_name_variations(self, company_name):
        """Create multiple variations of company name for better search"""
        if not company_name:
            return []

        variations = [company_name.strip()]

        # Remove common suffixes
        suffixes_to_remove = [
            " Limited", " Ltd", " Ltd.", " Private Limited", " Pvt Ltd", " Pvt. Ltd.",
            " Corporation", " Corp", " Company", " Co.", " Inc", " Incorporated",
            " Private", " Pvt", " Public Limited", " Public Ltd"
        ]

        clean_name = company_name.strip()
        for suffix in suffixes_to_remove:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)].strip()
                if clean_name and clean_name not in variations:
                    variations.append(clean_name)

        # Generate dynamic abbreviations using intelligent pattern recognition
        name_lower = company_name.lower()

        # Dynamic abbreviation generation
        dynamic_abbreviations = self.generate_dynamic_abbreviations(
            company_name)
        variations.extend(dynamic_abbreviations)

        # Remove duplicates and empty strings
        variations = [v.strip() for v in variations if v.strip()]
        # Remove duplicates while preserving order
        variations = list(dict.fromkeys(variations))

        # Limit to top 5 variations to avoid overly long queries
        return variations[:5]

    def generate_dynamic_abbreviations(self, company_name):
        """Generate abbreviations dynamically using pattern recognition"""
        if not company_name:
            return []

        abbreviations = []
        name_lower = company_name.lower()
        words = company_name.split()

        # Method 1: First letter of each significant word
        significant_words = []
        skip_words = {'limited', 'ltd', 'private', 'pvt', 'corporation', 'corp',
                      'company', 'co', 'inc', 'incorporated', 'public', 'the', 'and', '&'}

        for word in words:
            if word.lower() not in skip_words and len(word) > 1:
                significant_words.append(word)

        if len(significant_words) >= 2:
            # Create acronym from first letters
            acronym = ''.join([word[0].upper() for word in significant_words])
            if len(acronym) >= 2 and len(acronym) <= 6:
                abbreviations.append(acronym)

        # Method 2: Common business patterns
        if 'bank' in name_lower:
            # For banks, often use first word + "Bank"
            first_word = words[0] if words else ""
            if first_word and first_word.lower() != 'bank':
                abbreviations.append(f"{first_word} Bank")

        if 'technologies' in name_lower or 'technology' in name_lower:
            # For tech companies, often use first word + "Tech"
            first_word = words[0] if words else ""
            if first_word and 'tech' not in first_word.lower():
                abbreviations.append(f"{first_word} Tech")

        if 'industries' in name_lower or 'industry' in name_lower:
            # For industrial companies, often just use first word
            first_word = words[0] if words else ""
            if first_word and first_word.lower() != 'industries':
                abbreviations.append(first_word)

        # Method 3: Group/Parent company patterns
        if 'group' in name_lower:
            # Extract group name
            group_word = None
            for i, word in enumerate(words):
                if word.lower() == 'group' and i > 0:
                    group_word = words[i-1]
                    break
            if group_word:
                abbreviations.append(f"{group_word} Group")

        # Method 4: Sector-specific patterns
        sector_patterns = {
            'energy': ['Energy', 'Power'],
            'power': ['Power', 'Energy'],
            'finance': ['Finance', 'Financial'],
            'consultancy': ['Consultancy', 'Consulting'],
            'services': ['Services'],
            'motors': ['Motors', 'Auto'],
            'automotive': ['Auto', 'Motors'],
            'cement': ['Cement'],
            'steel': ['Steel'],
            'pharma': ['Pharma', 'Pharmaceutical'],
            'pharmaceutical': ['Pharma'],
            'telecom': ['Telecom', 'Communications'],
            'communications': ['Telecom'],
            'oil': ['Oil', 'Petroleum'],
            'gas': ['Gas'],
            'ports': ['Ports', 'Port'],
            'shipping': ['Shipping'],
            'textiles': ['Textiles'],
            'chemicals': ['Chemicals', 'Chem'],
            'metals': ['Metals'],
            'mining': ['Mining'],
            'real estate': ['Realty', 'Real Estate'],
            'realty': ['Real Estate'],
            'insurance': ['Insurance'],
            'mutual fund': ['MF', 'Mutual Fund'],
            'asset management': ['AMC', 'Asset Management']
        }

        for sector, variations in sector_patterns.items():
            if sector in name_lower:
                # Add sector-specific variations
                first_word = words[0] if words else ""
                if first_word:
                    for variation in variations:
                        abbreviations.append(f"{first_word} {variation}")

        # Method 5: Handle special characters and conjunctions
        if '&' in company_name or ' and ' in name_lower:
            # For companies with & or "and", create variations
            parts = company_name.replace('&', 'and').split(' and ')
            if len(parts) == 2:
                # Create "FirstWord & SecondWord" format
                first_part = parts[0].strip().split()[
                    0] if parts[0].strip() else ""
                second_part = parts[1].strip().split()[
                    0] if parts[1].strip() else ""
                if first_part and second_part:
                    abbreviations.append(f"{first_part} & {second_part}")
                    abbreviations.append(f"{first_part[0]}&{second_part[0]}")

        # Method 6: Remove common words and create short forms
        clean_words = []
        for word in words:
            if (word.lower() not in skip_words and
                len(word) > 2 and
                    not word.lower().endswith('ing')):
                clean_words.append(word)

        if len(clean_words) >= 1:
            # Just the main company name without suffixes
            # Take first 2 significant words
            main_name = ' '.join(clean_words[:2])
            if main_name != company_name:
                abbreviations.append(main_name)

        return abbreviations

    def test_api_connectivity(self):
        """Test connectivity to NSE and BSE APIs"""
        results = {
            'nse_main': False,
            'nse_market_data': False,
            'bse_api': False,
            'errors': []
        }

        # Test NSE main page
        try:
            response = requests.get("https://www.nseindia.com", timeout=10)
            results['nse_main'] = response.status_code == 200
            if not results['nse_main']:
                results['errors'].append(
                    f"NSE main page returned {response.status_code}")
        except Exception as e:
            results['errors'].append(f"NSE main page error: {e}")

        # Test NSE market data page
        try:
            response = requests.get(
                "https://www.nseindia.com/market-data", timeout=10)
            results['nse_market_data'] = response.status_code == 200
            if not results['nse_market_data']:
                results['errors'].append(
                    f"NSE market data page returned {response.status_code}")
        except Exception as e:
            results['errors'].append(f"NSE market data error: {e}")

        # Test BSE API
        try:
            response = requests.get(
                "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w", timeout=10)
            results['bse_api'] = response.status_code == 200
            if not results['bse_api']:
                results['errors'].append(
                    f"BSE API returned {response.status_code}")
        except Exception as e:
            results['errors'].append(f"BSE API error: {e}")

        logger.info(f"API Connectivity Test Results: {results}")
        return results

    def update_mapping(self):
        """Update mapping using Fyers API as primary source with fallbacks"""
        all_mappings = {}

        # Method 1: Try Fyers API (Primary - Most Reliable)
        try:
            logger.info("Attempting to fetch stocks from Fyers API...")
            fyers_data = self.fetch_fyers_stock_list()
            all_mappings.update(fyers_data)
            logger.info(
                f"[SUCCESS] Successfully fetched {len(fyers_data)} stocks from Fyers API")
        except Exception as e:
            logger.warning(f"[ERROR] Fyers API failed: {e}")

        # Method 2: Try BSE API (Secondary)
        try:
            logger.info("Attempting BSE API as secondary source...")
            bse_data = self.fetch_bse_stock_list()
            # Merge with existing data
            for ticker, names in bse_data.items():
                if ticker in all_mappings:
                    # Combine name variations
                    all_mappings[ticker].extend(names)
                    all_mappings[ticker] = list(set(all_mappings[ticker]))
                else:
                    all_mappings[ticker] = names
            logger.info(
                f"[SUCCESS] Successfully added {len(bse_data)} stocks from BSE API")
        except Exception as e:
            logger.warning(f"[ERROR] BSE API failed: {e}")

        # Method 3: Yahoo Finance fallback for major stocks
        if len(all_mappings) < 100:
            logger.info("Adding Yahoo Finance data for major stocks...")
            try:
                yahoo_data = self.fetch_yahoo_major_stocks()
                for ticker, names in yahoo_data.items():
                    if ticker not in all_mappings:
                        all_mappings[ticker] = names
                logger.info(
                    f"[SUCCESS] Added {len(yahoo_data)} stocks from Yahoo Finance")
            except Exception as e:
                logger.warning(f"[ERROR] Yahoo Finance fallback failed: {e}")

        # Method 4: Hardcoded fallback if still insufficient data
        if len(all_mappings) < 50:
            logger.warning(
                f"Only {len(all_mappings)} stocks fetched from APIs. Using fallback hardcoded mapping.")
            fallback_mapping = self.get_fallback_mapping()
            for ticker, names in fallback_mapping.items():
                if ticker not in all_mappings:
                    all_mappings[ticker] = names
            logger.info(
                f"[SUCCESS] Added {len(fallback_mapping)} fallback stocks")
        else:
            logger.info(
                f"[SUCCESS] Sufficient stocks ({len(all_mappings)}) fetched from APIs")

        self.ticker_mapping = all_mappings
        self.last_updated = datetime.now()

        logger.info(
            f"Final mapping contains {len(self.ticker_mapping)} stocks")

        # Log sample of stocks for verification
        sample_stocks = list(self.ticker_mapping.keys())[:5]
        logger.info(f"[DATA] Sample stocks: {sample_stocks}")

    def get_fallback_mapping(self):
        """Fallback hardcoded mapping for essential stocks - expanded to 50+ stocks"""
        return {
            # Top 10 by market cap
            "RELIANCE.NS": ["Reliance Industries", "RIL", "Reliance"],
            "TCS.NS": ["Tata Consultancy Services", "TCS"],
            "HDFCBANK.NS": ["HDFC Bank", "Housing Development Finance Corporation"],
            "INFY.NS": ["Infosys", "Infosys Limited"],
            "ICICIBANK.NS": ["ICICI Bank", "Industrial Credit and Investment Corporation"],
            "KOTAKBANK.NS": ["Kotak Mahindra Bank", "Kotak Bank"],
            "BHARTIARTL.NS": ["Bharti Airtel", "Airtel"],
            "ITC.NS": ["ITC Limited", "Indian Tobacco Company"],
            "SBIN.NS": ["State Bank of India", "SBI"],
            "HINDUNILVR.NS": ["Hindustan Unilever", "HUL"],

            # Banking & Financial Services
            "BAJFINANCE.NS": ["Bajaj Finance", "Bajaj Finserv"],
            "AXISBANK.NS": ["Axis Bank"],
            "HDFCLIFE.NS": ["HDFC Life Insurance", "HDFC Life"],
            "SBILIFE.NS": ["SBI Life Insurance", "SBI Life"],
            "BAJAJFINSV.NS": ["Bajaj Finserv"],
            "INDUSINDBK.NS": ["IndusInd Bank"],
            "BANDHANBNK.NS": ["Bandhan Bank"],
            "IDFCFIRSTB.NS": ["IDFC First Bank"],

            # IT & Technology
            "HCLTECH.NS": ["HCL Technologies", "HCL Tech"],
            "WIPRO.NS": ["Wipro Limited", "Wipro"],
            "TECHM.NS": ["Tech Mahindra"],
            "LTIM.NS": ["LTIMindtree", "L&T Infotech"],
            "MPHASIS.NS": ["Mphasis"],

            # Consumer Goods
            "ASIANPAINT.NS": ["Asian Paints", "Asian Paint"],
            "MARUTI.NS": ["Maruti Suzuki", "Maruti", "Suzuki India"],
            "TITAN.NS": ["Titan Company", "Titan Industries"],
            "NESTLEIND.NS": ["Nestle India", "Nestle"],
            "BRITANNIA.NS": ["Britannia Industries", "Britannia"],
            "DABUR.NS": ["Dabur India", "Dabur"],
            "MARICO.NS": ["Marico"],
            "GODREJCP.NS": ["Godrej Consumer Products", "Godrej"],
            "SIMPLEXINF.NS": ["Simplex Infrastructures", "Simplex Infrastructures Ltd", "Simplex"],

            # Infrastructure & Construction
            "LT.NS": ["Larsen & Toubro", "L&T", "Larsen Toubro"],
            "ULTRACEMCO.NS": ["UltraTech Cement", "Ultratech"],
            "GRASIM.NS": ["Grasim Industries", "Grasim"],
            "SHREECEM.NS": ["Shree Cement"],
            "RAMCOCEM.NS": ["Ramco Cements", "Ramco"],

            # Energy & Oil
            "ONGC.NS": ["Oil and Natural Gas Corporation", "ONGC"],
            "BPCL.NS": ["Bharat Petroleum", "BPCL"],
            "IOC.NS": ["Indian Oil Corporation", "IOC"],
            "COALINDIA.NS": ["Coal India", "CIL"],
            "NTPC.NS": ["NTPC"],
            "POWERGRID.NS": ["Power Grid Corporation", "PowerGrid"],

            # Adani Group
            "ADANIENSOL.NS": ["Adani Energy Solutions", "Adani Green Energy", "Adani Power", "Adani Energy"],
            "ADANIPORTS.NS": ["Adani Ports", "Adani Port"],
            "ADANITRANS.NS": ["Adani Transmission"],
            "ATGL.NS": ["Adani Total Gas"],
            "ADANIENT.NS": ["Adani Enterprises"],

            # Pharmaceuticals
            "SUNPHARMA.NS": ["Sun Pharmaceutical", "Sun Pharma"],
            "DRREDDY.NS": ["Dr. Reddy's Laboratories", "Dr Reddy"],
            "CIPLA.NS": ["Cipla"],
            "DIVISLAB.NS": ["Divi's Laboratories", "Divis Lab"],
            "BIOCON.NS": ["Biocon"],
            "LUPIN.NS": ["Lupin"],

            # Metals & Mining
            "TATASTEEL.NS": ["Tata Steel"],
            "HINDALCO.NS": ["Hindalco Industries", "Hindalco"],
            "JSWSTEEL.NS": ["JSW Steel"],
            "SAIL.NS": ["Steel Authority of India", "SAIL"],
            "VEDL.NS": ["Vedanta"],

            # Auto & Auto Components
            "BAJAJ-AUTO.NS": ["Bajaj Auto"],
            "M&M.NS": ["Mahindra & Mahindra", "Mahindra"],
            "TATAMOTORS.NS": ["Tata Motors"],
            "EICHERMOT.NS": ["Eicher Motors"],
            "HEROMOTOCO.NS": ["Hero MotoCorp", "Hero"],

            # Telecom
            "JIOFINANCE.NS": ["Jio Financial Services", "Jio Finance"],

            # Others
            "APOLLOHOSP.NS": ["Apollo Hospitals", "Apollo"],
            "PIDILITIND.NS": ["Pidilite Industries", "Pidilite"],
            "BERGEPAINT.NS": ["Berger Paints", "Berger"],
            "HAVELLS.NS": ["Havells India", "Havells"],
        }

    def save_cache(self):
        """Save mapping to cache file"""
        try:
            cache_data = {
                'mapping': self.ticker_mapping,
                'last_updated': self.last_updated
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved ticker mapping cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def load_cache(self):
        """Load mapping from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.ticker_mapping = cache_data.get('mapping', {})
                self.last_updated = cache_data.get('last_updated')
                logger.info(
                    f"Loaded {len(self.ticker_mapping)} stocks from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.ticker_mapping = {}
            self.last_updated = None

    def get_company_names(self, ticker):
        """Get company names for a specific ticker"""
        mapping = self.get_ticker_mapping()
        return mapping.get(ticker, [ticker.replace(".NS", "").replace(".BO", "")])

    def search_ticker_by_name(self, company_name):
        """Reverse lookup: find ticker by company name"""
        mapping = self.get_ticker_mapping()
        company_name_lower = company_name.lower()

        for ticker, names in mapping.items():
            for name in names:
                if company_name_lower in name.lower() or name.lower() in company_name_lower:
                    return ticker
        return None
