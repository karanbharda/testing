#!/usr/bin/env python3
"""
Advanced Resilient Data Scraping Pipeline
===========================================

Production-grade web scraping with:
- Retry logic and exponential backoff
- Proxy rotation
- Rate limiting and throttling
- Response validation and parsing
- Error recovery and logging
- Multiple data source support
- Async/concurrent fetching
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import time
import random
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import feedparser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    """Result from scraping operation"""
    success: bool
    url: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    status_code: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    latency_ms: float = 0.0


@dataclass
class SourceConfig:
    """Configuration for data source"""
    name: str
    base_url: str
    endpoints: Dict[str, str]  # Endpoint name -> URL pattern
    headers: Optional[Dict[str, str]] = None
    auth: Optional[Dict[str, str]] = None
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 10
    retry_count: int = 3
    enabled: bool = True


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate_per_minute: int):
        self.rate_per_second = rate_per_minute / 60.0
        self.tokens = rate_per_minute
        self.last_update = time.time()
    
    async def acquire(self):
        """Acquire one token (wait if necessary)"""
        while self.tokens < 1:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens += elapsed * self.rate_per_second
            self.last_update = now
            
            if self.tokens < 1:
                await asyncio.sleep(0.1)
        
        self.tokens -= 1
    
    def available(self) -> int:
        """Check available tokens"""
        now = time.time()
        elapsed = now - self.last_update
        available = int(self.tokens + elapsed * self.rate_per_second)
        return available


class ResilienceMetrics:
    """Track scraping resilience metrics"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.retried_requests = 0
        self.timeout_errors = 0
        self.parse_errors = 0
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[datetime] = None
        self.error_history = deque(maxlen=100)
    
    def record_success(self):
        self.total_requests += 1
        self.successful_requests += 1
    
    def record_failure(self, error: str):
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error
        self.last_error_time = datetime.now()
        self.error_history.append(error)
    
    def record_retry(self):
        self.retried_requests += 1
    
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'retried_requests': self.retried_requests,
            'success_rate': self.success_rate(),
            'timeout_errors': self.timeout_errors,
            'parse_errors': self.parse_errors,
            'last_error': self.last_error,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None
        }


class AdvancedScrapingPipeline:
    """
    Advanced resilient web scraping pipeline
    """
    
    def __init__(self, max_concurrent: int = 5, cache_ttl_minutes: int = 60):
        """
        Initialize scraping pipeline
        
        Args:
            max_concurrent: Max concurrent requests
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.max_concurrent = max_concurrent
        self.cache_ttl_minutes = cache_ttl_minutes
        
        # Source management
        self.sources: Dict[str, SourceConfig] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.metrics: Dict[str, ResilienceMetrics] = {}
        
        # Response caching
        self.response_cache: Dict[str, Tuple[Dict, datetime]] = {}
        
        # Request queue
        self.pending_requests = deque()
        
        # User agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15",
            "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15"
        ]
        
        # Proxy list (can be updated)
        self.proxies: List[str] = []
        self.current_proxy_idx = 0
        
        logger.info(f"Advanced Scraping Pipeline initialized (max_concurrent={max_concurrent})")
    
    def register_source(self, config: SourceConfig):
        """Register a data source"""
        self.sources[config.name] = config
        self.rate_limiters[config.name] = RateLimiter(config.rate_limit_per_minute)
        self.metrics[config.name] = ResilienceMetrics()
        logger.info(f"Registered source: {config.name}")
    
    def add_proxies(self, proxy_list: List[str]):
        """Add proxy servers for rotation"""
        self.proxies = proxy_list
        logger.info(f"Added {len(proxy_list)} proxies")
    
    def _get_random_user_agent(self) -> str:
        """Get random user agent"""
        return random.choice(self.user_agents)
    
    def _get_next_proxy(self) -> Optional[str]:
        """Get next proxy in rotation"""
        if not self.proxies:
            return None
        
        proxy = self.proxies[self.current_proxy_idx]
        self.current_proxy_idx = (self.current_proxy_idx + 1) % len(self.proxies)
        return proxy
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cached_response(self, url: str) -> Optional[Dict]:
        """Get cached response if valid"""
        cache_key = self._get_cache_key(url)
        
        if cache_key in self.response_cache:
            cached_data, cached_time = self.response_cache[cache_key]
            
            if datetime.now() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                logger.debug(f"Using cached response for {url}")
                return cached_data
            else:
                # Cache expired
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, url: str, data: Dict):
        """Cache response data"""
        cache_key = self._get_cache_key(url)
        self.response_cache[cache_key] = (data, datetime.now())
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _fetch_with_retry(self, url: str, session: aiohttp.ClientSession, 
                               config: SourceConfig) -> ScrapingResult:
        """Fetch URL with exponential backoff retry"""
        start_time = time.time()
        retry_count = 0
        
        try:
            # Check cache first
            cached = self._get_cached_response(url)
            if cached:
                return ScrapingResult(
                    success=True,
                    url=url,
                    data=cached,
                    latency_ms=0.0
                )
            
            # Prepare headers
            headers = config.headers or {}
            headers['User-Agent'] = self._get_random_user_agent()
            
            # Get proxy
            proxy = self._get_next_proxy()
            
            # Make request
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout_seconds),
                proxy=proxy,
                allow_redirects=True,
                ssl=False
            ) as response:
                latency_ms = (time.time() - start_time) * 1000
                
                # Check status
                if response.status == 200:
                    # Parse response
                    content_type = response.headers.get('content-type', '')
                    
                    if 'json' in content_type:
                        data = await response.json()
                    elif 'xml' in content_type or 'rss' in content_type:
                        text = await response.text()
                        data = {'raw': text, 'parsed_at': datetime.now().isoformat()}
                    else:
                        text = await response.text()
                        data = {'html': text, 'parsed_at': datetime.now().isoformat()}
                    
                    # Cache response
                    self._cache_response(url, data)
                    
                    # Record metrics
                    self.metrics[config.name].record_success()
                    
                    logger.info(f"Successfully fetched {url} ({latency_ms:.0f}ms)")
                    
                    return ScrapingResult(
                        success=True,
                        url=url,
                        data=data,
                        status_code=response.status,
                        latency_ms=latency_ms,
                        retry_count=retry_count
                    )
                
                else:
                    error = f"HTTP {response.status}"
                    self.metrics[config.name].record_failure(error)
                    
                    logger.warning(f"HTTP {response.status} for {url}")
                    
                    return ScrapingResult(
                        success=False,
                        url=url,
                        error=error,
                        status_code=response.status,
                        latency_ms=latency_ms,
                        retry_count=retry_count
                    )
        
        except asyncio.TimeoutError as e:
            self.metrics[config.name].timeout_errors += 1
            self.metrics[config.name].record_failure("Timeout")
            logger.error(f"Timeout fetching {url}")
            raise
        
        except Exception as e:
            self.metrics[config.name].record_failure(str(e))
            self.metrics[config.name].record_retry()
            logger.error(f"Error fetching {url}: {e}")
            raise
    
    async def fetch_from_source(self, source_name: str, endpoint: str, 
                               params: Optional[Dict] = None) -> ScrapingResult:
        """
        Fetch data from registered source
        
        Args:
            source_name: Name of registered source
            endpoint: Endpoint identifier
            params: URL parameters
        
        Returns:
            Scraping result
        """
        if source_name not in self.sources:
            return ScrapingResult(
                success=False,
                url="",
                error=f"Source {source_name} not registered"
            )
        
        config = self.sources[source_name]
        
        if not config.enabled:
            return ScrapingResult(
                success=False,
                url="",
                error=f"Source {source_name} is disabled"
            )
        
        # Construct URL
        if endpoint not in config.endpoints:
            return ScrapingResult(
                success=False,
                url="",
                error=f"Endpoint {endpoint} not found in {source_name}"
            )
        
        url = config.endpoints[endpoint]
        if params:
            param_str = '&'.join(f"{k}={v}" for k, v in params.items())
            url = f"{url}{'&' if '?' in url else '?'}{param_str}"
        
        # Rate limiting
        await self.rate_limiters[source_name].acquire()
        
        # Fetch with retry
        async with aiohttp.ClientSession() as session:
            return await self._fetch_with_retry(url, session, config)
    
    async def fetch_multiple(self, requests_list: List[Tuple[str, str, Optional[Dict]]]) -> List[ScrapingResult]:
        """
        Fetch from multiple sources concurrently
        
        Args:
            requests_list: List of (source_name, endpoint, params)
        
        Returns:
            List of scraping results
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_fetch(source_name, endpoint, params):
            async with semaphore:
                return await self.fetch_from_source(source_name, endpoint, params)
        
        tasks = [bounded_fetch(src, ep, params) for src, ep, params in requests_list]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return results
    
    def parse_rss_feed(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse RSS feed content"""
        try:
            feed = feedparser.parse(xml_content)
            
            articles = []
            for entry in feed.entries[:20]:  # Limit to 20 entries
                article = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': feed.feed.get('title', 'Unknown'),
                    'author': entry.get('author', '')
                }
                articles.append(article)
            
            logger.info(f"Parsed {len(articles)} articles from RSS feed")
            return articles
        
        except Exception as e:
            logger.error(f"Error parsing RSS feed: {e}")
            self.metrics.get('rss_source', ResilienceMetrics()).parse_errors += 1
            return []
    
    def parse_html(self, html_content: str, selectors: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Parse HTML content using CSS selectors
        
        Args:
            html_content: HTML content to parse
            selectors: Dict of {field_name: css_selector}
        
        Returns:
            List of parsed items
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            items = []
            # Assume first selector is the container
            container_selector = next(iter(selectors.values()))
            containers = soup.select(container_selector)
            
            for container in containers[:20]:  # Limit to 20 items
                item = {}
                for field_name, selector in selectors.items():
                    element = container.select_one(selector)
                    item[field_name] = element.get_text(strip=True) if element else ""
                
                items.append(item)
            
            logger.info(f"Parsed {len(items)} items from HTML")
            return items
        
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return []
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics summary for all sources"""
        summary = {}
        for source_name, metrics in self.metrics.items():
            summary[source_name] = metrics.to_dict()
        
        return summary
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_urls': len(self.response_cache),
            'cache_ttl_minutes': self.cache_ttl_minutes,
            'memory_usage_kb': sum(len(str(v[0])) for v in self.response_cache.values()) / 1024
        }


# Singleton instance
_scraping_pipeline = None

def get_scraping_pipeline() -> AdvancedScrapingPipeline:
    """Get or create singleton scraping pipeline"""
    global _scraping_pipeline
    if _scraping_pipeline is None:
        _scraping_pipeline = AdvancedScrapingPipeline()
    return _scraping_pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    async def main():
        pipeline = get_scraping_pipeline()
        
        # Register sources
        newsapi_config = SourceConfig(
            name="newsapi",
            base_url="https://newsapi.org/v2",
            endpoints={
                'everything': "https://newsapi.org/v2/everything",
                'top_headlines': "https://newsapi.org/v2/top-headlines"
            },
            rate_limit_per_minute=60,
            timeout_seconds=10
        )
        
        pipeline.register_source(newsapi_config)
        
        # Fetch example
        result = await pipeline.fetch_from_source(
            "newsapi",
            "everything",
            {'q': 'stock market', 'pageSize': '5'}
        )
        
        print(f"Result: {result.success}")
        if result.success:
            print(f"Data: {result.data}")
    
    asyncio.run(main())
