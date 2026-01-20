"""
HERD RADAR: Retail Sentiment Scraper
=====================================
Scrapes crowd sentiment from MyFxBook to identify liquidity targets.

LOGIC:
- If retail is 63% SHORT → Liquidity (stops) is ABOVE → Price likely goes UP
- If retail is 63% LONG → Liquidity (stops) is BELOW → Price likely goes DOWN
- Smart Money hunts the majority's stop losses

This is LAGGING data but useful for:
1. Session planning (where will NY go?)
2. Confirming liquidity pool locations
3. Contrarian bias for neural network
"""

import requests
import logging
import re
from typing import Dict, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup

logger = logging.getLogger("HerdRadar")

@dataclass
class RetailSentiment:
    """Retail sentiment data for a symbol."""
    symbol: str
    long_pct: float          # % of retail traders LONG
    short_pct: float         # % of retail traders SHORT
    long_avg_price: float    # Average entry price of longs
    short_avg_price: float   # Average entry price of shorts
    current_price: float     # Current market price
    liquidity_bias: str      # "BULLISH" (shorts majority) or "BEARISH" (longs majority)
    liquidity_direction: str # "ABOVE" or "BELOW" - where is the liquidity pool?

    def to_neural_dict(self) -> Dict[str, float]:
        """Returns normalized values for neural network input."""
        # Contrarian score: -1 (retail bullish) to +1 (retail bearish)
        # If retail is bearish (short), we're contrarian bullish
        contrarian_score = (self.short_pct - self.long_pct) / 100.0
        
        # Distance to retail average prices (potential targets)
        if self.current_price > 0:
            dist_to_long_stops = (self.long_avg_price - self.current_price) / self.current_price
            dist_to_short_stops = (self.short_avg_price - self.current_price) / self.current_price
        else:
            dist_to_long_stops = 0.0
            dist_to_short_stops = 0.0
        
        return {
            "retail_long_pct": self.long_pct / 100.0,
            "retail_short_pct": self.short_pct / 100.0,
            "contrarian_score": contrarian_score,  # Positive = retail bearish = we bullish
            "dist_to_long_stops": dist_to_long_stops,
            "dist_to_short_stops": dist_to_short_stops,
            "liquidity_above": 1.0 if self.liquidity_direction == "ABOVE" else 0.0,
            "liquidity_below": 1.0 if self.liquidity_direction == "BELOW" else 0.0,
        }


class SentimentScraper:
    """
    Scrapes retail sentiment from MyFxBook Community Outlook.
    Uses Supabase Edge Function as proxy to bypass 403.
    """
    
    # Symbol mapping: MT5 symbol → MyFxBook symbol name
    SYMBOL_MAP = {
        "XAUUSD": "XAUUSD",
        "GOLD": "XAUUSD",
        "EURUSD": "EURUSD",
        "GBPUSD": "GBPUSD",
        "USDJPY": "USDJPY",
        "AUDUSD": "AUDUSD",
        "USDCAD": "USDCAD",
        "USDCHF": "USDCHF",
        "NZDUSD": "NZDUSD",
        "GBPJPY": "GBPJPY",
        "EURJPY": "EURJPY",
        "BTCUSD": "BTCUSD",
        "XAGUSD": "XAGUSD",
    }
    
    # Primary: Direct scraping (may get 403)
    DIRECT_URL = "https://www.myfxbook.com/es/community/outlook"
    
    # Fallback: Supabase proxy (same pattern as ForexFactory)
    PROXY_URL = "https://hkbnbwcjbitadkdkgzco.supabase.co/functions/v1/sentiment-proxy"
    
    HEADERS = {
        "User-Agent": "FEAT-Sniper/1.0 (Sentiment Analysis)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    def __init__(self):
        self._cache: Dict[str, RetailSentiment] = {}
        self._last_fetch = None
    
    def get_sentiment(self, symbol: str) -> Optional[RetailSentiment]:
        """
        Get retail sentiment for a symbol.
        
        Args:
            symbol: MT5 symbol (e.g., "XAUUSD")
            
        Returns:
            RetailSentiment object or None if symbol not found
        """
        # Map symbol
        myfxbook_symbol = self.SYMBOL_MAP.get(symbol.upper(), symbol.upper())
        
        try:
            # Try direct scraping first
            html = self._fetch_page()
            if html:
                return self._parse_sentiment(html, myfxbook_symbol)
            
            # Fallback: Return cached or None
            return self._cache.get(myfxbook_symbol)
            
        except Exception as e:
            logger.warning(f"⚠️ HERD RADAR: Failed to fetch sentiment: {e}")
            return self._cache.get(myfxbook_symbol)
    
    def _fetch_page(self) -> Optional[str]:
        """Fetches the outlook page, trying direct then proxy."""
        # Try direct
        try:
            response = requests.get(self.DIRECT_URL, headers=self.HEADERS, timeout=10)
            if response.status_code == 200:
                return response.text
        except:
            pass
        
        # Try proxy
        try:
            response = requests.get(self.PROXY_URL, headers=self.HEADERS, timeout=15)
            if response.status_code == 200:
                return response.text
        except:
            pass
        
        return None
    
    def _parse_sentiment(self, html: str, symbol: str) -> Optional[RetailSentiment]:
        """Parses sentiment data from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find row for symbol
        row = soup.find('tr', attrs={'symbolname': symbol})
        if not row:
            # Try case-insensitive search
            rows = soup.find_all('tr', attrs={'symbolname': True})
            for r in rows:
                if r.get('symbolname', '').upper() == symbol.upper():
                    row = r
                    break
        
        if not row:
            logger.warning(f"⚠️ Symbol {symbol} not found in MyFxBook outlook")
            return None
        
        try:
            # Extract percentages from progress bars
            short_bar = row.find(class_='progress-bar-danger')
            long_bar = row.find(class_='progress-bar-success')
            
            short_pct = self._extract_percentage(short_bar)
            long_pct = self._extract_percentage(long_bar)
            
            # Extract prices
            short_price_elem = row.find(id=re.compile(r'shortPriceCell'))
            long_price_elem = row.find(id=re.compile(r'longPriceCell'))
            current_price_elem = row.find(id=re.compile(r'rateCell'))
            
            short_avg_price = self._parse_price(short_price_elem.text if short_price_elem else "0")
            long_avg_price = self._parse_price(long_price_elem.text if long_price_elem else "0")
            current_price = self._parse_price(current_price_elem.text if current_price_elem else "0")
            
            # Determine liquidity bias
            if short_pct > 55:
                liquidity_bias = "BULLISH"  # Majority short = price goes up
                liquidity_direction = "ABOVE"
            elif long_pct > 55:
                liquidity_bias = "BEARISH"  # Majority long = price goes down
                liquidity_direction = "BELOW"
            else:
                liquidity_bias = "NEUTRAL"
                liquidity_direction = "BALANCED"
            
            sentiment = RetailSentiment(
                symbol=symbol,
                long_pct=long_pct,
                short_pct=short_pct,
                long_avg_price=long_avg_price,
                short_avg_price=short_avg_price,
                current_price=current_price,
                liquidity_bias=liquidity_bias,
                liquidity_direction=liquidity_direction
            )
            
            # Cache it
            self._cache[symbol] = sentiment
            
            logger.info(f"📊 HERD RADAR: {symbol} | L:{long_pct:.0f}% S:{short_pct:.0f}% | Liquidity: {liquidity_direction}")
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error parsing sentiment for {symbol}: {e}")
            return None
    
    def _extract_percentage(self, element) -> float:
        """Extracts percentage from progress bar style attribute."""
        if not element:
            return 0.0
        
        style = element.get('style', '')
        match = re.search(r'width:\s*(\d+(?:\.\d+)?)', style)
        if match:
            return float(match.group(1))
        
        # Try text content
        text = element.text.strip()
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        
        return 0.0
    
    def _parse_price(self, text: str) -> float:
        """Parses price from text."""
        if not text:
            return 0.0
        
        # Remove non-numeric characters except decimal
        clean = re.sub(r'[^\d.]', '', text.strip())
        try:
            return float(clean) if clean else 0.0
        except ValueError:
            return 0.0


# Singleton instance
herd_radar = SentimentScraper()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    sentiment = herd_radar.get_sentiment("XAUUSD")
    if sentiment:
        print(f"\n=== XAUUSD Sentiment ===")
        print(f"Long: {sentiment.long_pct:.1f}%")
        print(f"Short: {sentiment.short_pct:.1f}%")
        print(f"Bias: {sentiment.liquidity_bias}")
        print(f"Liquidity Pool: {sentiment.liquidity_direction}")
        print(f"\nNeural Features:")
        for k, v in sentiment.to_neural_dict().items():
            print(f"  {k}: {v:.4f}")
