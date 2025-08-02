"""
Market data service for real-time stock prices
"""

import yfinance as yf
from typing import Dict, List, Optional
from decimal import Decimal
import time


class MarketDataService:
    """Handles real-time market data from Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # Cache prices for 60 seconds
    
    def get_stock_price(self, symbol: str) -> Optional[Decimal]:
        """Get current stock price for a symbol"""
        try:
            # Check cache first
            if symbol in self.cache:
                price, timestamp = self.cache[symbol]
                if time.time() - timestamp < self.cache_timeout:
                    return price
            
            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if 'regularMarketPrice' in info and info['regularMarketPrice']:
                price = Decimal(str(info['regularMarketPrice']))
                self.cache[symbol] = (price, time.time())
                return price
            else:
                print(f"Could not get price for {symbol}")
                return None
                
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'price': Decimal(str(info.get('regularMarketPrice', 0))),
                'change': Decimal(str(info.get('regularMarketChange', 0))),
                'change_percent': Decimal(str(info.get('regularMarketChangePercent', 0))),
                'market_cap': info.get('marketCap'),
                'volume': info.get('volume'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield')
            }
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        """Get prices for multiple symbols at once"""
        prices = {}
        for symbol in symbols:
            price = self.get_stock_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    def update_investment_prices(self, investments: List) -> List:
        """Update current prices for a list of investments"""
        updated_investments = []
        
        for investment in investments:
            if investment.symbol:
                new_price = self.get_stock_price(investment.symbol)
                if new_price:
                    investment.update_price(new_price)
            updated_investments.append(investment)
        
        return updated_investments 