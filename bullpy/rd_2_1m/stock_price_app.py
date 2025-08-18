import requests
import time
from datetime import datetime
from tabulate import tabulate
from config import ALPHA_VANTAGE_API_KEY

class StockPriceApp:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        
        # Define tickers with their types and display names
        self.tickers = {
            'BTC': {'type': 'crypto', 'name': 'Bitcoin'},
            'SOL': {'type': 'crypto', 'name': 'Solana'},
            'MSFT': {'type': 'stock', 'name': 'Microsoft'},
            'GB00BJS8SJ3': {'type': 'fund', 'name': 'Fidelity Index World Fund (ISIN)'},
            'VWRP.L': {'type': 'stock', 'name': 'Vanguard FTSE All-World UCITS ETF'}
        }
    
    def get_crypto_price(self, symbol):
        """Get cryptocurrency price from Alpha Vantage"""
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': symbol,
            'to_currency': 'USD',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Realtime Currency Exchange Rate' in data:
                rate_info = data['Realtime Currency Exchange Rate']
                return {
                    'price': float(rate_info['5. Exchange Rate']),
                    'currency': 'USD',
                    'last_updated': rate_info['6. Last Refreshed']
                }
            else:
                return None
        except Exception as e:
            print(f"Error fetching {symbol} price: {e}")
            return None
    
    def get_stock_price(self, symbol):
        """Get stock price from Alpha Vantage"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                # Check if we have valid price data
                if quote.get('05. price') and quote['05. price'] != 'None':
                    return {
                        'price': float(quote['05. price']),
                        'currency': quote.get('08. previous close', 'USD'),  # Try to get actual currency
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': quote.get('10. change percent', '0%'),
                        'last_updated': quote.get('07. latest trading day', 'N/A')
                    }
                else:
                    print(f"No valid price data for {symbol}")
                    return None
            elif 'Note' in data:
                print(f"API limit reached for {symbol}: {data['Note']}")
                return None
            elif 'Error Message' in data:
                print(f"API error for {symbol}: {data['Error Message']}")
                return None
            else:
                print(f"No quote data available for {symbol}")
                return None
        except Exception as e:
            print(f"Error fetching {symbol} price: {e}")
            return None

    def get_yahoo_quote(self, symbol: str):
        """Fetch quote from Yahoo Finance v7 quote endpoint without yfinance.
        Returns dict with price, currency, last_updated if available, else None.
        """
        try:
            url = "https://query1.finance.yahoo.com/v7/finance/quote"
            resp = requests.get(url, params={"symbols": symbol}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            result = (data or {}).get("quoteResponse", {}).get("result", [])
            if not result:
                return None
            row = result[0]
            price = row.get("regularMarketPrice")
            if price is None:
                return None
            currency = row.get("currency") or "GBP"
            ts = row.get("regularMarketTime")
            last_updated = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else datetime.now().strftime('%Y-%m-%d')
            return {
                'symbol': symbol,
                'price': float(price),
                'currency': currency,
                'last_updated': last_updated
            }
        except Exception:
            return None

    def get_fund_price_yf(self, isin: str):
        """Fetch fund price using Yahoo Finance by trying known candidate tickers for the ISIN."""
        try:
            candidates = ['PIWOA.L', '0P000125KV.L', f'{isin}.L', isin, 'PIWOA']
            for symbol in candidates:
                try:
                    # Try lightweight HTTP quote first
                    quote = self.get_yahoo_quote(symbol)
                    if quote:
                        return quote
                except Exception:
                    continue
            return None
        except Exception as exc:
            print(f"Yahoo Finance error for {isin}: {exc}")
            return None

    def get_stock_price_yf(self, symbol):
        """Get stock price from Yahoo Finance as fallback using direct HTTP"""
        try:
            # Use the direct Yahoo Finance quote endpoint
            url = "https://query1.finance.yahoo.com/v7/finance/quote"
            resp = requests.get(url, params={"symbols": symbol}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            result = (data or {}).get("quoteResponse", {}).get("result", [])
            if not result:
                return None
            row = result[0]
            price = row.get("regularMarketPrice")
            if price is None:
                return None
            currency = row.get("currency") or "USD"
            ts = row.get("regularMarketTime")
            last_updated = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else datetime.now().strftime('%Y-%m-%d')
            return {
                'price': float(price),
                'currency': currency,
                'change': 0,  # Yahoo Finance doesn't provide change in this format
                'change_percent': '0%',
                'last_updated': last_updated
            }
        except Exception as e:
            print(f"Yahoo Finance fallback error for {symbol}: {e}")
            return None

    def get_crypto_price_yf(self, symbol):
        """Get cryptocurrency price from Yahoo Finance as fallback using direct HTTP"""
        try:
            # For crypto, try both symbol-USD and symbol-USD formats
            symbols_to_try = [f"{symbol}-USD", f"{symbol}USD", symbol]
            
            for sym in symbols_to_try:
                try:
                    url = "https://query1.finance.yahoo.com/v7/finance/quote"
                    resp = requests.get(url, params={"symbols": sym}, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    result = (data or {}).get("quoteResponse", {}).get("result", [])
                    if not result:
                        continue
                    row = result[0]
                    price = row.get("regularMarketPrice")
                    if price is not None:
                        ts = row.get("regularMarketTime")
                        last_updated = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else datetime.now().strftime('%Y-%m-%d')
                        return {
                            'price': float(price),
                            'currency': 'USD',
                            'last_updated': last_updated
                        }
                except Exception:
                    continue
            return None
        except Exception as e:
            print(f"Yahoo Finance crypto fallback error for {symbol}: {e}")
            return None
    
    def get_all_prices(self):
        """Get prices for all tickers"""
        results = []
        alpha_vantage_calls = 0  # Track Alpha Vantage API calls for rate limiting
        
        for symbol, info in self.tickers.items():
            print(f"Fetching price for {info['name']} ({symbol})...")
            
            if info['type'] == 'crypto':
                # Try Alpha Vantage first
                price_data = None
                if alpha_vantage_calls == 0:  # Only try Alpha Vantage if we haven't hit limits
                    price_data = self.get_crypto_price(symbol)
                    alpha_vantage_calls += 1
                
                # If Alpha Vantage fails, try Yahoo Finance with delay
                if not price_data:
                    print(f"Alpha Vantage failed for {symbol}, trying Yahoo Finance...")
                    time.sleep(2)  # Small delay to avoid rate limiting
                    price_data = self.get_crypto_price_yf(symbol)
                
                if price_data:
                    results.append({
                        'Name': info['name'],
                        'Symbol': symbol,
                        'Type': 'Crypto',
                        'Price': f"${price_data['price']:.4f}",
                        'Currency': price_data['currency'],
                        'Last Updated': price_data['last_updated']
                    })
            elif info['type'] == 'fund':
                # Yahoo Finance - add delay to avoid rate limiting
                time.sleep(2)
                fund = self.get_fund_price_yf(symbol)
                if fund:
                    results.append({
                        'Name': info['name'],
                        'Symbol': fund['symbol'],
                        'Type': 'Fund',
                        'Price': f"{fund['currency']} {fund['price']:.2f}",
                        'Last Updated': fund['last_updated']
                    })
            else:
                # Try Alpha Vantage first
                price_data = None
                if alpha_vantage_calls == 0:  # Only try Alpha Vantage if we haven't hit limits
                    price_data = self.get_stock_price(symbol)
                    alpha_vantage_calls += 1
                
                # If Alpha Vantage fails, try Yahoo Finance with delay
                if not price_data:
                    print(f"Alpha Vantage failed for {symbol}, trying Yahoo Finance...")
                    time.sleep(2)  # Small delay to avoid rate limiting
                    price_data = self.get_stock_price_yf(symbol)
                
                # If the main ticker fails, try alternative formats for international funds
                if not price_data and ('.L' in symbol or len(symbol) == 7):  # .L suffix or SEDOL codes
                    print(f"Trying alternative ticker formats for {symbol}...")
                    
                    # Try different formats for international funds
                    alt_formats = []
                    if '.L' in symbol:
                        alt_formats.append(symbol.replace('.L', ''))  # Remove .L
                    if len(symbol) == 7:  # SEDOL code
                        alt_formats.extend([f"{symbol}.L", f"{symbol}.VI"])  # Try .L and .VI suffixes
                    
                    for alt_symbol in alt_formats:
                        print(f"  Trying {alt_symbol}...")
                        time.sleep(1)  # Small delay between attempts
                        # Try Yahoo Finance for alternative formats
                        alt_price_data = self.get_stock_price_yf(alt_symbol)
                        if alt_price_data:
                            price_data = alt_price_data
                            symbol = alt_symbol  # Update symbol for display
                            break
                
                if price_data:
                    results.append({
                        'Name': info['name'],
                        'Symbol': symbol,
                        'Type': 'Stock',
                        'Price': f"${price_data['price']:.2f}",
                        'Change': f"{price_data['change']:+.2f}",
                        'Change %': price_data['change_percent'],
                        'Last Updated': price_data['last_updated']
                    })
            
            # Add delay between tickers to avoid overwhelming APIs
            time.sleep(3)
        
        return results
    
    def display_prices(self, prices):
        """Display prices in a formatted table"""
        if not prices:
            print("No price data available.")
            return
        
        print(f"\nStock & Crypto Prices - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Create table with appropriate columns
        if any('Change' in price for price in prices):
            # Stock data with change information
            table_data = [[
                price['Name'],
                price['Symbol'],
                price['Type'],
                price['Price'],
                price.get('Change', ''),
                price.get('Change %', ''),
                price['Last Updated']
            ] for price in prices]
            
            headers = ['Name', 'Symbol', 'Type', 'Price', 'Change', 'Change %', 'Last Updated']
        else:
            # Crypto data without change information
            table_data = [[
                price['Name'],
                price['Symbol'],
                price['Type'],
                price['Price'],
                price['Currency'],
                price['Last Updated']
            ] for price in prices]
            
            headers = ['Name', 'Symbol', 'Type', 'Price', 'Currency', 'Last Updated']
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def run(self):
        """Main application loop"""
        if not self.api_key:
            print("API key not configured. Please set ALPHA_VANTAGE_API_KEY in your environment.")
            return
        
        print("Starting Stock & Crypto Price Tracker...")
        print("Fetching current prices for all tickers...")
        
        prices = self.get_all_prices()
        self.display_prices(prices)
        
        print(f"\nPrice update completed at {datetime.now().strftime('%H:%M:%S')}")
        print("Note: Alpha Vantage free tier has rate limits. Consider upgrading for real-time updates.")
        
        # Check if any tickers failed
        failed_tickers = []
        for symbol, info in self.tickers.items():
            if not any(price['Symbol'] == symbol for price in prices):
                failed_tickers.append(f"{info['name']} ({symbol})")
        
        if failed_tickers:
            print(f"\nThe following tickers could not be fetched:")
            for ticker in failed_tickers:
                print(f"   • {ticker}")
            print("\nThis may be due to:")
            print("   • International fund tickers not being available in Alpha Vantage")
            print("   • Different ticker naming conventions")
            print("   • API data limitations for certain markets")
            print("\nConsider using alternative data sources like Yahoo Finance or Bloomberg for international funds.")

def main():
    app = StockPriceApp()
    app.run()

if __name__ == "__main__":
    main()
