import requests
import time
import random
from datetime import datetime
from tabulate import tabulate
from config import ALPHA_VANTAGE_API_KEY

class RealtimeStockApp:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        
        # Define tickers with their types and display names
        self.tickers = {
            'BTC': {'type': 'crypto', 'name': 'Bitcoin'},
            'SOL': {'type': 'crypto', 'name': 'Solana'},
            'MSFT': {'type': 'stock', 'name': 'Microsoft'},
            'GB00BJS8SJ3': {'type': 'fund', 'name': 'Fidelity Index World Fund (ISIN)'},
            'VWRP.L': {'type': 'stock', 'name': 'Vanguard FTSE All-World UCITS ETF'}
        }
        
        # Multiple data sources for redundancy
        self.data_sources = [
            'alpha_vantage',
            'yahoo_finance',
            'alternative_sources'
        ]
    
    def get_alpha_vantage_crypto(self, symbol):
        """Get cryptocurrency price from Alpha Vantage"""
        if not self.api_key:
            return None
            
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': symbol,
            'to_currency': 'USD',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Realtime Currency Exchange Rate' in data:
                rate_info = data['Realtime Currency Exchange Rate']
                return {
                    'price': float(rate_info['5. Exchange Rate']),
                    'currency': 'USD',
                    'last_updated': rate_info['6. Last Refreshed'],
                    'source': 'Alpha Vantage'
                }
            elif 'Note' in data:
                print(f"Alpha Vantage limit: {data['Note']}")
                return None
        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")
        return None
    
    def get_alpha_vantage_stock(self, symbol):
        """Get stock price from Alpha Vantage"""
        if not self.api_key:
            return None
            
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                if quote.get('05. price') and quote['05. price'] != 'None':
                    return {
                        'price': float(quote['05. price']),
                        'currency': 'USD',
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': quote.get('10. change percent', '0%'),
                        'last_updated': quote.get('07. latest trading day', 'N/A'),
                        'source': 'Alpha Vantage'
                    }
            elif 'Note' in data:
                print(f"Alpha Vantage limit: {data['Note']}")
                return None
        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")
        return None
    
    def get_yahoo_finance_quote(self, symbol):
        """Get quote from Yahoo Finance using direct HTTP"""
        try:
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 2.0))
            
            url = "https://query1.finance.yahoo.com/v7/finance/quote"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            resp = requests.get(url, params={"symbols": symbol}, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            result = data.get("quoteResponse", {}).get("result", [])
            if not result:
                return None
                
            row = result[0]
            price = row.get("regularMarketPrice")
            if price is None:
                return None
                
            currency = row.get("currency") or "USD"
            ts = row.get("regularMarketTime")
            last_updated = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'price': float(price),
                'currency': currency,
                'change': 0,
                'change_percent': '0%',
                'last_updated': last_updated,
                'source': 'Yahoo Finance'
            }
        except Exception as e:
            print(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    def get_crypto_price(self, symbol):
        """Get cryptocurrency price from multiple sources"""
        print(f"  Trying Alpha Vantage...")
        price_data = self.get_alpha_vantage_crypto(symbol)
        if price_data:
            return price_data
        
        print(f"  Trying Yahoo Finance...")
        price_data = self.get_yahoo_finance_quote(f"{symbol}-USD")
        if price_data:
            return price_data
        
        print(f"  Trying alternative format...")
        price_data = self.get_yahoo_finance_quote(symbol)
        if price_data:
            return price_data
        
        return None
    
    def get_stock_price(self, symbol):
        """Get stock price from multiple sources"""
        print(f"  Trying Alpha Vantage...")
        price_data = self.get_alpha_vantage_stock(symbol)
        if price_data:
            return price_data
        
        print(f"  Trying Yahoo Finance...")
        price_data = self.get_yahoo_finance_quote(symbol)
        if price_data:
            return price_data
        
        return None
    
    def get_fund_price(self, isin):
        """Get fund price using multiple approaches"""
        print(f"  Trying Yahoo Finance with ISIN...")
        price_data = self.get_yahoo_finance_quote(isin)
        if price_data:
            return price_data
        
        # Try known ticker variations
        candidates = ['PIWOA.L', '0P000125KV.L', 'PIWOA']
        for candidate in candidates:
            print(f"  Trying {candidate}...")
            price_data = self.get_yahoo_finance_quote(candidate)
            if price_data:
                return price_data
        
        return None
    
    def get_all_prices(self):
        """Get prices for all tickers with multiple fallback strategies"""
        results = []
        
        for symbol, info in self.tickers.items():
            print(f"Fetching price for {info['name']} ({symbol})...")
            
            if info['type'] == 'crypto':
                price_data = self.get_crypto_price(symbol)
                if price_data:
                    results.append({
                        'Name': info['name'],
                        'Symbol': symbol,
                        'Type': 'Crypto',
                        'Price': f"${price_data['price']:.4f}",
                        'Currency': price_data['currency'],
                        'Last Updated': price_data['last_updated'],
                        'Source': price_data['source']
                    })
                else:
                    print(f"  ❌ Failed to get price for {symbol}")
                    
            elif info['type'] == 'fund':
                price_data = self.get_fund_price(symbol)
                if price_data:
                    results.append({
                        'Name': info['name'],
                        'Symbol': symbol,
                        'Type': 'Fund',
                        'Price': f"{price_data['currency']} {price_data['price']:.2f}",
                        'Last Updated': price_data['last_updated'],
                        'Source': price_data['source']
                    })
                else:
                    print(f"  ❌ Failed to get price for {symbol}")
                    
            else:
                price_data = self.get_stock_price(symbol)
                if price_data:
                    results.append({
                        'Name': info['name'],
                        'Symbol': symbol,
                        'Type': 'Stock',
                        'Price': f"${price_data['price']:.2f}",
                        'Change': f"{price_data['change']:+.2f}",
                        'Change %': price_data['change_percent'],
                        'Last Updated': price_data['last_updated'],
                        'Source': price_data['source']
                    })
                else:
                    print(f"  ❌ Failed to get price for {symbol}")
            
            # Add delay between tickers to be respectful to APIs
            if symbol != list(self.tickers.keys())[-1]:  # Don't delay after last ticker
                delay = random.uniform(2, 5)
                print(f"  Waiting {delay:.1f} seconds before next request...")
                time.sleep(delay)
        
        return results
    
    def display_prices(self, prices):
        """Display prices in a formatted table"""
        if not prices:
            print("No price data available.")
            return
        
        print(f"\n📊 Real-Time Stock & Crypto Prices - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
        
        # Create table with source information
        table_data = []
        for price in prices:
            if 'Change' in price:
                # Stock data with change information
                table_data.append([
                    price['Name'],
                    price['Symbol'],
                    price['Type'],
                    price['Price'],
                    price.get('Change', ''),
                    price.get('Change %', ''),
                    price['Last Updated'],
                    price.get('Source', 'Unknown')
                ])
            else:
                # Crypto/Fund data
                table_data.append([
                    price['Name'],
                    price['Symbol'],
                    price['Type'],
                    price['Price'],
                    price.get('Currency', ''),
                    price['Last Updated'],
                    price.get('Source', 'Unknown')
                ])
        
        if any('Change' in price for price in prices):
            headers = ['Name', 'Symbol', 'Type', 'Price', 'Change', 'Change %', 'Last Updated', 'Source']
        else:
            headers = ['Name', 'Symbol', 'Type', 'Price', 'Currency', 'Last Updated', 'Source']
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def run(self):
        """Main application loop"""
        print("🚀 Starting Real-Time Stock & Crypto Price Tracker...")
        print("This app will fetch LIVE prices every time you run it!")
        print("=" * 60)
        
        if not self.api_key:
            print("⚠️  No Alpha Vantage API key found. Some data sources may be limited.")
            print("   Set ALPHA_VANTAGE_API_KEY in your .env file for full functionality.")
        
        print("\nFetching current prices from multiple sources...")
        print("(This may take a few minutes due to API rate limiting)")
        
        start_time = time.time()
        prices = self.get_all_prices()
        end_time = time.time()
        
        self.display_prices(prices)
        
        print(f"\n✅ Price update completed in {end_time - start_time:.1f} seconds")
        print(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary
        successful = len(prices)
        total = len(self.tickers)
        print(f"\n📈 Successfully fetched: {successful}/{total} tickers")
        
        if successful < total:
            print("\n💡 Tips for better results:")
            print("   • Wait a few minutes between runs to avoid rate limits")
            print("   • Upgrade Alpha Vantage to premium for unlimited access")
            print("   • Try running during off-peak hours")
        else:
            print("\n🎉 All tickers successfully updated!")

def main():
    app = RealtimeStockApp()
    app.run()

if __name__ == "__main__":
    main()
