import requests
import time
from datetime import datetime
from tabulate import tabulate
from config import ALPHA_VANTAGE_API_KEY

class DemoStockPriceApp:
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
    
    def get_demo_prices(self):
        """Get demo prices to show what the app would display when working"""
        # These are sample prices - in a real working scenario, these would come from APIs
        demo_data = [
            {
                'Name': 'Bitcoin',
                'Symbol': 'BTC',
                'Type': 'Crypto',
                'Price': '$115,237.91',
                'Currency': 'USD',
                'Last Updated': '2025-08-18 10:40:00'
            },
            {
                'Name': 'Solana',
                'Symbol': 'SOL',
                'Type': 'Crypto',
                'Price': '$181.13',
                'Currency': 'USD',
                'Last Updated': '2025-08-18 10:40:05'
            },
            {
                'Name': 'Microsoft',
                'Symbol': 'MSFT',
                'Type': 'Stock',
                'Price': '$520.17',
                'Change': '-2.31',
                'Change %': '-0.4421%',
                'Last Updated': '2025-08-15'
            },
            {
                'Name': 'Fidelity Index World Fund (ISIN)',
                'Symbol': 'PIWOA.L',
                'Type': 'Fund',
                'Price': 'GBP 386.21',
                'Last Updated': '2025-08-18'
            },
            {
                'Name': 'Vanguard FTSE All-World UCITS ETF',
                'Symbol': 'VWRP.L',
                'Type': 'Stock',
                'Price': '$116.89',
                'Change': '+0.03',
                'Change %': '+0.0257%',
                'Last Updated': '2025-08-15'
            }
        ]
        return demo_data
    
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
        
        print("Starting Stock & Crypto Price Tracker (Demo Mode)...")
        print("Note: This is a demo showing what the app would display when APIs are working.")
        print("Current API status:")
        print("  • Alpha Vantage: Daily limit reached (25 requests/day)")
        print("  • Yahoo Finance: Rate limited due to too many requests")
        print("\nFetching demo prices...")
        
        prices = self.get_demo_prices()
        self.display_prices(prices)
        
        print(f"\nDemo completed at {datetime.now().strftime('%H:%M:%S')}")
        print("\nTo get real-time data, you can:")
        print("1. Wait until tomorrow for Alpha Vantage daily limit reset")
        print("2. Upgrade to Alpha Vantage premium plan")
        print("3. Use alternative data sources like IEX Cloud or Finnhub")
        print("4. Implement proper rate limiting with longer delays between requests")

def main():
    app = DemoStockPriceApp()
    app.run()

if __name__ == "__main__":
    main()
