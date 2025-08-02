#!/usr/bin/env python3
"""
Test API connections for broker integrations
"""

import sys
import os
from decimal import Decimal

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.storage import DataStorage
from src.services.portfolio_sync import PortfolioSyncService
from src.services.market_data import MarketDataService

def test_yahoo_finance():
    """Test Yahoo Finance API"""
    print("Testing Yahoo Finance API...")
    
    market_data = MarketDataService()
    
    # Test some common stocks
    test_symbols = ['AAPL', 'MSFT', 'VWRL.L', 'TSLA']
    
    for symbol in test_symbols:
        price = market_data.get_stock_price(symbol)
        if price:
            print(f"SUCCESS {symbol}: ${price}")
        else:
            print(f"FAILED {symbol}: Failed to get price")
    
    print()

def test_broker_apis():
    """Test broker API connections"""
    print("Testing Broker API Connections...")
    print("=" * 50)
    
    storage = DataStorage()
    sync_service = PortfolioSyncService(storage)
    
    # Get sync status
    status = sync_service.get_sync_status()
    
    for broker, info in status.items():
        if broker == 'yahoo_finance':
            continue
            
        print(f"{broker.upper()}:")
        print(f"  Connected: {'YES' if info['connected'] else 'NO'}")
        print(f"  API Key: {info['api_key']}")
        print()
    
    # Test Yahoo Finance
    print("YAHOO FINANCE:")
    print(f"  Status: {'Available' if status['yahoo_finance']['connected'] else 'Not available'}")
    print()

def setup_api_keys():
    """Guide for setting up API keys"""
    print("API Setup Guide")
    print("=" * 50)
    
    print("To connect your broker APIs, you need to:")
    print()
    
    print("1. FREETRADE:")
    print("   - Check if Freetrade offers API access")
    print("   - Look for API documentation on their website")
    print("   - Generate API key from your account settings")
    print("   - Add key to config/api_keys.py")
    print()
    
    print("2. AJ BELL:")
    print("   - Check if AJ Bell offers API access")
    print("   - Look for API documentation on their website")
    print("   - Generate API key from your account settings")
    print("   - Add key to config/api_keys.py")
    print()
    
    print("3. COINBASE:")
    print("   - Go to Coinbase Pro or Advanced Trade")
    print("   - Generate API key with appropriate permissions")
    print("   - Add key and secret to config/api_keys.py")
    print()
    
    print("4. YAHOO FINANCE:")
    print("   - No API key needed (already working)")
    print("   - Used for real-time stock prices")
    print()

def test_market_data():
    """Test market data functionality"""
    print("Testing Market Data Service...")
    print("=" * 50)
    
    market_data = MarketDataService()
    
    # Test getting stock info
    symbols = ['AAPL', 'VWRL.L']
    
    for symbol in symbols:
        info = market_data.get_stock_info(symbol)
        if info:
            print(f"SUCCESS {symbol}:")
            print(f"  Name: {info['name']}")
            print(f"  Price: ${info['price']}")
            print(f"  Change: {info['change_percent']}%")
        else:
            print(f"FAILED {symbol}: Failed to get info")
        print()

def main():
    """Main test function"""
    print("API Connection Test Suite")
    print("=" * 50)
    print()
    
    # Test Yahoo Finance
    test_yahoo_finance()
    
    # Test broker APIs
    test_broker_apis()
    
    # Test market data
    test_market_data()
    
    # Show setup guide
    setup_api_keys()
    
    print("Next Steps:")
    print("1. Get API keys from your brokers")
    print("2. Add them to config/api_keys.py")
    print("3. Run this test again to verify connections")
    print("4. Use the sync service to pull real data")

if __name__ == "__main__":
    main() 