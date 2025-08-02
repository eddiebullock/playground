#!/usr/bin/env python3
"""
Update all investment prices with real market data
"""

import sys
import os
from decimal import Decimal

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.storage import DataStorage
from src.services.portfolio_simulator import PortfolioSimulator

def main():
    """Update all investment prices"""
    print("Updating Investment Prices")
    print("=" * 40)
    
    storage = DataStorage()
    simulator = PortfolioSimulator(storage)
    
    # Update all prices
    result = simulator.update_all_prices()
    
    print(f"\nUpdate Results:")
    print(f"  Updated: {result['updated']} investments")
    print(f"  Failed: {result['failed']} investments")
    print(f"  Total: {result['total_investments']} investments")
    
    # Get updated summary
    summary = simulator.get_portfolio_summary()
    
    print(f"\nUpdated Portfolio Summary:")
    print(f"  Total Value: ${summary['total_value']:,.2f}")
    print(f"  Total Invested: ${summary['total_invested']:,.2f}")
    print(f"  Total Return: ${summary['total_return']:,.2f} ({summary['total_return_percentage']:.2f}%)")
    
    print(f"\nLast Updated: {summary['last_updated']}")

if __name__ == "__main__":
    main() 