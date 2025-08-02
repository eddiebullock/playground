#!/usr/bin/env python3
"""
Setup your real portfolio data with Yahoo Finance tracking
"""

import sys
import os
from decimal import Decimal

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.storage import DataStorage
from src.services.portfolio_simulator import PortfolioSimulator
from src.services.goal_calculator import GoalCalculator

def setup_freetrade_isa():
    """Setup Freetrade ISA portfolio"""
    print("Setting up Freetrade ISA...")
    
    # Example holdings - replace with your actual data
    holdings = [
        {
            'symbol': 'VWRL.L',
            'name': 'Vanguard FTSE All-World UCITS ETF',
            'quantity': Decimal('200'),
            'average_price': Decimal('85.50'),
            'type': 'etf'
        },
        {
            'symbol': 'AAPL',
            'name': 'Apple Inc',
            'quantity': Decimal('10'),
            'average_price': Decimal('150.00'),
            'type': 'stock'
        }
    ]
    
    # Replace with your actual total value
    total_value = Decimal('25000')
    
    return holdings, total_value

def setup_ajbell_sipp():
    """Setup AJ Bell SIPP portfolio"""
    print("Setting up AJ Bell SIPP...")
    
    holdings = [
        {
            'symbol': 'VWRL.L',
            'name': 'Vanguard FTSE All-World UCITS ETF',
            'quantity': Decimal('100'),
            'average_price': Decimal('82.00'),
            'type': 'etf'
        }
    ]
    
    total_value = Decimal('12000')
    
    return holdings, total_value

def setup_coinbase_crypto():
    """Setup Coinbase crypto portfolio"""
    print("Setting up Coinbase Crypto...")
    
    holdings = [
        {
            'symbol': 'BTC-USD',
            'name': 'Bitcoin',
            'quantity': Decimal('0.5'),
            'average_price': Decimal('40000'),
            'type': 'crypto'
        },
        {
            'symbol': 'ETH-USD',
            'name': 'Ethereum',
            'quantity': Decimal('2.0'),
            'average_price': Decimal('2500'),
            'type': 'crypto'
        }
    ]
    
    total_value = Decimal('7000')
    
    return holdings, total_value

def main():
    """Setup your real portfolio data"""
    print("Portfolio Setup - Real Data with Yahoo Finance Tracking")
    print("=" * 60)
    
    storage = DataStorage()
    simulator = PortfolioSimulator(storage)
    calculator = GoalCalculator()
    
    # Clear existing data
    print("Clearing existing sample data...")
    # Note: In a real implementation, you'd want to backup existing data first
    
    # Setup portfolios with your real data
    portfolios_data = [
        ('freetrade_isa', 'Freetrade ISA', 'Freetrade', 'ISA', setup_freetrade_isa()),
        ('ajbell_sipp', 'AJ Bell SIPP', 'AJ Bell', 'SIPP', setup_ajbell_sipp()),
        ('coinbase_crypto', 'Coinbase Crypto', 'Coinbase', 'CRYPTO', setup_coinbase_crypto())
    ]
    
    print("\nSetting up portfolios with real data...")
    for portfolio_id, name, broker, account_type, (holdings, total_value) in portfolios_data:
        print(f"\n{name}:")
        print(f"  Total Value: ${total_value:,.2f}")
        print(f"  Holdings: {len(holdings)}")
        
        for holding in holdings:
            print(f"    - {holding['name']} ({holding['symbol']}): {holding['quantity']} shares @ ${holding['average_price']}")
        
        # Create portfolio
        portfolio = simulator.create_manual_portfolio(
            portfolio_id, name, broker, account_type, total_value, holdings
        )
    
    # Update prices with real market data
    print("\nUpdating prices with real market data...")
    update_result = simulator.update_all_prices()
    
    # Get portfolio summary
    summary = simulator.get_portfolio_summary()
    
    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    
    print(f"Total Portfolio Value: ${summary['total_value']:,.2f}")
    print(f"Total Invested: ${summary['total_invested']:,.2f}")
    print(f"Total Return: ${summary['total_return']:,.2f} ({summary['total_return_percentage']:.2f}%)")
    
    print("\nPortfolio Breakdown:")
    for portfolio in summary['portfolios']:
        print(f"\n{portfolio['name']} ({portfolio['broker']}):")
        print(f"  Value: ${portfolio['total_value']:,.2f}")
        print(f"  Invested: ${portfolio['invested_amount']:,.2f}")
        print(f"  Return: ${portfolio['return']:,.2f} ({portfolio['return_percentage']:.2f}%)")
        print(f"  Holdings: {portfolio['holdings_count']}")
    
    # Calculate goal projections
    print("\n" + "=" * 60)
    print("GOAL PROJECTIONS")
    print("=" * 60)
    
    monthly_contribution = Decimal('2000')
    projection = calculator.calculate_goal_reach_date(summary['total_value'], monthly_contribution)
    
    if projection['months_to_goal']:
        years = projection['months_to_goal'] / 12
        print(f"With ${monthly_contribution} monthly contribution:")
        print(f"You'll reach $1M in {years:.1f} years")
        print(f"Target date: {projection['goal_date']}")
    else:
        print(f"With ${monthly_contribution} monthly contribution:")
        print("You won't reach $1M in 30 years")
        print(f"Final amount after 30 years: ${projection['final_amount']:,.2f}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Edit this script to add your real holdings and values")
    print("2. Run 'python main.py' to see your updated projections")
    print("3. Run 'python update_prices.py' to refresh market data")
    print("4. Add monthly contributions to track progress")

if __name__ == "__main__":
    main() 