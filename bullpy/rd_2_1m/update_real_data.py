#!/usr/bin/env python3
"""
Script to update your real portfolio data
"""

import sys
import os
from decimal import Decimal
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.storage import DataStorage
from src.data.portfolio_loader import PortfolioLoader
from src.models.portfolio import Portfolio

def update_portfolio_data():
    """Interactive script to update portfolio data with real values"""
    
    storage = DataStorage()
    loader = PortfolioLoader(storage)
    
    print("Portfolio Data Update Tool")
    print("=" * 50)
    
    # Load existing portfolios
    portfolios = storage.load_portfolios()
    
    if not portfolios:
        print("No portfolios found. Creating sample portfolios first...")
        loader.create_sample_portfolios()
        portfolios = storage.load_portfolios()
    
    print("Current portfolios:")
    for i, portfolio in enumerate(portfolios, 1):
        print(f"{i}. {portfolio.name} ({portfolio.broker}) - {portfolio.get_total_value_formatted()}")
    
    print("\nEnter your real portfolio values:")
    print("(Press Enter to skip updating a portfolio)")
    
    for portfolio in portfolios:
        print(f"\n{portfolio.name} ({portfolio.broker}):")
        
        try:
            total_value_str = input(f"Total value (current: {portfolio.get_total_value_formatted()}): ").strip()
            if total_value_str:
                total_value = Decimal(total_value_str.replace('$', '').replace(',', ''))
                
                cash_balance_str = input(f"Cash balance (current: {portfolio.cash_balance}): ").strip()
                cash_balance = Decimal(cash_balance_str) if cash_balance_str else portfolio.cash_balance
                
                invested_amount_str = input(f"Invested amount (current: {portfolio.invested_amount}): ").strip()
                invested_amount = Decimal(invested_amount_str) if invested_amount_str else portfolio.invested_amount
                
                # Update portfolio
                portfolio.update_values(total_value, cash_balance, invested_amount)
                print(f"Updated {portfolio.name} with new values")
                
        except (ValueError, KeyboardInterrupt):
            print("Skipping this portfolio...")
            continue
    
    print("\nPortfolio data updated!")
    print("Run 'python main.py' to see your updated projections")

def add_new_portfolio():
    """Add a new portfolio"""
    
    storage = DataStorage()
    
    print("Add New Portfolio")
    print("=" * 50)
    
    portfolio_id = input("Portfolio ID (e.g., 'freetrade_isa'): ").strip()
    name = input("Portfolio name: ").strip()
    broker = input("Broker (e.g., 'Freetrade', 'AJ Bell'): ").strip()
    account_type = input("Account type (e.g., 'ISA', 'SIPP', 'GIA'): ").strip()
    
    try:
        total_value = Decimal(input("Total value: ").strip())
        cash_balance = Decimal(input("Cash balance: ").strip())
        invested_amount = Decimal(input("Invested amount: ").strip())
        
        portfolio = Portfolio(
            id=portfolio_id,
            name=name,
            description=f"{broker} {account_type} account",
            broker=broker,
            account_type=account_type,
            created_date=datetime.now(),
            total_value=total_value,
            cash_balance=cash_balance,
            invested_amount=invested_amount
        )
        
        # Calculate returns
        if portfolio.invested_amount > 0:
            portfolio.total_return = portfolio.total_value - portfolio.invested_amount
            portfolio.total_return_percentage = (portfolio.total_return / portfolio.invested_amount) * 100
        
        storage.save_portfolio(portfolio)
        print(f"Added new portfolio: {name}")
        
    except ValueError:
        print("Invalid input. Please enter valid numbers.")

def main():
    """Main menu"""
    while True:
        print("\nPortfolio Data Manager")
        print("=" * 50)
        print("1. Update existing portfolio values")
        print("2. Add new portfolio")
        print("3. View current data")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            update_portfolio_data()
        elif choice == '2':
            add_new_portfolio()
        elif choice == '3':
            storage = DataStorage()
            portfolios = storage.load_portfolios()
            print("\nCurrent portfolios:")
            for portfolio in portfolios:
                print(f"- {portfolio.name}: {portfolio.get_total_value_formatted()}")
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 