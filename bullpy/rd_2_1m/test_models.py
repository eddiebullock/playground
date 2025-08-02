#!/usr/bin/env python3
"""
Simple test script to verify data models work
"""

import sys
import os
from datetime import datetime
from decimal import Decimal

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.portfolio import Portfolio
from models.investment import Investment
from models.transaction import Transaction, TransactionType

def test_portfolio():
    """Test portfolio creation"""
    print("Testing Portfolio model...")
    
    portfolio = Portfolio(
        id="portfolio_1",
        name="Main Investment Portfolio",
        description="Primary investment portfolio",
        broker="Freetrade",
        account_type="ISA",
        created_date=datetime.now()
    )
    
    print(f"Created portfolio: {portfolio.name}")
    print(f"Broker: {portfolio.broker}")
    print(f"Account type: {portfolio.account_type}")
    print("Portfolio test passed!")

def test_investment():
    """Test investment creation"""
    print("\nTesting Investment model...")
    
    investment = Investment(
        id="inv_1",
        portfolio_id="portfolio_1",
        symbol="VWRL",
        name="Vanguard FTSE All-World UCITS ETF",
        investment_type="etf",
        quantity=Decimal("100"),
        average_price=Decimal("85.50"),
        current_price=Decimal("90.25")
    )
    
    print(f"Created investment: {investment.name}")
    print(f"Symbol: {investment.symbol}")
    print(f"Current value: {investment.get_current_value_formatted()}")
    print(f"Gain/Loss: {investment.get_gain_loss_formatted()}")
    print("Investment test passed!")

def test_transaction():
    """Test transaction creation"""
    print("\nTesting Transaction model...")
    
    transaction = Transaction(
        id="txn_1",
        portfolio_id="portfolio_1",
        transaction_type=TransactionType.CONTRIBUTION,
        date=datetime.now(),
        amount=Decimal("2000"),
        description="Monthly contribution"
    )
    
    print(f"Created transaction: {transaction.transaction_type.value}")
    print(f"Amount: {transaction.get_formatted_amount()}")
    print(f"Description: {transaction.description}")
    print("Transaction test passed!")

def main():
    """Run all tests"""
    print("Testing data models...")
    print("=" * 50)
    
    try:
        test_portfolio()
        test_investment()
        test_transaction()
        print("\n" + "=" * 50)
        print("All tests passed! Data models are working correctly.")
    except Exception as e:
        print(f"Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 