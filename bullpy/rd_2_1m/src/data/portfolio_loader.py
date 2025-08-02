"""
Portfolio data loader for importing real portfolio data
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
from decimal import Decimal
import uuid

from src.models.portfolio import Portfolio
from src.models.investment import Investment
from src.models.transaction import Transaction, TransactionType
from src.data.storage import DataStorage


class PortfolioLoader:
    """Loads and manages portfolio data"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
    
    def create_sample_portfolios(self):
        """Create sample portfolios for testing - replace with your real data"""
        portfolios = [
            {
                'id': 'freetrade_isa',
                'name': 'Freetrade ISA',
                'description': 'Main investment portfolio in Freetrade ISA',
                'broker': 'Freetrade',
                'account_type': 'ISA',
                'total_value': Decimal('25000'),
                'cash_balance': Decimal('1500'),
                'invested_amount': Decimal('23500')
            },
            {
                'id': 'ajbell_sipp',
                'name': 'AJ Bell SIPP',
                'description': 'Pension portfolio in AJ Bell',
                'broker': 'AJ Bell',
                'account_type': 'SIPP',
                'total_value': Decimal('12000'),
                'cash_balance': Decimal('500'),
                'invested_amount': Decimal('11500')
            },
            {
                'id': 'freetrade_gia',
                'name': 'Freetrade GIA',
                'description': 'General investment account',
                'broker': 'Freetrade',
                'account_type': 'GIA',
                'total_value': Decimal('7000'),
                'cash_balance': Decimal('200'),
                'invested_amount': Decimal('6800')
            }
        ]
        
        for portfolio_data in portfolios:
            portfolio = Portfolio(
                id=portfolio_data['id'],
                name=portfolio_data['name'],
                description=portfolio_data['description'],
                broker=portfolio_data['broker'],
                account_type=portfolio_data['account_type'],
                created_date=datetime.now(),
                total_value=portfolio_data['total_value'],
                cash_balance=portfolio_data['cash_balance'],
                invested_amount=portfolio_data['invested_amount']
            )
            
            # Calculate returns
            if portfolio.invested_amount > 0:
                portfolio.total_return = portfolio.total_value - portfolio.invested_amount
                portfolio.total_return_percentage = (portfolio.total_return / portfolio.invested_amount) * 100
            
            self.storage.save_portfolio(portfolio)
    
    def create_sample_investments(self):
        """Create sample investments - replace with your real holdings"""
        investments = [
            {
                'portfolio_id': 'freetrade_isa',
                'symbol': 'VWRL',
                'name': 'Vanguard FTSE All-World UCITS ETF',
                'investment_type': 'etf',
                'quantity': Decimal('200'),
                'average_price': Decimal('85.50'),
                'current_price': Decimal('90.25')
            },
            {
                'portfolio_id': 'freetrade_isa',
                'symbol': 'AAPL',
                'name': 'Apple Inc',
                'investment_type': 'stock',
                'quantity': Decimal('10'),
                'average_price': Decimal('150.00'),
                'current_price': Decimal('175.50')
            },
            {
                'portfolio_id': 'ajbell_sipp',
                'symbol': 'VWRL',
                'name': 'Vanguard FTSE All-World UCITS ETF',
                'investment_type': 'etf',
                'quantity': Decimal('100'),
                'average_price': Decimal('82.00'),
                'current_price': Decimal('90.25')
            },
            {
                'portfolio_id': 'freetrade_gia',
                'symbol': 'TSLA',
                'name': 'Tesla Inc',
                'investment_type': 'stock',
                'quantity': Decimal('5'),
                'average_price': Decimal('200.00'),
                'current_price': Decimal('250.00')
            }
        ]
        
        for inv_data in investments:
            investment = Investment(
                id=str(uuid.uuid4()),
                portfolio_id=inv_data['portfolio_id'],
                symbol=inv_data['symbol'],
                name=inv_data['name'],
                investment_type=inv_data['investment_type'],
                quantity=inv_data['quantity'],
                average_price=inv_data['average_price'],
                current_price=inv_data['current_price']
            )
            
            self.storage.save_investment(investment)
    
    def create_sample_transactions(self):
        """Create sample transactions - replace with your real transaction history"""
        transactions = [
            {
                'portfolio_id': 'freetrade_isa',
                'transaction_type': TransactionType.CONTRIBUTION,
                'amount': Decimal('2000'),
                'description': 'Monthly ISA contribution'
            },
            {
                'portfolio_id': 'freetrade_isa',
                'transaction_type': TransactionType.PURCHASE,
                'amount': Decimal('17100'),
                'quantity': Decimal('200'),
                'price_per_share': Decimal('85.50'),
                'symbol': 'VWRL',
                'description': 'Purchase VWRL shares'
            },
            {
                'portfolio_id': 'ajbell_sipp',
                'transaction_type': TransactionType.CONTRIBUTION,
                'amount': Decimal('500'),
                'description': 'Monthly pension contribution'
            }
        ]
        
        for txn_data in transactions:
            transaction = Transaction(
                id=str(uuid.uuid4()),
                portfolio_id=txn_data['portfolio_id'],
                transaction_type=txn_data['transaction_type'],
                date=datetime.now(),
                amount=txn_data['amount'],
                quantity=txn_data.get('quantity'),
                price_per_share=txn_data.get('price_per_share'),
                symbol=txn_data.get('symbol'),
                description=txn_data['description']
            )
            
            self.storage.save_transaction(transaction)
    
    def load_real_portfolio_data(self, csv_file_path: str):
        """Load real portfolio data from a CSV file"""
        if not os.path.exists(csv_file_path):
            print(f"CSV file not found: {csv_file_path}")
            return
        
        with open(csv_file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create portfolio from CSV data
                portfolio = Portfolio(
                    id=row['id'],
                    name=row['name'],
                    description=row.get('description', ''),
                    broker=row['broker'],
                    account_type=row['account_type'],
                    created_date=datetime.fromisoformat(row['created_date']),
                    total_value=Decimal(row['total_value']),
                    cash_balance=Decimal(row['cash_balance']),
                    invested_amount=Decimal(row['invested_amount'])
                )
                
                self.storage.save_portfolio(portfolio)
    
    def update_portfolio_values(self, portfolio_id: str, 
                              total_value: Decimal, 
                              cash_balance: Decimal,
                              invested_amount: Decimal):
        """Update portfolio values with real data"""
        portfolios = self.storage.load_portfolios()
        
        for portfolio in portfolios:
            if portfolio.id == portfolio_id:
                portfolio.update_values(total_value, cash_balance, invested_amount)
                # Note: In a real implementation, you'd want to update the CSV file
                # For now, we'll just update the in-memory object
                break
    
    def get_total_portfolio_value(self) -> Decimal:
        """Get total value across all portfolios"""
        portfolios = self.storage.load_portfolios()
        return sum(p.total_value for p in portfolios)
    
    def get_portfolio_breakdown(self) -> List[Dict]:
        """Get breakdown of all portfolios"""
        portfolios = self.storage.load_portfolios()
        breakdown = []
        
        for portfolio in portfolios:
            breakdown.append({
                'id': portfolio.id,
                'name': portfolio.name,
                'broker': portfolio.broker,
                'account_type': portfolio.account_type,
                'total_value': portfolio.total_value,
                'cash_balance': portfolio.cash_balance,
                'invested_amount': portfolio.invested_amount,
                'return_percentage': portfolio.total_return_percentage
            })
        
        return breakdown 