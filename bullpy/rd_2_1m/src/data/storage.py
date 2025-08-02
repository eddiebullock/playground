"""
Data storage system for portfolio data
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
from decimal import Decimal

from src.models.portfolio import Portfolio
from src.models.investment import Investment
from src.models.transaction import Transaction, TransactionType


class DataStorage:
    """Handles data persistence for portfolios, investments, and transactions"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.portfolios_file = os.path.join(data_dir, "portfolios.csv")
        self.investments_file = os.path.join(data_dir, "investments.csv")
        self.transactions_file = os.path.join(data_dir, "transactions.csv")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize CSV files with headers if they don't exist
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Create CSV files with headers if they don't exist"""
        
        # Portfolios CSV
        if not os.path.exists(self.portfolios_file):
            with open(self.portfolios_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'id', 'name', 'description', 'broker', 'account_type',
                    'created_date', 'last_updated', 'total_value', 'cash_balance',
                    'invested_amount', 'total_return', 'total_return_percentage',
                    'is_active', 'notes'
                ])
        
        # Investments CSV
        if not os.path.exists(self.investments_file):
            with open(self.investments_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'id', 'portfolio_id', 'symbol', 'name', 'investment_type',
                    'quantity', 'average_price', 'current_price', 'total_cost',
                    'current_value', 'unrealized_gain_loss', 'unrealized_gain_loss_percentage',
                    'last_updated', 'purchase_date', 'is_active', 'notes'
                ])
        
        # Transactions CSV
        if not os.path.exists(self.transactions_file):
            with open(self.transactions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'id', 'portfolio_id', 'transaction_type', 'date', 'amount',
                    'quantity', 'price_per_share', 'symbol', 'total_value',
                    'fees', 'description', 'reference', 'notes'
                ])
    
    def save_portfolio(self, portfolio: Portfolio):
        """Save a portfolio to CSV"""
        with open(self.portfolios_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                portfolio.id, portfolio.name, portfolio.description,
                portfolio.broker, portfolio.account_type,
                portfolio.created_date.isoformat(),
                portfolio.last_updated.isoformat(),
                str(portfolio.total_value), str(portfolio.cash_balance),
                str(portfolio.invested_amount), str(portfolio.total_return),
                str(portfolio.total_return_percentage),
                portfolio.is_active, portfolio.notes
            ])
    
    def load_portfolios(self) -> List[Portfolio]:
        """Load all portfolios from CSV"""
        portfolios = []
        
        if not os.path.exists(self.portfolios_file):
            return portfolios
        
        with open(self.portfolios_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                portfolio = Portfolio.from_dict(row)
                portfolios.append(portfolio)
        
        return portfolios
    
    def save_investment(self, investment: Investment):
        """Save an investment to CSV"""
        with open(self.investments_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                investment.id, investment.portfolio_id, investment.symbol,
                investment.name, investment.investment_type,
                str(investment.quantity), str(investment.average_price),
                str(investment.current_price), str(investment.total_cost),
                str(investment.current_value), str(investment.unrealized_gain_loss),
                str(investment.unrealized_gain_loss_percentage),
                investment.last_updated.isoformat(),
                investment.purchase_date.isoformat() if investment.purchase_date else '',
                investment.is_active, investment.notes
            ])
    
    def load_investments(self, portfolio_id: Optional[str] = None) -> List[Investment]:
        """Load investments from CSV, optionally filtered by portfolio"""
        investments = []
        
        if not os.path.exists(self.investments_file):
            return investments
        
        with open(self.investments_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if portfolio_id and row['portfolio_id'] != portfolio_id:
                    continue
                investment = Investment.from_dict(row)
                investments.append(investment)
        
        return investments
    
    def save_transaction(self, transaction: Transaction):
        """Save a transaction to CSV"""
        with open(self.transactions_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                transaction.id, transaction.portfolio_id,
                transaction.transaction_type.value, transaction.date.isoformat(),
                str(transaction.amount),
                str(transaction.quantity) if transaction.quantity else '',
                str(transaction.price_per_share) if transaction.price_per_share else '',
                transaction.symbol or '',
                str(transaction.total_value), str(transaction.fees),
                transaction.description, transaction.reference, transaction.notes
            ])
    
    def load_transactions(self, portfolio_id: Optional[str] = None) -> List[Transaction]:
        """Load transactions from CSV, optionally filtered by portfolio"""
        transactions = []
        
        if not os.path.exists(self.transactions_file):
            return transactions
        
        with open(self.transactions_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if portfolio_id and row['portfolio_id'] != portfolio_id:
                    continue
                transaction = Transaction.from_dict(row)
                transactions.append(transaction)
        
        return transactions
    
    def get_portfolio_summary(self, portfolio_id: str) -> Dict:
        """Get a summary of a portfolio with its investments and transactions"""
        portfolios = self.load_portfolios()
        portfolio = next((p for p in portfolios if p.id == portfolio_id), None)
        
        if not portfolio:
            return {}
        
        investments = self.load_investments(portfolio_id)
        transactions = self.load_transactions(portfolio_id)
        
        # Calculate totals
        total_invested = sum(inv.total_cost for inv in investments)
        total_current_value = sum(inv.current_value for inv in investments)
        total_gain_loss = sum(inv.unrealized_gain_loss for inv in investments)
        
        return {
            'portfolio': portfolio,
            'investments': investments,
            'transactions': transactions,
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'total_gain_loss': total_gain_loss,
            'investment_count': len(investments),
            'transaction_count': len(transactions)
        } 