"""
Portfolio simulator using Yahoo Finance for real-time market data
"""

from datetime import datetime
from typing import Dict, List
from decimal import Decimal
import uuid

from src.data.storage import DataStorage
from src.services.market_data import MarketDataService
from src.models.portfolio import Portfolio
from src.models.investment import Investment
from src.models.transaction import Transaction, TransactionType


class PortfolioSimulator:
    """Simulates portfolio performance using real market data"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.market_data = MarketDataService()
    
    def create_manual_portfolio(self, portfolio_id: str, name: str, broker: str, 
                               account_type: str, total_value: Decimal, 
                               holdings: List[Dict]) -> Portfolio:
        """Create a portfolio with manual holdings that track real market prices"""
        
        # Calculate cash balance (total - invested)
        invested_amount = sum(holding.get('quantity', 0) * holding.get('average_price', 0) 
                            for holding in holdings)
        cash_balance = total_value - invested_amount
        
        # Create portfolio
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
        
        self.storage.save_portfolio(portfolio)
        
        # Create investments with real-time price tracking
        for holding in holdings:
            investment = Investment(
                id=str(uuid.uuid4()),
                portfolio_id=portfolio_id,
                symbol=holding['symbol'],
                name=holding['name'],
                investment_type=holding.get('type', 'stock'),
                quantity=Decimal(str(holding['quantity'])),
                average_price=Decimal(str(holding['average_price'])),
                current_price=Decimal('0')  # Will be updated with real price
            )
            
            self.storage.save_investment(investment)
        
        return portfolio
    
    def update_all_prices(self) -> Dict:
        """Update all investment prices using Yahoo Finance"""
        print("Updating prices from Yahoo Finance...")
        
        investments = self.storage.load_investments()
        updated_count = 0
        failed_count = 0
        
        for investment in investments:
            if investment.symbol:
                new_price = self.market_data.get_stock_price(investment.symbol)
                if new_price:
                    investment.update_price(new_price)
                    updated_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to get price for {investment.symbol}")
        
        # Update portfolio totals
        self._update_portfolio_totals()
        
        print(f"Updated {updated_count} investments, {failed_count} failed")
        return {
            'updated': updated_count,
            'failed': failed_count,
            'total_investments': len(investments)
        }
    
    def _update_portfolio_totals(self):
        """Update portfolio totals based on current investment values"""
        portfolios = self.storage.load_portfolios()
        
        for portfolio in portfolios:
            investments = self.storage.load_investments(portfolio.id)
            
            # Calculate new totals
            total_invested = sum(inv.total_cost for inv in investments)
            total_current_value = sum(inv.current_value for inv in investments)
            total_return = total_current_value - total_invested
            
            # Update portfolio
            portfolio.update_values(
                total_current_value + portfolio.cash_balance,
                portfolio.cash_balance,
                total_invested
            )
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary with real-time data"""
        portfolios = self.storage.load_portfolios()
        
        total_value = Decimal('0')
        total_invested = Decimal('0')
        total_return = Decimal('0')
        portfolio_details = []
        
        for portfolio in portfolios:
            investments = self.storage.load_investments(portfolio.id)
            
            portfolio_invested = sum(inv.total_cost for inv in investments)
            portfolio_current = sum(inv.current_value for inv in investments)
            portfolio_return = portfolio_current - portfolio_invested
            
            total_value += portfolio.total_value
            total_invested += portfolio_invested
            total_return += portfolio_return
            
            portfolio_details.append({
                'id': portfolio.id,
                'name': portfolio.name,
                'broker': portfolio.broker,
                'total_value': portfolio.total_value,
                'invested_amount': portfolio_invested,
                'current_value': portfolio_current,
                'return': portfolio_return,
                'return_percentage': (portfolio_return / portfolio_invested * 100) if portfolio_invested > 0 else 0,
                'holdings_count': len(investments)
            })
        
        return {
            'total_value': total_value,
            'total_invested': total_invested,
            'total_return': total_return,
            'total_return_percentage': (total_return / total_invested * 100) if total_invested > 0 else 0,
            'portfolios': portfolio_details,
            'last_updated': datetime.now()
        }
    
    def add_holding(self, portfolio_id: str, symbol: str, name: str, 
                   quantity: Decimal, average_price: Decimal, 
                   investment_type: str = 'stock') -> Investment:
        """Add a new holding to a portfolio"""
        
        investment = Investment(
            id=str(uuid.uuid4()),
            portfolio_id=portfolio_id,
            symbol=symbol,
            name=name,
            investment_type=investment_type,
            quantity=quantity,
            average_price=average_price,
            current_price=Decimal('0')  # Will be updated with real price
        )
        
        self.storage.save_investment(investment)
        
        # Update portfolio totals
        self._update_portfolio_totals()
        
        return investment
    
    def record_contribution(self, portfolio_id: str, amount: Decimal, 
                          description: str = "Monthly contribution") -> Transaction:
        """Record a cash contribution to a portfolio"""
        
        transaction = Transaction(
            id=str(uuid.uuid4()),
            portfolio_id=portfolio_id,
            transaction_type=TransactionType.CONTRIBUTION,
            date=datetime.now(),
            amount=amount,
            description=description
        )
        
        self.storage.save_transaction(transaction)
        
        # Update portfolio cash balance
        portfolios = self.storage.load_portfolios()
        for portfolio in portfolios:
            if portfolio.id == portfolio_id:
                portfolio.cash_balance += amount
                portfolio.total_value += amount
                portfolio.last_updated = datetime.now()
                break
        
        return transaction 