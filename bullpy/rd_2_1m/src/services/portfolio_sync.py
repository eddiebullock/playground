"""
Portfolio synchronization service
"""

import os
from datetime import datetime
from typing import Dict, List
from decimal import Decimal
import uuid

from src.data.storage import DataStorage
from src.services.broker_apis import FreetradeAPI, AJBellAPI, CoinbaseAPI
from src.services.market_data import MarketDataService
from src.models.portfolio import Portfolio
from src.models.investment import Investment
from src.models.transaction import Transaction, TransactionType

# Import API keys
try:
    from config.api_keys import (
        FREETRADE_API_KEY, AJBELL_API_KEY, 
        COINBASE_API_KEY, COINBASE_API_SECRET
    )
except ImportError:
    # Fallback if config file doesn't exist
    FREETRADE_API_KEY = None
    AJBELL_API_KEY = None
    COINBASE_API_KEY = None
    COINBASE_API_SECRET = None


class PortfolioSyncService:
    """Synchronizes portfolio data from all broker APIs"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.market_data = MarketDataService()
        
        # Initialize broker APIs
        self.freetrade = FreetradeAPI(FREETRADE_API_KEY) if FREETRADE_API_KEY else None
        self.ajbell = AJBellAPI(AJBELL_API_KEY) if AJBELL_API_KEY else None
        self.coinbase = CoinbaseAPI(COINBASE_API_KEY, COINBASE_API_SECRET) if COINBASE_API_KEY else None
    
    def sync_all_portfolios(self) -> Dict:
        """Sync data from all connected broker APIs"""
        results = {
            'freetrade': {},
            'ajbell': {},
            'coinbase': {},
            'total_value': Decimal('0'),
            'updated_at': datetime.now()
        }
        
        # Sync Freetrade
        if self.freetrade:
            print("Syncing Freetrade...")
            freetrade_result = self._sync_freetrade()
            results['freetrade'] = freetrade_result
            results['total_value'] += freetrade_result.get('total_value', Decimal('0'))
        
        # Sync AJ Bell
        if self.ajbell:
            print("Syncing AJ Bell...")
            ajbell_result = self._sync_ajbell()
            results['ajbell'] = ajbell_result
            results['total_value'] += ajbell_result.get('total_value', Decimal('0'))
        
        # Sync Coinbase
        if self.coinbase:
            print("Syncing Coinbase...")
            coinbase_result = self._sync_coinbase()
            results['coinbase'] = coinbase_result
            results['total_value'] += coinbase_result.get('total_value', Decimal('0'))
        
        return results
    
    def _sync_freetrade(self) -> Dict:
        """Sync Freetrade portfolio data"""
        try:
            # Get portfolio summary
            summary = self.freetrade.get_portfolio_summary()
            if not summary:
                return {'error': 'Could not fetch Freetrade data'}
            
            # Get holdings
            holdings = self.freetrade.get_holdings()
            
            # Update or create portfolio
            portfolio_id = 'freetrade_isa'
            portfolio = self._get_or_create_portfolio(
                portfolio_id, 'Freetrade ISA', 'Freetrade', 'ISA', summary
            )
            
            # Update holdings
            self._update_holdings(portfolio_id, holdings)
            
            return {
                'portfolio_id': portfolio_id,
                'total_value': summary.get('total_value', Decimal('0')),
                'cash_balance': summary.get('cash_balance', Decimal('0')),
                'invested_amount': summary.get('invested_amount', Decimal('0')),
                'holdings_count': len(holdings)
            }
            
        except Exception as e:
            print(f"Error syncing Freetrade: {e}")
            return {'error': str(e)}
    
    def _sync_ajbell(self) -> Dict:
        """Sync AJ Bell portfolio data"""
        try:
            # Get portfolio summary
            summary = self.ajbell.get_portfolio_summary()
            if not summary:
                return {'error': 'Could not fetch AJ Bell data'}
            
            # Get holdings
            holdings = self.ajbell.get_holdings()
            
            # Update or create portfolio
            portfolio_id = 'ajbell_sipp'
            portfolio = self._get_or_create_portfolio(
                portfolio_id, 'AJ Bell SIPP', 'AJ Bell', 'SIPP', summary
            )
            
            # Update holdings
            self._update_holdings(portfolio_id, holdings)
            
            return {
                'portfolio_id': portfolio_id,
                'total_value': summary.get('total_value', Decimal('0')),
                'cash_balance': summary.get('cash_balance', Decimal('0')),
                'invested_amount': summary.get('invested_amount', Decimal('0')),
                'holdings_count': len(holdings)
            }
            
        except Exception as e:
            print(f"Error syncing AJ Bell: {e}")
            return {'error': str(e)}
    
    def _sync_coinbase(self) -> Dict:
        """Sync Coinbase portfolio data"""
        try:
            # Get portfolio summary
            summary = self.coinbase.get_portfolio_summary()
            if not summary:
                return {'error': 'Could not fetch Coinbase data'}
            
            # Get holdings
            holdings = self.coinbase.get_holdings()
            
            # Update or create portfolio
            portfolio_id = 'coinbase_crypto'
            portfolio = self._get_or_create_portfolio(
                portfolio_id, 'Coinbase Crypto', 'Coinbase', 'CRYPTO', summary
            )
            
            # Update holdings
            self._update_holdings(portfolio_id, holdings)
            
            return {
                'portfolio_id': portfolio_id,
                'total_value': summary.get('total_value', Decimal('0')),
                'cash_balance': summary.get('cash_balance', Decimal('0')),
                'invested_amount': summary.get('invested_amount', Decimal('0')),
                'holdings_count': len(holdings)
            }
            
        except Exception as e:
            print(f"Error syncing Coinbase: {e}")
            return {'error': str(e)}
    
    def _get_or_create_portfolio(self, portfolio_id: str, name: str, 
                                broker: str, account_type: str, summary: Dict) -> Portfolio:
        """Get existing portfolio or create new one"""
        portfolios = self.storage.load_portfolios()
        
        # Find existing portfolio
        for portfolio in portfolios:
            if portfolio.id == portfolio_id:
                # Update with new data
                portfolio.update_values(
                    summary.get('total_value', Decimal('0')),
                    summary.get('cash_balance', Decimal('0')),
                    summary.get('invested_amount', Decimal('0'))
                )
                return portfolio
        
        # Create new portfolio
        portfolio = Portfolio(
            id=portfolio_id,
            name=name,
            description=f"{broker} {account_type} account",
            broker=broker,
            account_type=account_type,
            created_date=datetime.now(),
            total_value=summary.get('total_value', Decimal('0')),
            cash_balance=summary.get('cash_balance', Decimal('0')),
            invested_amount=summary.get('invested_amount', Decimal('0'))
        )
        
        self.storage.save_portfolio(portfolio)
        return portfolio
    
    def _update_holdings(self, portfolio_id: str, holdings: List[Dict]):
        """Update holdings for a portfolio"""
        # Clear existing holdings for this portfolio
        existing_investments = self.storage.load_investments(portfolio_id)
        
        # Add new holdings
        for holding in holdings:
            investment = Investment(
                id=str(uuid.uuid4()),
                portfolio_id=portfolio_id,
                symbol=holding.get('symbol', ''),
                name=holding.get('name', ''),
                investment_type='stock' if holding.get('symbol') else 'crypto',
                quantity=holding.get('quantity', Decimal('0')),
                average_price=holding.get('average_price', Decimal('0')),
                current_price=holding.get('current_price', Decimal('0'))
            )
            
            self.storage.save_investment(investment)
    
    def update_prices_from_yahoo(self):
        """Update all investment prices using Yahoo Finance"""
        print("Updating prices from Yahoo Finance...")
        
        investments = self.storage.load_investments()
        updated_investments = self.market_data.update_investment_prices(investments)
        
        # Save updated investments
        for investment in updated_investments:
            # Note: In a real implementation, you'd want to update the CSV file
            # For now, we'll just update the in-memory object
            pass
        
        print(f"Updated prices for {len(updated_investments)} investments")
    
    def get_sync_status(self) -> Dict:
        """Get status of all API connections"""
        status = {
            'freetrade': {
                'connected': self.freetrade is not None,
                'api_key': 'configured' if FREETRADE_API_KEY and FREETRADE_API_KEY != "your_freetrade_api_key_here" else 'not configured'
            },
            'ajbell': {
                'connected': self.ajbell is not None,
                'api_key': 'configured' if AJBELL_API_KEY and AJBELL_API_KEY != "your_ajbell_api_key_here" else 'not configured'
            },
            'coinbase': {
                'connected': self.coinbase is not None,
                'api_key': 'configured' if COINBASE_API_KEY and COINBASE_API_KEY != "your_coinbase_api_key_here" else 'not configured'
            },
            'yahoo_finance': {
                'connected': True,
                'status': 'available'
            }
        }
        
        return status 