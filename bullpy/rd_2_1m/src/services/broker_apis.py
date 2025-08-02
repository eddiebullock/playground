"""
Broker API integrations for portfolio data
"""

import requests
import json
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
import os
from abc import ABC, abstractmethod

from src.models.portfolio import Portfolio
from src.models.investment import Investment
from src.models.transaction import Transaction, TransactionType


class BrokerAPI(ABC):
    """Abstract base class for broker APIs"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
    
    @abstractmethod
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from broker"""
        pass
    
    @abstractmethod
    def get_holdings(self) -> List[Dict]:
        """Get current holdings from broker"""
        pass
    
    @abstractmethod
    def get_transactions(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get transaction history from broker"""
        pass


class FreetradeAPI(BrokerAPI):
    """Freetrade API integration"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.freetrade.io"
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from Freetrade"""
        try:
            # Note: This is a placeholder. You'll need to check Freetrade's actual API
            response = self.session.get(f"{self.base_url}/v1/accounts", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'total_value': Decimal(str(data.get('total_value', 0))),
                    'cash_balance': Decimal(str(data.get('cash_balance', 0))),
                    'invested_amount': Decimal(str(data.get('invested_amount', 0))),
                    'account_type': data.get('account_type', 'ISA'),
                    'broker': 'Freetrade'
                }
            else:
                print(f"Freetrade API error: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error connecting to Freetrade API: {e}")
            return {}
    
    def get_holdings(self) -> List[Dict]:
        """Get current holdings from Freetrade"""
        try:
            response = self.session.get(f"{self.base_url}/v1/accounts/positions", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                holdings = []
                
                for position in data.get('positions', []):
                    holdings.append({
                        'symbol': position.get('symbol'),
                        'name': position.get('name'),
                        'quantity': Decimal(str(position.get('quantity', 0))),
                        'average_price': Decimal(str(position.get('average_price', 0))),
                        'current_price': Decimal(str(position.get('current_price', 0))),
                        'total_cost': Decimal(str(position.get('total_cost', 0))),
                        'current_value': Decimal(str(position.get('current_value', 0)))
                    })
                
                return holdings
            else:
                print(f"Freetrade API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching Freetrade holdings: {e}")
            return []
    
    def get_transactions(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get transaction history from Freetrade"""
        try:
            params = {}
            if start_date:
                params['from'] = start_date
            if end_date:
                params['to'] = end_date
            
            response = self.session.get(f"{self.base_url}/v1/accounts/transactions", 
                                      headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                transactions = []
                
                for txn in data.get('transactions', []):
                    transactions.append({
                        'id': txn.get('id'),
                        'date': txn.get('date'),
                        'type': txn.get('type'),
                        'symbol': txn.get('symbol'),
                        'quantity': Decimal(str(txn.get('quantity', 0))),
                        'price': Decimal(str(txn.get('price', 0))),
                        'amount': Decimal(str(txn.get('amount', 0))),
                        'description': txn.get('description', '')
                    })
                
                return transactions
            else:
                print(f"Freetrade API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching Freetrade transactions: {e}")
            return []


class AJBellAPI(BrokerAPI):
    """AJ Bell API integration"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.ajbell.co.uk"
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from AJ Bell"""
        try:
            # Note: This is a placeholder. You'll need to check AJ Bell's actual API
            response = self.session.get(f"{self.base_url}/v1/accounts", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'total_value': Decimal(str(data.get('total_value', 0))),
                    'cash_balance': Decimal(str(data.get('cash_balance', 0))),
                    'invested_amount': Decimal(str(data.get('invested_amount', 0))),
                    'account_type': data.get('account_type', 'SIPP'),
                    'broker': 'AJ Bell'
                }
            else:
                print(f"AJ Bell API error: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error connecting to AJ Bell API: {e}")
            return {}
    
    def get_holdings(self) -> List[Dict]:
        """Get current holdings from AJ Bell"""
        try:
            response = self.session.get(f"{self.base_url}/v1/accounts/holdings", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                holdings = []
                
                for position in data.get('holdings', []):
                    holdings.append({
                        'symbol': position.get('symbol'),
                        'name': position.get('name'),
                        'quantity': Decimal(str(position.get('quantity', 0))),
                        'average_price': Decimal(str(position.get('average_price', 0))),
                        'current_price': Decimal(str(position.get('current_price', 0))),
                        'total_cost': Decimal(str(position.get('total_cost', 0))),
                        'current_value': Decimal(str(position.get('current_value', 0)))
                    })
                
                return holdings
            else:
                print(f"AJ Bell API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching AJ Bell holdings: {e}")
            return []
    
    def get_transactions(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get transaction history from AJ Bell"""
        try:
            params = {}
            if start_date:
                params['from'] = start_date
            if end_date:
                params['to'] = end_date
            
            response = self.session.get(f"{self.base_url}/v1/accounts/transactions", 
                                      headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                transactions = []
                
                for txn in data.get('transactions', []):
                    transactions.append({
                        'id': txn.get('id'),
                        'date': txn.get('date'),
                        'type': txn.get('type'),
                        'symbol': txn.get('symbol'),
                        'quantity': Decimal(str(txn.get('quantity', 0))),
                        'price': Decimal(str(txn.get('price', 0))),
                        'amount': Decimal(str(txn.get('amount', 0))),
                        'description': txn.get('description', '')
                    })
                
                return transactions
            else:
                print(f"AJ Bell API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching AJ Bell transactions: {e}")
            return []


class CoinbaseAPI(BrokerAPI):
    """Coinbase API integration"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__(api_key, api_secret)
        self.base_url = "https://api.coinbase.com/v2"
        self.headers = {
            'CB-ACCESS-KEY': api_key,
            'CB-ACCESS-SIGN': '',  # You'll need to generate this
            'CB-ACCESS-TIMESTAMP': '',  # You'll need to generate this
            'Content-Type': 'application/json'
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from Coinbase"""
        try:
            response = self.session.get(f"{self.base_url}/accounts", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                total_value = Decimal('0')
                cash_balance = Decimal('0')
                
                for account in data.get('data', []):
                    balance = Decimal(str(account.get('balance', {}).get('amount', 0)))
                    currency = account.get('balance', {}).get('currency', 'USD')
                    
                    if currency == 'USD':
                        cash_balance += balance
                    else:
                        # For crypto, you'd need to convert to USD
                        # This is simplified - you'd need real-time conversion rates
                        total_value += balance
                
                return {
                    'total_value': total_value,
                    'cash_balance': cash_balance,
                    'invested_amount': total_value,
                    'account_type': 'CRYPTO',
                    'broker': 'Coinbase'
                }
            else:
                print(f"Coinbase API error: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error connecting to Coinbase API: {e}")
            return {}
    
    def get_holdings(self) -> List[Dict]:
        """Get current holdings from Coinbase"""
        try:
            response = self.session.get(f"{self.base_url}/accounts", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                holdings = []
                
                for account in data.get('data', []):
                    currency = account.get('balance', {}).get('currency', '')
                    if currency != 'USD':  # Skip USD accounts
                        holdings.append({
                            'symbol': currency,
                            'name': f"{currency} (Crypto)",
                            'quantity': Decimal(str(account.get('balance', {}).get('amount', 0))),
                            'average_price': Decimal('0'),  # Would need historical data
                            'current_price': Decimal('0'),  # Would need real-time rates
                            'total_cost': Decimal('0'),
                            'current_value': Decimal('0')
                        })
                
                return holdings
            else:
                print(f"Coinbase API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching Coinbase holdings: {e}")
            return []
    
    def get_transactions(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get transaction history from Coinbase"""
        try:
            params = {}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            response = self.session.get(f"{self.base_url}/accounts/transactions", 
                                      headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                transactions = []
                
                for txn in data.get('data', []):
                    transactions.append({
                        'id': txn.get('id'),
                        'date': txn.get('created_at'),
                        'type': txn.get('type'),
                        'symbol': txn.get('currency'),
                        'quantity': Decimal(str(txn.get('amount', {}).get('amount', 0))),
                        'price': Decimal('0'),
                        'amount': Decimal(str(txn.get('amount', {}).get('amount', 0))),
                        'description': txn.get('description', '')
                    })
                
                return transactions
            else:
                print(f"Coinbase API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching Coinbase transactions: {e}")
            return [] 