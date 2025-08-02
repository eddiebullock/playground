"""
Goal calculator service for $1M target projections
"""

from datetime import datetime, date
from typing import Dict, List, Tuple
from decimal import Decimal
import math

from src.models.portfolio import Portfolio
from src.models.investment import Investment
from src.models.transaction import Transaction


class GoalCalculator:
    """Calculates projections toward $1M goal"""
    
    def __init__(self, target_amount: Decimal = Decimal('1000000')):
        self.target_amount = target_amount
    
    def calculate_current_status(self, portfolios: List[Portfolio]) -> Dict:
        """Calculate current status toward goal"""
        total_value = sum(p.total_value for p in portfolios)
        total_invested = sum(p.invested_amount for p in portfolios)
        total_return = sum(p.total_return for p in portfolios)
        
        progress_percentage = (total_value / self.target_amount) * 100
        remaining_amount = self.target_amount - total_value
        
        return {
            'current_value': total_value,
            'total_invested': total_invested,
            'total_return': total_return,
            'progress_percentage': progress_percentage,
            'remaining_amount': remaining_amount,
            'target_amount': self.target_amount
        }
    
    def calculate_monthly_projection(self, 
                                   current_value: Decimal,
                                   monthly_contribution: Decimal,
                                   annual_return_rate: Decimal = Decimal('0.08'),
                                   months: int = 360) -> List[Dict]:
        """Calculate monthly projections toward goal"""
        projections = []
        monthly_return_rate = annual_return_rate / 12
        
        current_amount = current_value
        
        for month in range(1, months + 1):
            # Add monthly contribution
            current_amount += monthly_contribution
            
            # Apply monthly return
            monthly_return = current_amount * monthly_return_rate
            current_amount += monthly_return
            
            # Check if we've reached the goal
            reached_goal = current_amount >= self.target_amount
            
            projection = {
                'month': month,
                'amount': current_amount,
                'contribution': monthly_contribution,
                'return': monthly_return,
                'reached_goal': reached_goal,
                'date': self._add_months(date.today(), month)
            }
            
            projections.append(projection)
            
            if reached_goal:
                break
        
        return projections
    
    def calculate_goal_reach_date(self,
                                 current_value: Decimal,
                                 monthly_contribution: Decimal,
                                 annual_return_rate: Decimal = Decimal('0.08')) -> Dict:
        """Calculate when you'll reach the $1M goal"""
        projections = self.calculate_monthly_projection(
            current_value, monthly_contribution, annual_return_rate
        )
        
        goal_reached = None
        for proj in projections:
            if proj['reached_goal']:
                goal_reached = proj
                break
        
        if goal_reached:
            return {
                'months_to_goal': goal_reached['month'],
                'years_to_goal': goal_reached['month'] / 12,
                'goal_date': goal_reached['date'],
                'final_amount': goal_reached['amount']
            }
        else:
            return {
                'months_to_goal': None,
                'years_to_goal': None,
                'goal_date': None,
                'final_amount': projections[-1]['amount'] if projections else current_value
            }
    
    def calculate_required_contribution(self,
                                      current_value: Decimal,
                                      target_date_years: int,
                                      annual_return_rate: Decimal = Decimal('0.08')) -> Decimal:
        """Calculate required monthly contribution to reach goal by target date"""
        target_months = target_date_years * 12
        monthly_return_rate = annual_return_rate / 12
        
        # Formula: PMT = (FV - PV * (1 + r)^n) / (((1 + r)^n - 1) / r)
        # Where: FV = future value (target), PV = present value, r = monthly rate, n = months
        
        future_value_factor = (1 + monthly_return_rate) ** target_months
        present_value_future = current_value * future_value_factor
        
        if monthly_return_rate == 0:
            # Simple case: no returns, just contributions
            required_contribution = (self.target_amount - current_value) / target_months
        else:
            # With returns
            required_contribution = (self.target_amount - present_value_future) / (
                (future_value_factor - 1) / monthly_return_rate
            )
        
        return max(required_contribution, Decimal('0'))
    
    def calculate_scenarios(self,
                           current_value: Decimal,
                           base_monthly_contribution: Decimal) -> Dict:
        """Calculate different scenarios for reaching the goal"""
        scenarios = {}
        
        # Scenario 1: Current contribution rate
        scenario1 = self.calculate_goal_reach_date(
            current_value, base_monthly_contribution
        )
        scenarios['current_rate'] = {
            'monthly_contribution': base_monthly_contribution,
            'projection': scenario1
        }
        
        # Scenario 2: Increased contribution (20% more)
        increased_contribution = base_monthly_contribution * Decimal('1.2')
        scenario2 = self.calculate_goal_reach_date(
            current_value, increased_contribution
        )
        scenarios['increased_rate'] = {
            'monthly_contribution': increased_contribution,
            'projection': scenario2
        }
        
        # Scenario 3: Target 10 years
        target_10_years = self.calculate_required_contribution(
            current_value, 10
        )
        scenario3 = self.calculate_goal_reach_date(
            current_value, target_10_years
        )
        scenarios['target_10_years'] = {
            'monthly_contribution': target_10_years,
            'projection': scenario3
        }
        
        # Scenario 4: Target 15 years
        target_15_years = self.calculate_required_contribution(
            current_value, 15
        )
        scenario4 = self.calculate_goal_reach_date(
            current_value, target_15_years
        )
        scenarios['target_15_years'] = {
            'monthly_contribution': target_15_years,
            'projection': scenario4
        }
        
        return scenarios
    
    def _add_months(self, current_date: date, months: int) -> date:
        """Add months to a date"""
        year = current_date.year + (current_date.month + months - 1) // 12
        month = (current_date.month + months - 1) % 12 + 1
        return date(year, month, current_date.day)
    
    def format_currency(self, amount: Decimal) -> str:
        """Format decimal as currency string"""
        return f"${amount:,.2f}"
    
    def format_percentage(self, percentage: Decimal) -> str:
        """Format decimal as percentage string"""
        return f"{percentage:.2f}%" 