"""
investment portfolio tracker - road to 1m
"""

import sys
import os
from decimal import Decimal

# add src to python path 
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.storage import DataStorage
from data.portfolio_loader import PortfolioLoader
from services.goal_calculator import GoalCalculator
from services.portfolio_simulator import PortfolioSimulator
from services.market_data import MarketDataService

def main():
    """main application entry point"""
    print("investment portfolio tracker - road to 1m")
    print("=" * 50)
    
    # Initialize data storage and services
    storage = DataStorage()
    simulator = PortfolioSimulator(storage)
    calculator = GoalCalculator()
    market_data = MarketDataService()
    
    # Load portfolios
    portfolios = storage.load_portfolios()
    if not portfolios:
        print("No portfolio data found. Run 'python setup_real_portfolio.py' to set up your real data.")
        return
    
    # Update prices with real market data
    print("Updating prices with real market data...")
    simulator.update_all_prices()
    
    # Get portfolio summary
    summary = simulator.get_portfolio_summary()
    
    # Calculate current status
    current_status = calculator.calculate_current_status(portfolios)
    total_value = summary['total_value']
    
    print("current status:")
    print(f"total portfolio value: {calculator.format_currency(total_value)}")
    print(f"total invested: {calculator.format_currency(summary['total_invested'])}")
    print(f"total return: {calculator.format_currency(summary['total_return'])} ({summary['total_return_percentage']:.2f}%)")
    print(f"progress toward goal: {calculator.format_percentage(current_status['progress_percentage'])}")
    print(f"remaining to reach 1m: {calculator.format_currency(current_status['remaining_amount'])}")
    print("=" * 50)
    
    # Show portfolio breakdown
    print("portfolio breakdown:")
    for portfolio in summary['portfolios']:
        print(f"- {portfolio['name']} ({portfolio['broker']}): {calculator.format_currency(portfolio['total_value'])}")
    
    print("=" * 50)
    
    # Calculate goal projections
    monthly_contribution = Decimal('2000')
    projection = calculator.calculate_goal_reach_date(total_value, monthly_contribution)
    
    if projection['months_to_goal']:
        years = projection['months_to_goal'] / 12
        print(f"goal projection:")
        print(f"- with £{monthly_contribution} monthly contribution")
        print(f"- you'll reach 1m in {years:.1f} years")
        print(f"- target date: {projection['goal_date']}")
    else:
        print("goal projection:")
        print(f"- with £{monthly_contribution} monthly contribution")
        print("- you won't reach 1m in 30 years")
        print(f"- final amount after 30 years: {calculator.format_currency(projection['final_amount'])}")
    
    print("=" * 50)
    print("next steps:")
    print("1. edit setup_real_portfolio.py with your actual holdings and values")
    print("2. run 'python update_prices.py' to refresh market data")
    print("3. add monthly contributions to track progress")
    print("4. add more detailed analysis and visualizations")

if __name__ == "__main__":
    main()