"""
Configuration settings for the Investment Portfolio Tracker
"""

# Investment Goals
TARGET_AMOUNT = 1_000_000  # $1M target
CURRENT_CAPITAL = 44_000    # Current portfolio value
MONTHLY_CONTRIBUTION = 2_000 # Monthly savings contribution

# Portfolio Configuration
PORTFOLIOS = {
    "portfolio_1": "Main Investment Portfolio",
    "portfolio_2": "Retirement Account", 
    "portfolio_3": "Alternative Investments"
}

# Investment Strategy
INVESTMENT_STRATEGY = {
    "global_tracker_allocation": 0.8,  # 80% in global tracker
    "individual_stocks_allocation": 0.2,  # 20% in individual stocks
    "expected_annual_return": 0.08,  # 8% expected annual return
    "volatility": 0.15  # 15% annual volatility
}

# Data Storage
DATA_DIR = "data"
CSV_FILES = {
    "portfolios": "portfolios.csv",
    "investments": "investments.csv", 
    "transactions": "transactions.csv",
    "performance": "performance.csv"
}

# API Configuration (for future use)
MARKET_DATA_API = {
    "provider": "yfinance",
    "update_frequency": "daily"
} 