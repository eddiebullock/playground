# Investment Portfolio Tracker - Road to $1M

A Python application to track investment portfolios, predict future performance, and ensure you're on track to reach $1M as quickly as possible.

## Project Goals

- Track investments across 3 different portfolios
- Monitor performance and calculate returns
- Predict future portfolio value based on contributions and market performance
- Ensure consistent progress toward $1M goal
- Provide insights and recommendations for optimization

## Current Status

- **Current Capital**: $44,000
- **Monthly Contribution**: $2,000
- **Target**: $1,000,000

## Project Structure

```
rd_2_1m/
├── src/                 # Main application code
│   ├── models/         # Data models (Portfolio, Investment, etc.)
│   ├── services/       # Business logic (calculations, predictions)
│   ├── data/           # Data handling and storage
│   └── utils/          # Helper functions
├── tests/              # Unit tests
├── data/               # Data storage (CSV files, etc.)
├── config/             # Configuration files
└── main.py             # Application entry point
```

## Learning Objectives

1. **Data Modeling**: Create robust data models for portfolios and investments
2. **Financial Calculations**: Implement compound interest, returns, and projections
3. **Data Visualization**: Create charts and dashboards for portfolio analysis
4. **API Integration**: Fetch real-time market data
5. **Testing**: Write comprehensive unit tests
6. **Configuration Management**: Handle different environments and settings

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your configuration in `config/`

3. Run the application:
   ```bash
   python main.py
   ```

## Development Workflow

1. Start with data models in `src/models/`
2. Implement core calculations in `src/services/`
3. Add data handling in `src/data/`
4. Create visualizations and reports
5. Write tests for each component
6. Iterate and improve based on insights

## Key Features to Implement

- [ ] Portfolio data models
- [ ] Investment tracking across 3 portfolios
- [ ] Performance calculations (returns, volatility)
- [ ] Future value projections
- [ ] Goal tracking toward $1M
- [ ] Monthly contribution planning
- [ ] Risk analysis
- [ ] Visualization dashboard
- [ ] Export reports 