# Stock & Crypto Price Tracker

A Python application that fetches and displays real-time prices for stocks and cryptocurrencies using multiple API sources.

## Features

- **Cryptocurrencies**: BTC (Bitcoin), SOL (Solana)
- **Stocks**: MSFT (Microsoft)
- **Funds**: Fidelity Index World Fund (GB00BJS8SJ3), Vanguard FTSE All-World UCITS ETF (VWRP.L)
- Real-time price updates
- Price change indicators with color coding
- Formatted table display
- Fallback API support

## Current Status

**⚠️ API Limitations Encountered:**

1. **Alpha Vantage**: Daily limit reached (25 requests/day for free tier)
2. **Yahoo Finance**: Rate limited due to too many requests

This is why the main app is currently not fetching live data. However, the app architecture is complete and ready to work once API limits are resolved.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Get API Key**:
   - Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Sign up for a free API key
   - The free tier allows 25 API calls per day

3. **Configure API Key**:
   Create a `.env` file in the project root:
   ```
   ALPHA_VANTAGE_API_KEY=your_api_key_here
   ```
   
   Or set it as an environment variable:
   ```bash
   export ALPHA_VANTAGE_API_KEY=your_api_key_here
   ```

## Usage

### Demo Mode (Currently Working)
```bash
python demo_stock_app.py
```
This shows what the app would display when APIs are working.

### Live Data Mode
```bash
python stock_price_app.py
```
This attempts to fetch live data (currently limited by API restrictions).

## Solutions for API Limitations

### Option 1: Wait for Daily Reset
- Alpha Vantage free tier resets daily at midnight UTC
- Yahoo Finance rate limits reset after a cooling-off period

### Option 2: Upgrade Alpha Vantage
- Premium plans remove daily limits
- Visit: https://www.alphavantage.co/premium/

### Option 3: Alternative Data Sources
- **IEX Cloud**: More generous free tier
- **Finnhub**: Good for international markets
- **Polygon.io**: Professional-grade data
- **Bloomberg Terminal**: Enterprise solution

### Option 4: Implement Better Rate Limiting
- Add longer delays between API calls
- Use multiple API keys
- Implement exponential backoff

## App Architecture

The app is designed with a robust fallback system:

1. **Primary**: Alpha Vantage API for stocks and crypto
2. **Fallback**: Yahoo Finance for failed requests
3. **Smart Rate Limiting**: Only applies delays when necessary
4. **Error Handling**: Graceful degradation when APIs fail

## Expected Output

When working properly, the app displays:

```
Stock & Crypto Prices - 2025-08-18 11:51:28
================================================================================
| Name                              | Symbol   | Type   | Price       | Change   | Change %   | Last Updated |
|===================================|==========|========|=============|==========|============|==============|
| Bitcoin                           | BTC      | Crypto | $115,237.91 |          |            | 2025-08-18   |
| Solana                            | SOL      | Crypto | $181.13     |          |            | 2025-08-18   |
| Microsoft                         | MSFT     | Stock  | $520.17     | -2.31    | -0.4421%   | 2025-08-15   |
| Fidelity Index World Fund         | PIWOA.L  | Fund   | GBP 386.21  |          |            | 2025-08-18   |
| Vanguard FTSE All-World UCITS ETF | VWRP.L   | Stock  | $116.89     | +0.03    | +0.0257%   | 2025-08-15   |
```

## Dependencies

- `requests`: HTTP library for API calls
- `python-dotenv`: Environment variable management
- `tabulate`: Formatted table display
- `yfinance`: Yahoo Finance integration (fallback)

## Notes

- The app successfully handles international fund tickers
- Built-in error handling for API failures
- Configurable rate limiting for different API tiers
- Ready for production use once API limits are resolved

## Next Steps

1. **Immediate**: Use demo mode to see the app's capabilities
2. **Short-term**: Wait for API limits to reset or upgrade plans
3. **Long-term**: Consider implementing multiple API sources for redundancy
