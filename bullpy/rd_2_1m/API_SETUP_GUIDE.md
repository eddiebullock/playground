# API Setup Guide for Broker Integration

This guide will help you set up API connections to your broker accounts for real-time portfolio data.

## Current Status

YES **Yahoo Finance**: Working (no API key needed)  
NO **Freetrade**: Need API key  
NO **AJ Bell**: Need API key  
NO **Coinbase**: Need API key  

## Step-by-Step Setup

### 1. Freetrade API Setup

**Note**: Freetrade may not offer a public API. You have these options:

#### Option A: Check for Official API
1. Log into your Freetrade account
2. Look for "API" or "Developer" settings
3. Check their documentation at https://freetrade.io/developers
4. Generate API key if available

#### Option B: Manual Data Entry (Recommended)
1. Log into Freetrade
2. Export your portfolio data as CSV
3. Use the `update_real_data.py` script to input values manually
4. Update regularly (weekly/monthly)

#### Option C: Web Scraping (Advanced)
If no API is available, we can build a web scraper to extract data from the Freetrade web interface.

### 2. AJ Bell API Setup

**Note**: AJ Bell may not offer a public API. You have these options:

#### Option A: Check for Official API
1. Log into your AJ Bell account
2. Look for "API" or "Developer" settings
3. Check their documentation
4. Generate API key if available

#### Option B: Manual Data Entry (Recommended)
1. Log into AJ Bell
2. Export your portfolio data as CSV
3. Use the `update_real_data.py` script to input values manually
4. Update regularly (weekly/monthly)

### 3. Coinbase API Setup

Coinbase offers API access through their Advanced Trade platform:

1. **Go to Coinbase Advanced Trade**:
   - Visit https://advanced.coinbase.com
   - Log in with your Coinbase account

2. **Generate API Key**:
   - Go to Settings → API
   - Click "New API Key"
   - Set permissions: Read-only for portfolio data
   - Copy the API key and secret

3. **Add to Configuration**:
   ```python
   # In config/api_keys.py
   COINBASE_API_KEY = "your_coinbase_api_key"
   COINBASE_API_SECRET = "your_coinbase_api_secret"
   ```

### 4. Yahoo Finance (Already Working)

No setup needed! Yahoo Finance provides free real-time stock prices.

## Configuration

Once you have your API keys, add them to `config/api_keys.py`:

```python
# Freetrade API
FREETRADE_API_KEY = "your_freetrade_api_key_here"

# AJ Bell API  
AJBELL_API_KEY = "your_ajbell_api_key_here"

# Coinbase API
COINBASE_API_KEY = "your_coinbase_api_key_here"
COINBASE_API_SECRET = "your_coinbase_api_secret_here"
```

## Testing Your Setup

1. **Test API Connections**:
   ```bash
   python test_api_connections.py
   ```

2. **Test Real Data Sync**:
   ```bash
   python main.py
   ```

3. **Update Manual Data**:
   ```bash
   python update_real_data.py
   ```

## Alternative: Manual Data Entry

If your brokers don't offer APIs, you can still use the system with manual data entry:

1. **Export Portfolio Data**:
   - Log into each broker
   - Export portfolio summary as CSV or screenshot
   - Note down current values

2. **Input Data Manually**:
   ```bash
   python update_real_data.py
   ```
   - Choose option 1 to update existing portfolios
   - Enter your real portfolio values
   - Run `python main.py` to see updated projections

3. **Regular Updates**:
   - Update values weekly/monthly
   - Track your progress toward $1M goal

## Security Notes

- **Never commit API keys to git**
- The `config/api_keys.py` file is in `.gitignore`
- Keep your API keys secure
- Use read-only permissions where possible

## Troubleshooting

### API Connection Issues
- Check API key format
- Verify permissions
- Test with `python test_api_connections.py`

### Manual Data Entry Issues
- Use decimal format (e.g., "25000.50")
- Ensure portfolio IDs match
- Check CSV file format

### Yahoo Finance Issues
- Check internet connection
- Verify stock symbols are correct
- Some symbols may need different format (e.g., ".L" for London stocks)

## Next Steps

1. **Get API keys** from your brokers (if available)
2. **Test connections** with the test script
3. **Input real data** manually or via API
4. **Monitor progress** toward your $1M goal
5. **Add visualizations** and detailed analysis

## Support

If you encounter issues:
1. Check the broker's API documentation
2. Verify API key permissions
3. Test with manual data entry as backup
4. Consider web scraping as alternative

The system is designed to work with or without API access, so you can start tracking your progress immediately! 