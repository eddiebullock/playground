import os
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not ALPHA_VANTAGE_API_KEY:
    print("Warning: ALPHA_VANTAGE_API_KEY not found in environment variables.")
    print("Please set your Alpha Vantage API key in a .env file or as an environment variable.")
    print("You can get a free API key from: https://www.alphavantage.co/support/#api-key")
