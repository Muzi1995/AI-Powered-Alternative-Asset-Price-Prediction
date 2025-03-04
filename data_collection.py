import os
from dotenv import load_dotenv
import time
import requests
import pandas as pd
import yfinance as yf

load_dotenv()

# ✅ Access API keys securely
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

# ✅ Defined Storage Path
SAVE_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\data"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ Alternative Assets (Yahoo Finance - 5 Years Data)
ASSET_SYMBOLS = {
    "gold": "GLD",
    "real_estate": "VNQ",
    "private_equity": "PSP",
    "green_energy": "ICLN",
    "solar_energy": "TAN",
    "infrastructure": "IGF"
}

# ✅ Cryptocurrencies (CoinGecko API - 1 Year Data)
CRYPTO_ASSETS = ["bitcoin", "ethereum", "solana", "avalanche-2"]


# ✅ Function: Fetch Yahoo Finance Data (5 Years)


def fetch_asset_data(asset_name, asset_symbol, start_date, end_date):
    try:
        asset = yf.Ticker(asset_symbol)
        df = asset.history(start=start_date, end=end_date)
        if df.empty:
            print(f"❌ No data found for {asset_name} ({asset_symbol}).")
            return None

        file_path = os.path.join(SAVE_PATH, f"{asset_name}.csv")
        df.to_csv(file_path)
        print(f"✅ {asset_name} data saved at {file_path}!")
        return df
    except Exception as e:
        print(f"⚠️ Error fetching {asset_name}: {e}")
        return None

# ✅ Function: Fetch Ethereum Transactions (Etherscan)


def fetch_etherscan_transactions(wallet_address):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": wallet_address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }

    response = requests.get(url, params=params)
    try:
        data = response.json()
        if data["status"] != "1":
            print(f"❌ API Error: {data.get('message', 'Unknown Error')}")
            return None

        df = pd.DataFrame(data["result"])
        file_path = os.path.join(SAVE_PATH, f"ethereum_{wallet_address}.csv")
        df.to_csv(file_path, index=False)
        print(f"✅ Ethereum transactions saved at {file_path}!")
        time.sleep(5)  # Prevent API rate limits
        return df
    except Exception as e:
        print(f"⚠️ JSON Parsing Error: {e}")
        return None


# ✅ Running Data Collection
if __name__ == "__main__":
    start_date = "2019-01-01"
    end_date = "2024-01-01"

    # Fetch Traditional Alternative Assets
    for asset_name, symbol in ASSET_SYMBOLS.items():
        fetch_asset_data(asset_name, symbol, start_date, end_date)

    # Fetch Ethereum Transactions (Replace with another wallet if needed)
    fetch_etherscan_transactions("0x742d35Cc6634C0532925a3b844Bc454e4438f44e")
