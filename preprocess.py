import os
import pandas as pd
import numpy as np

# ✅ Defined Directory Where CSVs Are Stored
DATA_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\data"

# ✅ List of All CSV Files (Luxury Watches Removed)
CSV_FILES = {
    "gold": "gold.csv",
    "real_estate": "real_estate.csv",
    "private_equity": "private_equity.csv",
    "green_energy": "green_energy.csv",
    "solar_energy": "solar_energy.csv",
    "infrastructure": "infrastructure.csv",
    "bitcoin": "bitcoin.csv",
    "ethereum": "ethereum.csv",
    "solana": "solana.csv",
    "avalanche": "avalanche-2.csv",
    "financial_news": "financial_news.csv",
    "ethereum_tx": "ethereum_0x742d35Cc6634C0532925a3b844Bc454e4438f44e.csv"
}

# ✅ Function: Load CSV File


def load_csv(filename):
    """Load a CSV file into a Pandas DataFrame."""
    file_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {filename}")
        return None
    df = pd.read_csv(file_path)
    print(
        f"✅ Loaded {filename} with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# ✅ Function: Standardize Date Format


def clean_date_column(df, column_name="timestamp"):
    """Convert date column to a standardized datetime format."""
    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
        # Remove rows where date is not parsed
        df = df.dropna(subset=[column_name])
        df = df.sort_values(by=column_name)  # Ensure chronological order
    return df

# ✅ Function: Handle Missing Values


def handle_missing_values(df):
    """Handle missing values by forward-filling or dropping."""
    df = df.ffill().bfill()  # Forward-fill & back-fill missing values
    return df

# ✅ Function: Convert Price Columns to Numeric


def convert_to_numeric(df, columns):
    """Convert specific columns to numeric data type."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ✅ Load & Clean Data (Luxury Watches Removed)
dataframes = {}

for asset, filename in CSV_FILES.items():
    df = load_csv(filename)
    if df is not None:
        # Standardize date format
        df = clean_date_column(
            df, "timestamp") if "timestamp" in df.columns else clean_date_column(df, "date")

        # Convert price & value columns to numeric
        df = convert_to_numeric(df, ["price", "value"])

        # Handle missing values
        df = handle_missing_values(df)

        # Store cleaned DataFrame
        dataframes[asset] = df

print("✅ All datasets loaded and cleaned successfully!")
