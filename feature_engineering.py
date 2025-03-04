import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# ✅ Ensured necessary nltk data is downloaded
nltk.download("vader_lexicon")

# ✅ Defined Data Paths
DATA_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\data"
SAVE_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\processed_data"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ Loaded Dataset with Fixes


def load_csv(filename):
    """Load a CSV file into Pandas DataFrame and standardize column names."""
    file_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {filename}")
        return None

    df = pd.read_csv(file_path)

    # ✅ Standardized timestamp column
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", utc=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", utc=True)
    elif "timeStamp" in df.columns:  # ✅ Fix for blockchain transactions
        df.rename(columns={"timeStamp": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", unit="s", utc=True)
    else:
        print(f"❌ Error: No timestamp column found in {filename}")
        return None

    # ✅ Standardized "price" column
    if "price" not in df.columns:
        if "Close" in df.columns:
            df.rename(columns={"Close": "price"}, inplace=True)
        else:
            print(
                f"⚠️ Warning: No 'price' column found in {filename}. Skipping price-based calculations.")

    return df

# ✅ Step 1: Generated Market Volatility Indicators


def add_volatility_features(df):
    """Compute Moving Averages, RSI, and Bollinger Bands, but only if 'price' exists."""
    if "price" not in df.columns:
        print("⚠️ Skipping volatility indicators: 'price' column missing.")
        return df  # Return unchanged DataFrame

    df["7_day_MA"] = df["price"].rolling(window=7).mean()
    df["30_day_MA"] = df["price"].rolling(window=30).mean()

    # ✅ Computed RSI (Relative Strength Index)
    delta = df["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

# ✅ Step 2: Computed Blockchain Transaction Metrics


def process_blockchain_data(df):
    """Compute daily transaction count and total transaction volume."""
    if "value" not in df.columns or "hash" not in df.columns:
        print("⚠️ Skipping blockchain processing: required columns missing.")
        return df

    df["value"] = df["value"].astype(float) / 10**18  # Convert Wei to ETH
    df["transaction_count"] = df.groupby(df["timestamp"].dt.date)[
        "hash"].transform("count")
    df["daily_volume"] = df.groupby(df["timestamp"].dt.date)[
        "value"].transform("sum")

    df = df[["timestamp", "transaction_count", "daily_volume"]
            ].drop_duplicates().reset_index(drop=True)
    return df

# ✅ Step 3: Performed Sentiment Analysis


def analyze_sentiment(df):
    """Compute sentiment scores for financial news articles."""
    analyzer = SentimentIntensityAnalyzer()

    if "Title" not in df.columns:
        print("⚠️ Skipping sentiment analysis: 'Title' column missing.")
        return df

    df["sentiment_score"] = df["Title"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"])
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda score: "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True)

    return df[["timestamp", "sentiment_score"]]

# ✅ Step 4: Merged All Features into a Final Dataset


def merge_all_features():
    """Merge financial, blockchain, and sentiment features into a single dataset."""
    assets = ["gold", "real_estate", "private_equity", "green_energy", "solar_energy", "infrastructure",
              "bitcoin", "ethereum", "solana", "avalanche-2"]

    financial_data = []
    for asset in assets:
        df = load_csv(f"{asset}.csv")
        if df is not None:
            df = add_volatility_features(df)
            df["asset"] = asset  # Label dataset with asset name
            financial_data.append(df)

    # ✅ Merged all financial data
    final_df = pd.concat(financial_data, ignore_index=True)

    # ✅ Merged blockchain transactions
    blockchain_df = load_csv(
        "ethereum_0x742d35Cc6634C0532925a3b844Bc454e4438f44e.csv")
    if blockchain_df is not None:
        blockchain_df = process_blockchain_data(blockchain_df)
        final_df = final_df.merge(blockchain_df, on="timestamp", how="left")

    # ✅ Merged financial sentiment data
    sentiment_df = load_csv("financial_news.csv")
    if sentiment_df is not None:
        sentiment_df = analyze_sentiment(sentiment_df)
        # ✅ Ensure timestamps are in UTC before merging
        final_df["timestamp"] = pd.to_datetime(
            final_df["timestamp"], errors="coerce", utc=True)
        sentiment_df["timestamp"] = pd.to_datetime(
            sentiment_df["timestamp"], errors="coerce", utc=True)
        final_df = final_df.merge(sentiment_df, on="timestamp", how="left")

    # ✅ Handled missing values
    final_df.ffill(inplace=True)

    # ✅ Normalized numerical features
    scaler = MinMaxScaler()
    numeric_columns = ["price", "7_day_MA", "30_day_MA", "RSI",
                       "transaction_count", "daily_volume", "sentiment_score"]

    # Ignored missing columns
    # Ensured columns exist
    numeric_columns = [
        col for col in numeric_columns if col in final_df.columns]

    # ✅ Dropped columns that contain only NaN values
    valid_numeric_columns = [
        col for col in numeric_columns if not final_df[col].isna().all()]

    if valid_numeric_columns:
        scaler = MinMaxScaler()
        final_df[valid_numeric_columns] = scaler.fit_transform(
            final_df[valid_numeric_columns])
    else:
        print("⚠️ Warning: No valid numeric columns for normalization.")

    # ✅ Saved processed dataset
    file_path = os.path.join(SAVE_PATH, "final_dataset.csv")
    final_df.to_csv(file_path, index=False)
    print(f"✅ Final dataset saved at {file_path}!")


# ✅ Run Feature Engineering
if __name__ == "__main__":
    merge_all_features()
