import os
import sqlite3
import pandas as pd

# ✅ Defined Paths
DB_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\database"
DATA_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\processed_data"
os.makedirs(DB_PATH, exist_ok=True)

DB_FILE = os.path.join(DB_PATH, "alternative_assets.db")

# ✅ Step 1: Connected to SQLite Database


def create_connection():
    """Create or connect to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    return conn

# ✅ Step 2: Createed Tables


def create_tables():
    """Define tables for assets, crypto, and sentiment data."""
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        asset TEXT,
        price REAL,
        RSI REAL,
        moving_avg_7d REAL,
        moving_avg_30d REAL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crypto (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        asset TEXT,
        price REAL,
        transaction_count INTEGER,
        daily_volume REAL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        sentiment_score REAL,
        sentiment_label TEXT
    );
    """)

    conn.commit()
    conn.close()
    print("✅ Tables created successfully!")

# ✅ Step 3: Insert Data into Database


def insert_data():
    """Load preprocessed data from CSV and insert into SQLite tables."""
    conn = create_connection()
    cursor = conn.cursor()

    file_path = os.path.join(DATA_PATH, "final_dataset.csv")
    if not os.path.exists(file_path):
        print("❌ Error: Processed dataset not found!")
        return

    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(
        str)  # Convert timestamp to string format for SQL

    # ✅ Rename columns to match database schema
    df.rename(columns={"7_day_MA": "moving_avg_7d",
              "30_day_MA": "moving_avg_30d"}, inplace=True)

    # ✅ Insert Asset Data
    asset_columns = ["timestamp", "asset", "price",
                     "RSI", "moving_avg_7d", "moving_avg_30d"]
    asset_data = df[asset_columns].dropna()
    asset_data.to_sql("assets", conn, if_exists="append", index=False)
    print(f"✅ Inserted {len(asset_data)} asset records.")

    # ✅ Insert Crypto Data (Allow Missing Values)
    crypto_columns = ["timestamp", "asset", "price",
                      "transaction_count", "daily_volume"]
    if all(col in df.columns for col in crypto_columns):  # Check if columns exist
        crypto_data = df[crypto_columns].copy()

        # Fill missing transaction count and daily volume with 0
        crypto_data["transaction_count"] = crypto_data["transaction_count"].fillna(
            0).astype(int)
        crypto_data["daily_volume"] = crypto_data["daily_volume"].fillna(0)

        # Insert only crypto-related assets
        crypto_assets = ["bitcoin", "ethereum", "solana", "avalanche-2"]
        crypto_data = crypto_data[crypto_data["asset"].isin(crypto_assets)]

        crypto_data.to_sql("crypto", conn, if_exists="append", index=False)
        print(f"✅ Inserted {len(crypto_data)} crypto records.")
    else:
        print("⚠️ Warning: Crypto columns missing from dataset, skipping crypto data insertion.")

    # ✅ Insert Sentiment Data
    sentiment_columns = ["timestamp", "sentiment_score"]
    if all(col in df.columns for col in sentiment_columns):
        sentiment_data = df[sentiment_columns].dropna()
        sentiment_data["sentiment_label"] = sentiment_data["sentiment_score"].apply(
            lambda x: "positive" if x > 0.05 else "negative" if x < -0.05 else "neutral"
        )
        sentiment_data.to_sql(
            "sentiment", conn, if_exists="append", index=False)
        print(f"✅ Inserted {len(sentiment_data)} sentiment records.")
    else:
        print("⚠️ Warning: Sentiment columns missing from dataset, skipping sentiment data insertion.")

    conn.commit()
    conn.close()
    print("✅ Data inserted successfully!")

# ✅ Step 4: Test Queries


def test_queries():
    """Run test queries to verify data storage."""
    conn = create_connection()
    cursor = conn.cursor()

    print("\n🔍 Testing Queries...")

    # ✅ Check Asset Data
    cursor.execute("SELECT * FROM assets LIMIT 5;")
    print("📊 Sample Asset Data:", cursor.fetchall())

    # ✅ Check Crypto Data
    cursor.execute("SELECT * FROM crypto LIMIT 5;")
    print("📊 Sample Crypto Data:", cursor.fetchall())

    # ✅ Check Sentiment Data
    cursor.execute("SELECT * FROM sentiment LIMIT 5;")
    print("📊 Sample Sentiment Data:", cursor.fetchall())

    conn.close()


# ✅ Run Database Operations
if __name__ == "__main__":
    create_tables()
    insert_data()
    test_queries()
