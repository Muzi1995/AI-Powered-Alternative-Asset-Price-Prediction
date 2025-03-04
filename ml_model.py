import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# ✅ Define Paths
DATA_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\processed_data"
SAVE_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\models"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ Step 1: Load Cleaned Dataset


def load_data():
    """Load the cleaned dataset from feature engineering."""
    file_path = os.path.join(DATA_PATH, "final_dataset.csv")

    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return None

    df = pd.read_csv(file_path)

    # ✅ Ensure timestamp format is correct
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(
        f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# ✅ Step 2: Select Features (X) and Target (y)


def prepare_features(df):
    """Select relevant features and define the target variable."""
    feature_columns = ["7_day_MA", "30_day_MA", "RSI",
                       "transaction_count", "daily_volume", "sentiment_score"]

    # ✅ Only keep features that exist in the dataset
    feature_columns = [col for col in feature_columns if col in df.columns]

    if "price" not in df.columns:
        print("❌ Error: No 'price' column found. Cannot train model.")
        return None, None, None, None

    # ✅ Drop NaN values for ML compatibility
    df = df.dropna(subset=["price"])

    # ✅ Features (X) and Target (y)
    X = df[feature_columns].fillna(0)  # Replace remaining NaNs with 0
    y = df["price"]

    # ✅ Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"✅ Features selected: {feature_columns}")
    print(f"✅ Train set: {X_train.shape}, Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test

# ✅ Step 3: Train Machine Learning Model


def train_model(X_train, y_train):
    """Train a RandomForestRegressor model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training completed.")
    return model

# ✅ Step 4: Evaluate Model Performance


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance using MAE and RMSE."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print("\n📊 Model Performance:")
    print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f}")

# ✅ Step 5: Save the Model


def save_model(model):
    """Save trained model to disk."""
    model_path = os.path.join(SAVE_PATH, "trained_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path}!")


# ✅ Run the Full Training Pipeline
if __name__ == "__main__":
    df = load_data()

    if df is not None:
        X_train, X_test, y_train, y_test = prepare_features(df)

        if X_train is not None and y_train is not None:
            model = train_model(X_train, y_train)
            evaluate_model(model, X_test, y_test)
            save_model(model)
