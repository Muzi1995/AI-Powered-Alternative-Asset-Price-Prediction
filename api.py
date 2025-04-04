import os
from dotenv import load_dotenv
import json
import pandas as pd
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

MODEL_PATH = os.path.join("models", "trained_model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Error: Model file not found at {MODEL_PATH}!")
    model = None

app = Flask(__name__)
CORS(app)

def fetch_sentiment_score(asset):
    try:
        NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # ✅ secure
        url = f"https://newsapi.org/v2/everything?q={asset}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        news_data = response.json()

        if "articles" in news_data:
            from nltk.sentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()

            sentiment_scores = [
                analyzer.polarity_scores(article["title"])["compound"]
                for article in news_data["articles"]
            ]
            return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    except Exception as e:
        print("⚠️ Sentiment API Error:", str(e))
        return 0

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the AI Asset Price Predictor API! Use /predict to get predictions."})

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        required_fields = ["7_day_MA", "30_day_MA", "RSI"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

        asset = data.get("asset", "").lower()
        sentiment_score = fetch_sentiment_score(asset)

        feature_order = ["7_day_MA", "30_day_MA", "RSI", "transaction_count", "daily_volume", "sentiment_score"]
        df = pd.DataFrame([[data["7_day_MA"], data["30_day_MA"], data["RSI"],
                            data.get("transaction_count", 10), data.get("daily_volume", 100),
                            sentiment_score]], columns=feature_order)

        prediction = model.predict(df)[0]
        return jsonify({
            "predicted_price": float(prediction),
            "sentiment_score": sentiment_score
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
