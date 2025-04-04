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

# ✅ Access API keys securely
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

# ✅ Defined model path (Ensure correct model name)
MODEL_PATH = os.path.join("models", "trained_model.pkl")

# ✅ Loaded the trained model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Error: Model file not found at {MODEL_PATH}!")
    model = None

# ✅ Initialized Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# ✅ Fetched Real-Time Sentiment Score (NewsAPI)


def fetch_sentiment_score(asset):
    try:
        # Replaced with my actual NewsAPI key
        NEWS_API_KEY = "f01de60560f44a52905a4aca40321bc8"
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
        return 0  # Default neutral sentiment if API fails

# ✅ Root Endpoint - Prevents 404 errors


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the AI Asset Price Predictor API! Use /predict to get predictions."})

# ✅ Prediction API Endpoint


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        print("❌ Model is not loaded!")
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        print("📩 Received Data:", data)

        # ✅ Ensured all required fields are present
        required_fields = ["7_day_MA", "30_day_MA", "RSI"]
        missing_fields = [
            field for field in required_fields if field not in data]

        if missing_fields:
            print("❌ Missing fields:", missing_fields)
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

        # ✅ Auto-fetched Sentiment Score
        asset = data.get("asset", "").lower()
        sentiment_score = fetch_sentiment_score(asset)

        # ✅ Ensured features are in the exact order expected by the model
        feature_order = ["7_day_MA", "30_day_MA", "RSI",
                         "transaction_count", "daily_volume", "sentiment_score"]

        df = pd.DataFrame([[data["7_day_MA"], data["30_day_MA"], data["RSI"],
                            data.get("transaction_count", 10), data.get(
                                "daily_volume", 100),
                            sentiment_score]], columns=feature_order)

        print("🧠 Model Input Data:", df)

        # ✅ Make Prediction
        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_price": float(prediction),
            "sentiment_score": sentiment_score
        })

    except Exception as e:
        print("❌ API Internal Error:", str(e))
        # Printed full error traceback for debugging
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ✅ Run Flask API
if __name__ == "__main__":
    print("🚀 Starting Flask API on http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
