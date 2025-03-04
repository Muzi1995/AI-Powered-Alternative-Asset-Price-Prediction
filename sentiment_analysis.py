import os
import pandas as pd
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# ✅ Download NLTK Data (Run Once)
nltk.download("vader_lexicon")

# ✅ Define Data Paths
DATA_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\data"
SAVE_PATH = r"D:\Repository\python_learn\AI-Powered Alternative Asset Returns Predictor\asset_env\processed_data"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ NewsAPI Key (Replace with Your Key)
NEWS_API_KEY = "f01de60560f44a52905a4aca40321bc8"

# ✅ Fetch Financial News from NewsAPI


def fetch_financial_news():
    """Fetch latest financial news from NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q=finance&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Failed to fetch news. Status code: {response.status_code}")
        return None

    data = response.json()

    if "articles" not in data:
        print("❌ No news articles found in response.")
        return None

    news_data = []
    for article in data["articles"]:
        news_data.append({
            "timestamp": article["publishedAt"],
            "Title": article["title"],
            "Description": article["description"]
        })

    df = pd.DataFrame(news_data)
    file_path = os.path.join(DATA_PATH, "financial_news.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ Financial news data saved at {file_path}!")
    return df

# ✅ Perform Sentiment Analysis Using VADER


def analyze_sentiment(df):
    """Compute sentiment scores for financial news articles."""
    analyzer = SentimentIntensityAnalyzer()

    # ✅ Compute Sentiment Scores
    df["sentiment_score"] = df["Title"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"])

    # ✅ Categorize Sentiment as Positive, Neutral, or Negative
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda score: "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"])  # Convert to datetime format

    # ✅ Save Sentiment Scores
    file_path = os.path.join(SAVE_PATH, "sentiment_scores.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ Sentiment scores saved at {file_path}!")
    return df


# ✅ Run Sentiment Analysis Process
if __name__ == "__main__":
    news_df = fetch_financial_news()
    if news_df is not None:
        analyze_sentiment(news_df)
