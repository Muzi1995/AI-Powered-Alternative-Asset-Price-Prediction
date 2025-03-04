import streamlit as st
import requests
import json

# ✅ Defined Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

# 🌟 **Modern UI Setup**
st.set_page_config(page_title="📊 AI Asset Predictor",
                   page_icon="💰", layout="wide")

# 🎨 **Custom CSS for a Sleek Design**
st.markdown("""
    <style>
        body {background-color: #121212; color: white;}
        .stButton>button {width: 100%; border-radius: 8px; font-size: 18px; font-weight: bold; background: linear-gradient(to right, #00c6ff, #0072ff); color: white; padding: 10px;}
        .stTextInput>div>div>input, .stNumberInput>div>div>input {background-color: #1e3a5f; color: white; border-radius: 5px;}
        .stMarkdown {font-size: 18px !important;}
        .metric-box {background-color: #1e3a5f; padding: 15px; border-radius: 8px; color: white;}
    </style>
""", unsafe_allow_html=True)

# ✅ **Header & Introduction**
st.markdown("<h1 style='text-align: center;'>🔮 AI-Powered Alternative Asset Price Prediction</h1>",
            unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Predict future prices for Real Estate, Crypto & Commodities using AI!</h3>", unsafe_allow_html=True)
st.write("---")

# 📌 **Step 1: Select Asset Type**
st.subheader("🔍 Select an Asset Type")
asset_type = st.selectbox(
    "Choose an Asset",
    ["Gold", "Real Estate", "Bitcoin", "Ethereum", "Solana", "Avalanche"],
    help="Choose the asset you want to analyze."
)

# 📆 **Step 2: Choose Investment Timeframe**
st.subheader("📆 Select Investment Timeframe")
timeframe = st.radio(
    "How long do you plan to hold this asset?",
    ["Short-term (1 Week)", "Mid-term (1 Month)", "Long-term (6 Months)"],
    horizontal=True
)

# 💰 **Step 3: Enter Investment Amount**
st.subheader("💰 Investment Amount (Optional)")
investment_amount = st.number_input(
    "Enter the amount you plan to invest ($)",
    value=1000.0, min_value=1.0, step=100.0, format="%.2f"
)

# 🔍 **Fetching Market Data Simulation**
st.markdown("🔄 **Fetching real-time market data...**")

# 📊 **Market Data (Simulated)**
market_data = {
    "7_day_MA": 0.0012,  # Replace with API call
    "30_day_MA": 0.0014,  # Replace with API call
    "RSI": 55,  # Replace with API call
    "sentiment_score": 0.3  # Replace with NewsAPI sentiment analysis
}

# 🎯 **Display Market Data in a Stylish Layout**
st.subheader("📊 Market Indicators")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="📉 7-Day Moving Average",
              value=f"${market_data['7_day_MA']:.6f}")
with col2:
    st.metric(label="📈 30-Day Moving Average",
              value=f"${market_data['30_day_MA']:.6f}")
with col3:
    st.metric(label="📊 Relative Strength Index (RSI)",
              value=f"{market_data['RSI']} (Momentum Indicator)")
with col4:
    st.metric(label="📰 Sentiment Score",
              value=f"{market_data['sentiment_score']} (Positive if > 0.05)")

st.write("---")

# 🔮 **Predict Button with Modern Styling**
if st.button("🔮 Predict Future Price"):
    # Prepare data for API call
    input_data = {
        "asset": asset_type.lower(),
        "7_day_MA": market_data["7_day_MA"],
        "30_day_MA": market_data["30_day_MA"],
        "RSI": market_data["RSI"],
        "sentiment_score": market_data["sentiment_score"]
    }

    # ✅ Send request to Flask API
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            prediction = response.json().get("predicted_price", "Error")

            # 🎯 **Display Prediction in a Modern Box**
            st.markdown(f"""
                <div class='metric-box'>
                    <h2>💰 Predicted Price for {asset_type}: <span style='color: #ffcc00;'>${prediction:,.6f}</span></h2>
                </div>
            """, unsafe_allow_html=True)

            # 💸 **Investment Return Simulation**
            if investment_amount:
                estimated_return = investment_amount * \
                    (prediction / market_data["7_day_MA"])
                st.markdown(f"""
                    <div class='metric-box'>
                        📊 If you invest <span style='color: #00ffcc;'>${investment_amount:,.2f}</span>, 
                        your <i>estimated return</i> is <span style='color: #ffcc00;'>${estimated_return:,.2f}</span>.
                    </div>
                """, unsafe_allow_html=True)

            # 📊 **Market Trend Analysis**
            if prediction > market_data["30_day_MA"]:
                st.success(
                    "📈 **Uptrend Expected!** This asset is likely to increase in value.")
            elif prediction < market_data["7_day_MA"]:
                st.warning(
                    "📉 **Possible Decline** – Recent indicators suggest a downward trend.")
            else:
                st.info(
                    "⚖ **Stable Market Conditions** – No major price movements expected.")

        else:
            st.error(f"⚠️ API Error: {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Unable to connect to API. Ensure `api.py` is running.")

# ✅ **Footer for Branding**
st.markdown("---")
st.markdown("<h4 style='text-align: center;'>🚀 Built with Streamlit & Flask | AI-Powered Alternative Asset Predictor</h4>", unsafe_allow_html=True)
