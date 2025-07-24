import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Live Stock Dashboard", layout="wide")
st.title("📈 Live Stock Price Dashboard with AI Trend Prediction")

# Helper functions
def fetch_intraday_data(ticker):
    try:
        df = yf.download(tickers=ticker, period="5d", interval="5m")
        df.dropna(inplace=True)
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    except Exception as e:
        st.error(f"Error fetching intraday data for {ticker}: {e}")
        return pd.DataFrame()

def detect_patterns_intraday(df):
    bullet_points = []

    if 'SMA10' not in df.columns or 'RSI' not in df.columns:
        bullet_points.append("• Required indicators (SMA10 or RSI) not present in data.")
        return bullet_points, "Neutral"

    df = df.dropna(subset=['Close', 'SMA10', 'RSI']).copy()
    if df.empty or len(df) < 2:
        bullet_points.append("• Not enough valid data points for pattern detection.")
        return bullet_points, "Neutral"

    close = df['Close']
    sma = df['SMA10']
    rsi = df['RSI']

    cross_above = ((close.shift(1) < sma.shift(1)) & (close > sma)).any()
    cross_below = ((close.shift(1) > sma.shift(1)) & (close < sma)).any()

    rsi_trend_up = rsi.iloc[-1] > rsi.iloc[0]
    rsi_trend_down = rsi.iloc[-1] < rsi.iloc[0]
    rsi_cross_overbought_down = ((rsi.shift(1) > 70) & (rsi < 70)).any()
    rsi_cross_oversold_up = ((rsi.shift(1) < 30) & (rsi > 30)).any()

    if cross_above and rsi_trend_up and rsi.iloc[-1] < 70:
        bullet_points.append("• Bullish signal: Price crossed above SMA10 and RSI trending up below 70.")
    if cross_below and rsi_trend_down and rsi.iloc[-1] > 30:
        bullet_points.append("• Bearish signal: Price crossed below SMA10 and RSI trending down above 30.")
    if rsi_cross_overbought_down:
        bullet_points.append("• Reversal signal: RSI crossed down below 70 (overbought).")
    if rsi_cross_oversold_up:
        bullet_points.append("• Reversal signal: RSI crossed up above 30 (oversold).")

    if not bullet_points:
        bullet_points.append("• No clear intraday bullish, bearish, or reversal patterns detected.")

    if any("Bullish" in pt for pt in bullet_points):
        trend = "Bullish"
    elif any("Bearish" in pt for pt in bullet_points):
        trend = "Bearish"
    elif any("Reversal" in pt for pt in bullet_points):
        trend = "Reversal"
    else:
        trend = "Neutral"

    return bullet_points, trend

def predict_next_close(df):
    df = df.dropna(subset=['Close']).copy()
    df['Timestamp'] = df.index.astype(np.int64) // 10**9
    X = df[['Timestamp']]
    y = df['Close']
    model = LinearRegression().fit(X, y)
    future_time = np.array([[X['Timestamp'].iloc[-1] + 60*60*24]])  # 1 day ahead
    pred_price = model.predict(future_time)[0]
    return round(pred_price, 2)

# Sidebar ticker selection
st.sidebar.header("Stock Selector")
ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA, MSFT)", "AAPL")

if ticker:
    df_intraday = fetch_intraday_data(ticker)

    if not df_intraday.empty:
        st.subheader(f"📊 {ticker.upper()} - 5-Minute Chart (Last 5 Days)")

        # Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday['SMA10'], mode='lines', name='SMA10'))
        fig.update_layout(title=f"{ticker.upper()} Intraday Movement", xaxis_title="Time", yaxis_title="Price (USD)",
                          template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # AI Predictions
        prediction = predict_next_close(df_intraday)
        st.metric(label="🔮 Predicted Next Day Close", value=f"${prediction}")

        # Pattern Detection
        bullets, trend = detect_patterns_intraday(df_intraday)
        st.markdown(f"### 📌 Trend Analysis: {trend}")
        for point in bullets:
            st.markdown(point)
    else:
        st.warning("No intraday data available. Try another ticker or check API limits.")
