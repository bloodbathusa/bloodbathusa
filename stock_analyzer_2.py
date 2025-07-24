import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
from streamlit_autorefresh import st_autorefresh

# === CONFIGURATION ===
TOP_OPTION_STOCKS = ['AAPL', 'NVDA', 'AMD', 'MSFT', 'TSLA']
HEADERS = {'User-Agent': 'LiveMarketAnalyzer/1.0'}

REFRESH_SECONDS = 60  # Auto-refresh interval

# === DATA FETCHING FUNCTIONS ===
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1mo')
        close = hist['Close'].iloc[-1]
        sma_10 = hist['Close'].rolling(10).mean().iloc[-1]

        delta = hist['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean().iloc[-1]
        avg_loss = loss.rolling(14).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        signal = "Buy" if rsi < 35 and close > sma_10 else "Watch"
        target = round(sma_10 * 1.03, 2)

        return {
            'symbol': ticker,
            'price': round(close, 2),
            'SMA10': round(sma_10, 2),
            'RSI': round(rsi, 2),
            'signal': signal,
            'target_price': target
        }
    except:
        return None

def fetch_all_stock_signals():
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_stock_data, TOP_OPTION_STOCKS))
    return pd.DataFrame([r for r in results if r])

def fetch_crypto_data():
    try:
        url = 'https://api.kraken.com/0/public/Ticker?pair=BTCUSD,ETHUSD'
        r = requests.get(url, headers=HEADERS)
        data = r.json()['result']
        return [
            {
                'exchange': 'Kraken',
                'symbol': k,
                'price': float(v['c'][0]),
                'volume': float(v['v'][1]),
                'timestamp': datetime.now()
            }
            for k, v in data.items()
        ]
    except:
        return []

# === AI Prediction: Linear Regression for Next Day Close ===
def predict_next_day_price(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d")
        df = df.dropna(subset=['Close'])
        df['Day'] = np.arange(len(df))
        X = df[['Day']]
        y = df['Close']

        model = LinearRegression()
        model.fit(X, y)

        next_day = np.array([[len(df)]])
        pred = model.predict(next_day)[0]
        last_close = y.iloc[-1]

        return round(pred, 2), round(pred - last_close, 2)
    except:
        return None, None

# === PLOTTING ===
def plot_stock_chart(ticker):
    df = yf.download(ticker, period="2mo", interval="1d")
    df['SMA10'] = df['Close'].rolling(10).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close"))
    price_fig.add_trace(go.Scatter(x=df.index, y=df['SMA10'], name="SMA10", line=dict(dash='dot')))
    price_fig.update_layout(title=f"{ticker} Price + SMA", yaxis_title="Price")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
    rsi_fig.add_hline(y=70, line_dash='dash', line_color='red')
    rsi_fig.add_hline(y=30, line_dash='dash', line_color='green')
    rsi_fig.update_layout(title=f"{ticker} RSI", yaxis_title="RSI")

    return price_fig, rsi_fig

# === STREAMLIT APP ===
st.set_page_config(page_title="Live Stock & Crypto Dashboard", layout="wide")
st.title("📈 Live Stock & Crypto Dashboard with Auto-Refresh & AI Prediction")

# Auto-refresh every REFRESH_SECONDS seconds
count = st_autorefresh(interval=REFRESH_SECONDS * 1000, limit=None, key="datarefresh")

st.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"Auto-refresh count: {count}")

# Fetch data
stock_df = fetch_all_stock_signals()
crypto_data = fetch_crypto_data()
crypto_df = pd.DataFrame(crypto_data)

st.subheader("📊 Stock Buy Signals")
st.dataframe(stock_df, use_container_width=True)

st.subheader("💰 Crypto Prices (Kraken)")
st.dataframe(crypto_df[['exchange', 'symbol', 'price', 'volume']], use_container_width=True)

selected = st.selectbox("📌 Select Stock for Chart & Prediction", TOP_OPTION_STOCKS)

price_fig, rsi_fig = plot_stock_chart(selected)
st.plotly_chart(price_fig, use_container_width=True)
st.plotly_chart(rsi_fig, use_container_width=True)

predicted_price, price_change = predict_next_day_price(selected)
if predicted_price is not None:
    st.markdown(f"### 🤖 AI Prediction for {selected}:")
    st.markdown(f"**Next Day Close Price:** ${predicted_price}  _(change: {price_change:+.2f})_")
else:
    st.write("Prediction unavailable.")
