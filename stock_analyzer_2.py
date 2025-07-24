import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
from streamlit_autorefresh import st_autorefresh

# === CONFIGURATION ===
TOP_OPTION_STOCKS = ['AAPL', 'NVDA', 'AMD', 'MSFT', 'TSLA']
HEADERS = {'User-Agent': 'LiveMarketAnalyzer/1.0'}
REFRESH_SECONDS = 60

# === DATA FETCHING ===
def fetch_intraday_data(ticker, period="5d", interval="5m"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        st.warning(f"No intraday data for {ticker}")
        return None
    df.index = df.index.tz_localize(None)  # Remove timezone for consistency
    return df

def prepare_features(df):
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['SMA10'].fillna(method='bfill', inplace=True)
    df['RSI'].fillna(method='bfill', inplace=True)
    return df

def plot_intraday_chart(ticker):
    df = fetch_intraday_data(ticker)
    if df is None:
        return None, None, None

    df = prepare_features(df)

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close"))
    price_fig.add_trace(go.Scatter(x=df.index, y=df['SMA10'], name="SMA10", line=dict(dash='dot')))
    price_fig.update_layout(title=f"{ticker} 5-min Close + SMA10 (Last 5 Days)",
                            yaxis_title="Price", xaxis_title="Time")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
    rsi_fig.add_hline(y=70, line_dash='dash', line_color='red')
    rsi_fig.add_hline(y=30, line_dash='dash', line_color='green')
    rsi_fig.update_layout(title=f"{ticker} RSI (5-min)", yaxis_title="RSI", xaxis_title="Time")

    return df, price_fig, rsi_fig

def detect_patterns_intraday(df):
    bullet_points = []

    if 'SMA10' not in df.columns or 'RSI' not in df.columns:
        bullet_points.append("• Required indicators (SMA10 or RSI) not present in data.")
        return bullet_points, "Neutral"

    df = df.dropna(subset=['Close', 'SMA10', 'RSI']).copy()
    df = df.sort_index()  # Ensure proper chronological order

    if df.empty or len(df) < 2:
        bullet_points.append("• Not enough valid data points for pattern detection.")
        return bullet_points, "Neutral"

    # Ensure aligned indices for all Series
    close = df['Close']
    sma = df['SMA10']
    rsi = df['RSI']

    aligned = pd.concat([close.shift(1), sma.shift(1), close, sma], axis=1)
    aligned.columns = ['close_prev', 'sma_prev', 'close', 'sma']
    cross_above = ((aligned['close_prev'] < aligned['sma_prev']) & (aligned['close'] > aligned['sma'])).any()
    cross_below = ((aligned['close_prev'] > aligned['sma_prev']) & (aligned['close'] < aligned['sma'])).any()

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

def predict_next_day_price(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d")
        df = prepare_features(df)
        df = df.dropna(subset=['Close', 'SMA10', 'RSI', 'Volume'])

        features = df[['Close', 'SMA10', 'RSI', 'Volume']].values[:-1]
        target = df['Close'].values[1:]

        if len(features) < 10:
            return None, None

        model = LinearRegression()
        model.fit(features, target)

        last_row = df.iloc[-1]
        next_day_features = np.array([[last_row['Close'], last_row['SMA10'], last_row['RSI'], last_row['Volume']]])
        pred = model.predict(next_day_features)[0]
        last_close = last_row['Close']

        return round(pred, 2), round(pred - last_close, 2)
    except Exception as e:
        st.warning(f"Prediction error for {ticker}: {e}")
        return None, None

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

# === STREAMLIT APP ===
st.set_page_config(page_title="Live NASDAQ 5-min Dashboard", layout="wide")
st.title("📈 Live NASDAQ 5-Minute Intraday Dashboard with Pattern Detection & AI Prediction")

count = st_autorefresh(interval=REFRESH_SECONDS * 1000, limit=None, key="datarefresh")

st.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"Auto-refresh count: {count}")

stock_df = fetch_all_stock_signals()
crypto_data = fetch_crypto_data()
crypto_df = pd.DataFrame(crypto_data)

st.subheader("📊 Stock Buy Signals (Daily)")
st.dataframe(stock_df, use_container_width=True)

st.subheader("💰 Crypto Prices (Kraken)")
st.dataframe(crypto_df[['exchange', 'symbol', 'price', 'volume']], use_container_width=True)

selected = st.selectbox("📌 Select NASDAQ Stock for Intraday Chart & Prediction", TOP_OPTION_STOCKS)

df_intraday, price_fig, rsi_fig = plot_intraday_chart(selected)
if df_intraday is not None:
    st.plotly_chart(price_fig, use_container_width=True)
    st.plotly_chart(rsi_fig, use_container_width=True)

    bullets, trend = detect_patterns_intraday(df_intraday)
    st.markdown("### 🔍 Intraday Pattern Detection:")
    for b in bullets:
        st.markdown(b)
    st.markdown(f"**Next Intraday Trending Movement Prediction:** {trend}")
else:
    st.warning("Intraday data not available.")

try:
    predicted_price, price_change = predict_next_day_price(selected)
    if predicted_price is not None:
        st.markdown(f"### 🤖 AI Next-Day Close Price Prediction for {selected}:")
        st.markdown(f"**Predicted Price:** ${predicted_price}  _(change: {price_change:+.2f})_")
    else:
        st.warning("Prediction unavailable due to insufficient data.")
except Exception as e:
    st.error(f"Error predicting price for {selected}: {e}")
