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

TOP_OPTION_STOCKS = ['AAPL', 'NVDA', 'AMD', 'MSFT', 'TSLA']
HEADERS = {'User-Agent': 'LiveMarketAnalyzer/1.0'}
REFRESH_SECONDS = 60

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

def plot_stock_chart_week(ticker):
    df = yf.download(ticker, period="1mo", interval="1d")
    df = prepare_features(df)

    # Filter last 7 calendar days (including weekends for continuity)
    last_date = df.index[-1]
    week_ago = last_date - timedelta(days=7)
    df_week = df[df.index >= week_ago]

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=df_week.index, y=df_week['Close'], name="Close"))
    price_fig.add_trace(go.Scatter(x=df_week.index, y=df_week['SMA10'], name="SMA10", line=dict(dash='dot')))
    price_fig.update_layout(title=f"{ticker} Price + SMA (Past Week)", yaxis_title="Price")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df_week.index, y=df_week['RSI'], name='RSI'))
    rsi_fig.add_hline(y=70, line_dash='dash', line_color='red')
    rsi_fig.add_hline(y=30, line_dash='dash', line_color='green')
    rsi_fig.update_layout(title=f"{ticker} RSI (Past Week)", yaxis_title="RSI")

    return df_week, price_fig, rsi_fig

def detect_patterns(df_week):
    bullet_points = []

    # Detect price crossing SMA10
    close = df_week['Close']
    sma = df_week['SMA10']

    cross_above = ((close.shift(1) < sma.shift(1)) & (close > sma)).any()
    cross_below = ((close.shift(1) > sma.shift(1)) & (close < sma)).any()

    rsi = df_week['RSI']
    rsi_trend_up = rsi.iloc[-1] > rsi.iloc[0]
    rsi_trend_down = rsi.iloc[-1] < rsi.iloc[0]

    # RSI reversal signals
    rsi_cross_overbought_down = ((rsi.shift(1) > 70) & (rsi < 70)).any()
    rsi_cross_oversold_up = ((rsi.shift(1) < 30) & (rsi > 30)).any()

    # Patterns
    if cross_above and rsi_trend_up and rsi.iloc[-1] < 70:
        bullet_points.append("• Bullish signal: Price crossed above SMA10 and RSI trending up below 70.")
    if cross_below and rsi_trend_down and rsi.iloc[-1] > 30:
        bullet_points.append("• Bearish signal: Price crossed below SMA10 and RSI trending down above 30.")
    if rsi_cross_overbought_down:
        bullet_points.append("• Reversal signal: RSI crossed down below 70 (overbought).")
    if rsi_cross_oversold_up:
        bullet_points.append("• Reversal signal: RSI crossed up above 30 (oversold).")

    if not bullet_points:
        bullet_points.append("• No clear bullish, bearish, or reversal patterns detected.")

    # Determine next trend based on simple heuristics
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
st.set_page_config(page_title="Live Stock & Crypto Dashboard", layout="wide")
st.title("📈 Live Stock & Crypto Dashboard with Pattern Detection & AI Prediction")

count = st_autorefresh(interval=REFRESH_SECONDS * 1000, limit=None, key="datarefresh")

st.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"Auto-refresh count: {count}")

stock_df = fetch_all_stock_signals()
crypto_data = fetch_crypto_data()
crypto_df = pd.DataFrame(crypto_data)

st.subheader("📊 Stock Buy Signals")
st.dataframe(stock_df, use_container_width=True)

st.subheader("💰 Crypto Prices (Kraken)")
st.dataframe(crypto_df[['exchange', 'symbol', 'price', 'volume']], use_container_width=True)

selected = st.selectbox("📌 Select Stock for Chart, Patterns & Prediction", TOP_OPTION_STOCKS)

try:
    df_week, price_fig, rsi_fig = plot_stock_chart_week(selected)
    st.plotly_chart(price_fig, use_container_width=True)
    st.plotly_chart(rsi_fig, use_container_width=True)

    bullets, trend = detect_patterns(df_week)
    st.markdown("### 🔍 Pattern Detection Summary:")
    for b in bullets:
        st.markdown(b)

    st.markdown(f"**Next Trending Movement Prediction:** {trend}")

except Exception as e:
    st.error(f"Error processing data for {selected}: {e}")

try:
    predicted_price, price_change = predict_next_day_price(selected)
    if predicted_price is not None:
        st.markdown(f"### 🤖 AI Prediction for {selected}:")
        st.markdown(f"**Next Day Close Price:** ${predicted_price}  _(change: {price_change:+.2f})_")
    else:
        st.warning("Prediction unavailable due to insufficient data.")
except Exception as e:
    st.error(f"Error predicting price for {selected}: {e}")
