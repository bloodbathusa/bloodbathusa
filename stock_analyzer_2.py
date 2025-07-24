import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objs as go
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
    df.index = df.index.tz_localize(None)
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
    """
    Analyzes intraday stock data to find bullish, bearish, and reversal patterns.

    Args:
        df (pd.DataFrame): DataFrame with 'Close', 'SMA10', and 'RSI' columns.

    Returns:
        tuple: A list of descriptive strings (bullet_points) and a final trend string.
    """
    bullet_points = []
    trend = "Neutral"

    # 1. --- Data Validation ---
    expected_cols = ['Close', 'SMA10', 'RSI']
    if not all(col in df.columns for col in expected_cols):
        return ["• Data columns ('Close', 'SMA10', 'RSI') not found."], "Error"

    df = df.dropna(subset=expected_cols).copy()
    if len(df) < 10: # Increased minimum length for more reliable analysis
        return ["• Not enough data for pattern detection."], "Neutral"

    # 2. --- Get Most Recent Data ---
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 3. --- Check for Recent SMA Crossover ---
    crossed_above_sma = prev['Close'] < prev['SMA10'] and last['Close'] > last['SMA10']
    crossed_below_sma = prev['Close'] > prev['SMA10'] and last['Close'] < last['SMA10']

    # 4. --- Check Recent RSI Trend and State ---
    # Check if RSI has been rising over the last 5 periods
    rsi_trending_up = last['RSI'] > df['RSI'].iloc[-5:].mean()
    rsi_trending_down = last['RSI'] < df['RSI'].iloc[-5:].mean()
    
    # Check if RSI just crossed out of oversold/overbought
    rsi_crossed_up_from_oversold = prev['RSI'] < 30 and last['RSI'] > 30
    rsi_crossed_down_from_overbought = prev['RSI'] > 70 and last['RSI'] < 70

    # 5. --- Generate Bullet Points based on Logic ---
    if crossed_above_sma:

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
st.title("📈 Live NASDAQ 5-Minute Intraday Dashboard with Pattern Detection")

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

selected = st.selectbox("📌 Select NASDAQ Stock for Intraday Chart", TOP_OPTION_STOCKS)

df_intraday, price_fig, rsi_fig = plot_intraday_chart(selected)
if df_intraday is not None:
    st.plotly_chart(price_fig, use_container_width=True)
    st.plotly_chart(rsi_fig, use_container_width=True)

    # Corrected line: removed the "return" statement
    bullet_points, trend = detect_patterns_intraday(df_intraday)
    
    st.markdown("### 🔍 Intraday Pattern Detection:")
    for b in bullet_points:
        st.markdown(b)
    st.markdown(f"**Next Intraday Trending Movement Prediction:** {trend}")
else:
    st.warning("Intraday data not available.")
