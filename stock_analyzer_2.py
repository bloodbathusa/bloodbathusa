import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# === Technical Indicator Calculation ===
def calculate_indicators(df):
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator().squeeze()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator().squeeze()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi().squeeze()

    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['Upper_Band'] = bb.bollinger_hband().squeeze()
    df['Lower_Band'] = bb.bollinger_lband().squeeze()

    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd().squeeze()
    df['MACD_Signal'] = macd.macd_signal().squeeze()

    return df

# === Trend Analysis ===
def detect_trend(df):
    latest = df.iloc[-1]
    if latest['SMA_20'] > latest['SMA_50'] and latest['MACD'] > latest['MACD_Signal']:
        return 'Bullish'
    elif latest['SMA_20'] < latest['SMA_50'] and latest['MACD'] < latest['MACD_Signal']:
        return 'Bearish'
    else:
        return 'Neutral'

# === Trade Suggestion ===
def trade_suggestion(trend, rsi):
    if trend == 'Bullish' and rsi < 70:
        return '📈 Suggestion: CALL (Uptrend, RSI Healthy)'
    elif trend == 'Bearish' and rsi > 30:
        return '📉 Suggestion: PUT (Downtrend)'
    else:
        return '⏸️ Suggestion: WAIT (Unclear or Overbought/Oversold)'

# === Main App ===
st.set_page_config(layout="wide")
st.title("📊 Stock Technical Analysis & Trade Assistant")

# Sidebar Input
with st.sidebar:
    st.header("🔍 Stock Settings")
    ticker = st.text_input("Enter stock ticker:", value="AAPL")
    period = st.selectbox("Select timeframe", ["3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Select interval", ["1d", "1h"], index=0)
    show_raw = st.checkbox("Show raw data table")
    st.markdown("---")
    st.markdown("Built with `yfinance` + `ta` + `plotly`")

# Data Load
if ticker:
    df = yf.download(ticker, period=period, interval=interval)
    
    if df.empty:
        st.error("No data available. Check ticker or date range.")
    else:
        df = calculate_indicators(df)
        trend = detect_trend(df)
        rsi = df['RSI'].iloc[-1]
        suggestion = trade_suggestion(trend, rsi)

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candlesticks'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], name='Upper Band', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], name='Lower Band', line=dict(color='gray', dash='dot')))
        fig.update_layout(title=f"{ticker} Chart with Indicators", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        st.subheader("📌 Analysis Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Trend", trend)
        col2.metric("RSI", f"{rsi:.2f}")
        col3.metric("Trade Suggestion", suggestion)

        # Optional table
        if show_raw:
            st.subheader("🔢 Raw Technical Data")
            st.dataframe(df.tail(50))

        # Download data
        csv = df.to_csv().encode('utf-8')
        st.download_button("Download CSV", csv, f"{ticker}_analysis.csv", "text/csv")
