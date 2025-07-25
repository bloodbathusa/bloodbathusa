import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# === Helper Functions ===

def calculate_indicators(df):
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()

    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['Upper_Band'] = bb.bollinger_hband().values.flatten()
    df['Lower_Band'] = bb.bollinger_lband().values.flatten()

    macd = MACD(df['Close'])
    df['MACD'] = macd.macd().values.flatten()
    df['MACD_Signal'] = macd.macd_signal().values.flatten()

    return df

def detect_trend(df):
    latest = df.iloc[-1]
    if latest['SMA_20'] > latest['SMA_50'] and latest['MACD'] > latest['MACD_Signal']:
        return 'Bullish'
    elif latest['SMA_20'] < latest['SMA_50'] and latest['MACD'] < latest['MACD_Signal']:
        return 'Bearish'
    else:
        return 'Neutral'

def trade_suggestion(trend, rsi):
    if trend == 'Bullish' and rsi < 70:
        return '📈 Suggestion: CALL (Momentum Positive)'
    elif trend == 'Bearish' and rsi > 30:
        return '📉 Suggestion: PUT (Downtrend)'
    else:
        return '⏸️ Suggestion: WAIT (Sideways or Overbought/Oversold)'

# === Streamlit UI ===

st.title("📊 Technical Analysis & Trade Suggestion App")

ticker = st.text_input("Enter Stock Ticker", value="AAPL")

if ticker:
    df = yf.download(ticker, period="6mo", interval="1d")

    if df.empty:
        st.warning("No data found for ticker.")
    else:
        df = calculate_indicators(df)
        trend = detect_trend(df)
        rsi = df['RSI'].iloc[-1]
        suggestion = trade_suggestion(trend, rsi)

        # === Chart ===
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candlesticks'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue'), name='SMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='red'), name='SMA 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='gray'), name='Upper Band', line_dash='dot'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='gray'), name='Lower Band', line_dash='dot'))
        fig.update_layout(title=f"{ticker} Technical Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # === Summary ===
        st.subheader("📌 Analysis Summary")
        st.markdown(f"- **Trend:** {trend}")
        st.markdown(f"- **RSI:** {rsi:.2f}")
        st.markdown(f"- {suggestion}")

        if st.checkbox("Show Raw Data"):
            st.dataframe(df.tail(50))
