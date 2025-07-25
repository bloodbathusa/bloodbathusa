import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import talib

# === Helper Functions ===

def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = talib.BBANDS(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    macd, macd_signal, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'], df['MACD_Signal'] = macd, macd_signal
    return df

def detect_trend(df):
    latest = df.iloc[-1]
    if latest['SMA_20'] > latest['SMA_50'] and latest['MACD'] > latest['MACD_Signal']:
        return 'Bullish'
    elif latest['SMA_20'] < latest['SMA_50'] and latest['MACD'] < latest['MACD_Signal']:
        return 'Bearish'
    else:
        return 'Neutral'

def detect_patterns(df):
    patterns = {
        'Doji': talib.CDLDOJI,
        'Hammer': talib.CDLHAMMER,
        'Engulfing': talib.CDLENGULFING,
        'Shooting Star': talib.CDLSHOOTINGSTAR,
    }
    results = {}
    for name, func in patterns.items():
        pattern_series = func(df['Open'], df['High'], df['Low'], df['Close'])
        if pattern_series.iloc[-1] != 0:
            results[name] = int(pattern_series.iloc[-1])
    return results

def trade_suggestion(trend, rsi, patterns):
    if 'Hammer' in patterns or 'Engulfing' in patterns:
        return '📈 Suggestion: CALL (Reversal Pattern Detected)'
    elif 'Shooting Star' in patterns or trend == 'Bearish':
        return '📉 Suggestion: PUT (Bearish Sentiment)'
    elif trend == 'Bullish' and rsi < 70:
        return '📈 Suggestion: CALL (Momentum)'
    elif trend == 'Bearish' and rsi > 30:
        return '📉 Suggestion: PUT'
    else:
        return '⏸️ Suggestion: Wait'

# === Streamlit UI ===

st.title("📊 Advanced Technical Analysis & Trade Advisor")

ticker = st.text_input("Enter Stock Ticker", value="AAPL")

if ticker:
    df = yf.download(ticker, period="6mo", interval="1d")
    
    if df.empty:
        st.warning("No data found. Check ticker symbol.")
    else:
        df = calculate_indicators(df)
        trend = detect_trend(df)
        patterns = detect_patterns(df)
        rsi = df['RSI'].iloc[-1]
        suggestion = trade_suggestion(trend, rsi, patterns)

        # === Chart ===
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candlestick'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='gray'), name='Upper Band'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='gray'), name='Lower Band'))
        fig.update_layout(title=f"{ticker} Candlestick Chart with Bollinger Bands", xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

        # === Summary ===
        st.subheader("📌 Technical Summary")
        st.markdown(f"- **Trend:** {trend}")
        st.markdown(f"- **RSI:** {rsi:.2f}")
        st.markdown(f"- **Patterns Detected:** {', '.join(patterns.keys()) if patterns else 'None'}")
        st.markdown(f"- {suggestion}")

        if st.checkbox("Show Data"):
            st.dataframe(df.tail(50))
