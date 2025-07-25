import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import numpy as np
from datetime import datetime, timedelta

# === Candlestick Pattern Detection ===
def detect_candlestick_patterns(df):
    patterns = []
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        open_price = current['Open']
        close_price = current['Close']
        high_price = current['High']
        low_price = current['Low']
        
        prev_open = prev['Open']
        prev_close = prev['Close']
        
        body = abs(close_price - open_price)
        prev_body = abs(prev_close - prev_open)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        # Doji Pattern
        if body <= (high_price - low_price) * 0.1:
            patterns.append((df.index[i], 'Doji', '⚖️'))
        
        # Hammer Pattern
        elif (lower_shadow > body * 2 and upper_shadow < body * 0.1 and 
              close_price > open_price):
            patterns.append((df.index[i], 'Hammer', '🔨'))
        
        # Shooting Star
        elif (upper_shadow > body * 2 and lower_shadow < body * 0.1 and 
              close_price < open_price):
            patterns.append((df.index[i], 'Shooting Star', '🌟'))
        
        # Bullish Engulfing
        elif (i > 0 and prev_close < prev_open and close_price > open_price and
              open_price < prev_close and close_price > prev_open):
            patterns.append((df.index[i], 'Bullish Engulfing', '🟢'))
        
        # Bearish Engulfing
        elif (i > 0 and prev_close > prev_open and close_price < open_price and
              open_price > prev_close and close_price < prev_open):
            patterns.append((df.index[i], 'Bearish Engulfing', '🔴'))
    
    return patterns[-10:]  # Return last 10 patterns

# === Option Chain Data ===
def get_option_chain(ticker):
    try:
        stock = yf.Ticker(ticker)
        exp_dates = stock.options
        
        if not exp_dates:
            return None, None
            
        # Get nearest expiration
        nearest_exp = exp_dates[0]
        option_chain = stock.option_chain(nearest_exp)
        
        calls = option_chain.calls.head(10)  # Top 10 calls
        puts = option_chain.puts.head(10)   # Top 10 puts
        
        return calls, puts, nearest_exp
    except Exception as e:
        st.error(f"Could not fetch options data: {str(e)}")
        return None, None, None

# === AI Signal Scoring ===
def calculate_ai_signal_score(df, trend, rsi, patterns):
    score = 50  # Base score (neutral)
    confidence = 0
    signals = []
    
    try:
        latest = df.iloc[-1]
        
        # Trend Analysis (30% weight)
        if trend == 'Bullish':
            score += 15
            signals.append("📈 Bullish trend detected")
        elif trend == 'Bearish':
            score -= 15
            signals.append("📉 Bearish trend detected")
        
        # RSI Analysis (20% weight)
        if not pd.isna(rsi):
            if rsi < 30:
                score += 10
                signals.append(f"🔥 RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                score -= 10
                signals.append(f"❄️ RSI overbought ({rsi:.1f})")
            elif 40 <= rsi <= 60:
                score += 5
                signals.append(f"✅ RSI healthy zone ({rsi:.1f})")
        
        # Volume Analysis (15% weight)
        if len(df) >= 20:
            avg_volume = df['Volume'].tail(20).mean()
            current_volume = latest.get('Volume', 0)
            if current_volume > avg_volume * 1.5:
                score += 8
                signals.append("📊 High volume confirmation")
            elif current_volume < avg_volume * 0.5:
                score -= 5
                signals.append("📊 Low volume warning")
        
        # MACD Analysis (15% weight)
        if not pd.isna(latest.get('MACD')) and not pd.isna(latest.get('MACD_Signal')):
            if latest['MACD'] > latest['MACD_Signal']:
                score += 7
                signals.append("🚀 MACD bullish crossover")
            else:
                score -= 7
                signals.append("🛑 MACD bearish crossover")
        
        # Candlestick Patterns (10% weight)
        bullish_patterns = ['Hammer', 'Bullish Engulfing']
        bearish_patterns = ['Shooting Star', 'Bearish Engulfing']
        
        for _, pattern, _ in patterns[-3:]:  # Last 3 patterns
            if pattern in bullish_patterns:
                score += 5
                signals.append(f"🕯️ {pattern} pattern detected")
            elif pattern in bearish_patterns:
                score -= 5
                signals.append(f"🕯️ {pattern} pattern detected")
        
        # Price Position Analysis (10% weight)
        if not pd.isna(latest.get('SMA_20')) and not pd.isna(latest.get('SMA_50')):
            price = latest['Close']
            if price > latest['SMA_20'] > latest['SMA_50']:
                score += 5
                signals.append("📊 Price above both MAs")
            elif price < latest['SMA_20'] < latest['SMA_50']:
                score -= 5
                signals.append("📊 Price below both MAs")
        
        # Normalize score and calculate confidence
        score = max(0, min(100, score))
        confidence = abs(score - 50) * 2  # 0-100 confidence based on deviation from neutral
        
        return score, confidence, signals
        
    except Exception as e:
        return 50, 0, [f"Error in AI analysis: {str(e)}"]
def calculate_indicators(df):
    try:
        # Flatten MultiIndex columns if they exist (common yfinance issue)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure we have proper 1D Series by using .squeeze()
        close_series = df['Close'].squeeze()
        
        df['SMA_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=close_series, window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
        
        bb = BollingerBands(close=close_series, window=20, window_dev=2)
        df['Upper_Band'] = bb.bollinger_hband()
        df['Lower_Band'] = bb.bollinger_lband()
        
        macd = MACD(close=close_series)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df

# === Trend Analysis ===
def detect_trend(df):
    try:
        latest = df.iloc[-1]
        # Add null checks
        if pd.isna(latest['SMA_20']) or pd.isna(latest['SMA_50']) or pd.isna(latest['MACD']) or pd.isna(latest['MACD_Signal']):
            return 'Insufficient Data'
            
        if latest['SMA_20'] > latest['SMA_50'] and latest['MACD'] > latest['MACD_Signal']:
            return 'Bullish'
        elif latest['SMA_20'] < latest['SMA_50'] and latest['MACD'] < latest['MACD_Signal']:
            return 'Bearish'
        else:
            return 'Neutral'
    except Exception as e:
        return 'Error'

# === Trade Suggestion ===
def trade_suggestion(trend, rsi):
    try:
        if pd.isna(rsi):
            return '⏸️ Suggestion: WAIT (Insufficient RSI data)'
        
        if trend == 'Bullish' and rsi < 70:
            return '📈 Suggestion: CALL (Uptrend, RSI Healthy)'
        elif trend == 'Bearish' and rsi > 30:
            return '📉 Suggestion: PUT (Downtrend)'
        else:
            return '⏸️ Suggestion: WAIT (Unclear or Overbought/Oversold)'
    except Exception as e:
        return '⏸️ Suggestion: WAIT (Error in analysis)'

# === Main App ===
st.set_page_config(layout="wide")
st.title("📊 Stock Technical Analysis & Trade Assistant")

# Sidebar Input
with st.sidebar:
    st.header("🔍 Stock Settings")
    ticker = st.text_input("Enter stock ticker:", value="AAPL").upper()
    period = st.selectbox("Select timeframe", ["3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Select interval", ["1d", "1h"], index=0)
    show_raw = st.checkbox("Show raw data table")
    st.markdown("---")
    st.markdown("Built with `yfinance` + `ta` + `plotly`")

# Data Load
if ticker:
    try:
        with st.spinner(f'Loading data for {ticker}...'):
            # Add progress indication
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            st.error("❌ No data available. Check ticker symbol or try a different timeframe.")
        else:
            df = calculate_indicators(df)
            trend = detect_trend(df)
            
            # Get candlestick patterns and AI analysis
            patterns = detect_candlestick_patterns(df)
            ai_score, confidence, ai_signals = calculate_ai_signal_score(df, trend, rsi, patterns)
            
            # Get options data
            calls, puts, exp_date = get_option_chain(ticker)
            
            # Chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price',
                showlegend=True
            ))
            
            # Add moving averages (only if data exists)
            if not df['SMA_20'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['SMA_20'], 
                    name='SMA 20', line=dict(color='blue', width=2)
                ))
            
            if not df['SMA_50'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['SMA_50'], 
                    name='SMA 50', line=dict(color='red', width=2)
                ))
            
            # Add Bollinger Bands
            if not df['Upper_Band'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Upper_Band'], 
                    name='Upper Band', line=dict(color='gray', dash='dot'),
                    fill=None
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Lower_Band'], 
                    name='Lower Band', line=dict(color='gray', dash='dot'),
                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
                ))
            
            # Handle potential NaN values in RSI
            rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 0
            suggestion = trade_suggestion(trend, rsi)
            
            # Add pattern annotations to chart
            for date, pattern, emoji in patterns:
                fig.add_annotation(
                    x=date, y=df.loc[date]['High'] * 1.02,
                    text=f"{emoji}<br>{pattern}",
                    showarrow=True, arrowhead=2,
                    arrowsize=1, arrowwidth=2,
                    bgcolor="yellow", bordercolor="orange",
                    font=dict(size=10)
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            st.subheader("📌 Analysis Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_color = "normal"
                if trend == "Bullish":
                    trend_color = "normal"  # Green in Streamlit
                elif trend == "Bearish":
                    trend_color = "inverse"  # Red in Streamlit
                st.metric("Trend", trend)
            
            with col2:
                rsi_delta = None
                if not pd.isna(rsi):
                    if rsi > 70:
                        rsi_delta = "Overbought"
                    elif rsi < 30:
                        rsi_delta = "Oversold"
                st.metric("RSI", f"{rsi:.2f}" if not pd.isna(rsi) else "N/A")
            
            with col3:
                st.write("**Trade Suggestion:**")
                st.write(suggestion)
            
            # Additional metrics
            st.subheader("📈 Additional Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            latest_data = df.iloc[-1]
            with col1:
                st.metric("Current Price", f"${latest_data['Close']:.2f}")
            with col2:
                if not pd.isna(latest_data['MACD']):
                    st.metric("MACD", f"{latest_data['MACD']:.4f}")
                else:
                    st.metric("MACD", "N/A")
            with col3:
                if len(df) > 1:
                    price_change = latest_data['Close'] - df.iloc[-2]['Close']
                    st.metric("Daily Change", f"${price_change:.2f}", f"{price_change:.2f}")
                else:
                    st.metric("Daily Change", "N/A")
            with col4:
                volume = latest_data.get('Volume', 0)
                if volume > 0:
                    st.metric("Volume", f"{volume:,.0f}")
                else:
                    st.metric("Volume", "N/A")
            
            # Optional table
            if show_raw:
                st.subheader("🔢 Raw Technical Data")
                display_df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal']].tail(50)
                st.dataframe(display_df, use_container_width=True)
            
            # Download data
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{ticker}_analysis.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Try checking your internet connection or using a different ticker symbol.")
else:
    st.info("👆 Enter a stock ticker symbol in the sidebar to get started!")
