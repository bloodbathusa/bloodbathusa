import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# === Technical Indicator Calculation ===
def calculate_indicators(df):
    try:
        # Remove .squeeze() calls - they're unnecessary and can cause issues
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['Upper_Band'] = bb.bollinger_hband()
        df['Lower_Band'] = bb.bollinger_lband()
        
        macd = MACD(close=df['Close'])
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
            
            # Handle potential NaN values in RSI
            rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 0
            suggestion = trade_suggestion(trend, rsi)
            
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
            
            fig.update_layout(
                title=f"{ticker} Technical Analysis",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white",
                height=600
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
