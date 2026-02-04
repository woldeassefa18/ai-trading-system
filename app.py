import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AI Trading Terminal", layout="wide")
st.title("📊 AI Trading Terminal")

# ==============================
# ⚡ FAST DATA LOADING (CACHED)
# ==============================
@st.cache_data(ttl=300)
def get_data(symbol, timeframe):
    df = yf.download(symbol, period="3y", interval=timeframe)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=300)
def add_indicators(df):
    df = df.copy()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['EMA50'] = ta.trend.EMAIndicator(df['Close'], 50).ema_indicator()
    df['EMA200'] = ta.trend.EMAIndicator(df['Close'], 200).ema_indicator()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['ATR'] = ta.volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close']
    ).average_true_range()
    return df

# ==============================
# 📌 SIDEBAR
# ==============================
st.sidebar.header("Market Settings")
symbol = st.sidebar.text_input("Stock", "RELIANCE.NS")
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "15m"])

data = get_data(symbol, timeframe)
data = add_indicators(data)

# ==============================
# 📈 TRADINGVIEW-STYLE CHART
# ==============================
st.subheader("📈 Price Chart")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Candles"
))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA50'], name="EMA50"))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA200'], name="EMA200"))
fig.update_layout(height=600, xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# ⏳ MULTI-TIMEFRAME TREND
# ==============================
st.subheader("⏳ Multi-Timeframe Trend")

@st.cache_data(ttl=300)
def get_trend(symbol, tf):
    df = yf.download(symbol, period="6mo", interval=tf)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df['Close']
    ma = close.rolling(50).mean().iloc[-1]
    return "Bullish" if close.iloc[-1] > ma else "Bearish"

daily_trend = get_trend(symbol, "1d")
hourly_trend = get_trend(symbol, "1h")

st.write(f"Daily Trend: **{daily_trend}**")
st.write(f"Hourly Trend: **{hourly_trend}**")

# ==============================
# 🧠 AI ENGINE (ON DEMAND)
# ==============================
st.subheader("🧠 AI Signal")

if st.button("Run AI Analysis"):
    features = data[['Close','RSI','EMA50','EMA200','MACD','ATR','Volume']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    window = 30
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=3, batch_size=16, verbose=0)

    last_seq = X[-1].reshape(1, window, X.shape[2])
    pred_scaled = model.predict(last_seq, verbose=0)
    pred_price = scaler.inverse_transform(
        np.concatenate([pred_scaled, np.zeros((1,6))], axis=1)
    )[0][0]

    current_price = features['Close'].iloc[-1]
    signal = "BUY" if pred_price > current_price else "SELL"
    confidence = min(abs(pred_price-current_price)/current_price*100*5, 95)

    st.success(f"Signal: {signal} | Confidence: {confidence:.1f}%")
else:
    st.info("Click **Run AI Analysis** to generate signal.")

# ==============================
# 🛡 RISK TOOL
# ==============================
st.subheader("🛡 Risk Calculator")

capital = st.number_input("Account Capital", 10000)
risk_percent = st.slider("Risk %", 1, 5, 2)

risk_amount = capital * (risk_percent/100)
atr = data['ATR'].iloc[-1]
price = data['Close'].iloc[-1]

stop_loss = price - atr
position_size = risk_amount / atr

st.metric("Risk Amount", f"${risk_amount:.2f}")
st.metric("Position Size", f"{position_size:.2f} shares")
