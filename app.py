import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
import requests
import time
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# -------- PAGE SETUP --------
st.set_page_config(page_title="AI Trading Terminal", layout="wide")
st.title("📊 AI Trading Terminal")

# -------- TELEGRAM --------
def send_telegram_alert(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": message}
            )
        except:
            pass

# -------- SIDEBAR --------
st.sidebar.header("Market Settings")
symbol = st.sidebar.text_input("Stock", "RELIANCE.NS")
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "15m"])

# -------- DATA --------
data = yf.download(symbol, period="3y", interval=timeframe)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# -------- INDICATORS --------
data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data['EMA50'] = ta.trend.EMAIndicator(data['Close'], 50).ema_indicator()
data['EMA200'] = ta.trend.EMAIndicator(data['Close'], 200).ema_indicator()
data['MACD'] = ta.trend.MACD(data['Close']).macd()
data['ATR'] = ta.volatility.AverageTrueRange(
    data['High'], data['Low'], data['Close']
).average_true_range()

# -------- CHART --------
st.subheader("📈 Price Chart")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'], high=data['High'],
    low=data['Low'], close=data['Close'],
    name="Candles"
))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA50'], name="EMA50"))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA200'], name="EMA200"))
fig.update_layout(height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# -------- MULTI TIMEFRAME TREND (FIXED) --------
st.subheader("⏳ Multi-Timeframe Trend")

daily = yf.download(symbol, period="6mo", interval="1d")
hourly = yf.download(symbol, period="1mo", interval="1h")

daily_close = daily['Close'].iloc[:,0] if isinstance(daily['Close'], pd.DataFrame) else daily['Close']
hourly_close = hourly['Close'].iloc[:,0] if isinstance(hourly['Close'], pd.DataFrame) else hourly['Close']

daily_ma = daily_close.rolling(50).mean().iloc[-1]
hourly_ma = hourly_close.rolling(50).mean().iloc[-1]

daily_trend = "Bullish" if daily_close.iloc[-1] > daily_ma else "Bearish"
hourly_trend = "Bullish" if hourly_close.iloc[-1] > hourly_ma else "Bearish"

st.write(f"Daily Trend: **{daily_trend}**")
st.write(f"Hourly Trend: **{hourly_trend}**")

# -------- AI MODEL --------
features = data[['Close','RSI','EMA50','EMA200','MACD','ATR','Volume']].dropna()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

window = 30
X, y = [], []
for i in range(window, len(scaled)):
    X.append(scaled[i-window:i])
    y.append(scaled[i,0])
X, y = np.array(X), np.array(y)

model_path="ai_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse')
else:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    model.save(model_path)

last_seq = X[-1].reshape(1, window, X.shape[2])
pred_scaled = model.predict(last_seq, verbose=0)
pred_price = scaler.inverse_transform(
    np.concatenate([pred_scaled, np.zeros((1,6))], axis=1)
)[0][0]

current_price = features['Close'].iloc[-1]
ai_signal = "BUY" if pred_price > current_price else "SELL"
confidence = min(abs(pred_price-current_price)/current_price*100*5, 95)

st.subheader("🧠 AI Signal")
st.write(f"{ai_signal} | Confidence {confidence:.1f}%")

# -------- JOURNAL --------
log_file="trade_log.csv"
entry=pd.DataFrame([{
    "Date":datetime.now(),"Stock":symbol,
    "Price":current_price,"Prediction":pred_price,
    "Signal":ai_signal,"Confidence":confidence
}])
entry.to_csv(log_file,mode='a',header=not os.path.exists(log_file),index=False)
journal=pd.read_csv(log_file)
st.dataframe(journal.tail())

# -------- PERFORMANCE --------
journal['Result'] = np.where(
    (journal['Signal']=="BUY") & (journal['Prediction']>journal['Price']),"Win",
    np.where((journal['Signal']=="SELL") & (journal['Prediction']<journal['Price']),"Win","Loss")
)
wins=(journal['Result']=="Win").sum()
losses=(journal['Result']=="Loss").sum()
st.metric("Accuracy",f"{wins/(wins+losses)*100:.2f}%")

# -------- RISK --------
st.subheader("🛡 Risk Management")
capital = st.number_input("Capital",10000)
risk_percent = st.slider("Risk %",1,5,2)
risk_amount = capital*(risk_percent/100)
atr=data['ATR'].iloc[-1]
stop_loss=current_price-atr if ai_signal=="BUY" else current_price+atr
take_profit=current_price+atr*2 if ai_signal=="BUY" else current_price-atr*2
position_size=risk_amount/abs(current_price-stop_loss)
st.metric("Position Size",f"{position_size:.2f}")

# -------- PORTFOLIO --------
st.subheader("💼 Portfolio Allocation")

portfolio = {
    "RELIANCE.NS": 0.25,
    "TCS.NS": 0.25,
    "INFY.NS": 0.25,
    "HDFCBANK.NS": 0.25
}

portfolio_value = 0.0  # ensure float

for stock, weight in portfolio.items():
    try:
        df = yf.download(stock, period="1y", progress=False)

        if df.empty or 'Close' not in df:
            continue

        close = df['Close']

        # If multi-column, take first column
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        # Drop NaNs
        close = close.dropna()

        if len(close) < 2:
            continue

        start_price = float(close.iloc[0])
        end_price = float(close.iloc[-1])

        ret = (end_price / start_price) - 1
        portfolio_value += float(capital) * float(weight) * (1 + ret)

    except:
        continue

st.metric("Portfolio Value", f"${float(np.asarray(portfolio_value).item()):,.2f}")


# -------- AUTO LOOP --------
time.sleep(900)
st.rerun()
