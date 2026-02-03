import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# -------- PAGE SETUP --------
st.set_page_config(page_title="AI Trading Platform", layout="wide")
st.title("🤖 AI Trading Platform")

# 🔄 AUTO REFRESH EVERY 15 MINUTES
st_autorefresh(interval=15 * 60 * 1000, key="scanner")

# -------- TELEGRAM FUNCTION --------
def send_telegram_alert(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": chat_id, "text": message})
        except:
            pass

# -------- SINGLE STOCK ANALYSIS --------
st.sidebar.subheader("🔍 Single Stock Analysis")
symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS")

data = yf.download(symbol, period="3y", interval="1d")
if data.empty:
    st.error("No market data found.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# -------- INDICATORS --------
data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data['EMA50'] = ta.trend.EMAIndicator(data['Close'], 50).ema_indicator()
data['EMA200'] = ta.trend.EMAIndicator(data['Close'], 200).ema_indicator()
data['MACD'] = ta.trend.MACD(data['Close']).macd()
data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data['Trend_Strength'] = data['EMA50'] - data['EMA200']

features = data[['Close','RSI','EMA50','EMA200','MACD','ATR','Log_Return','Trend_Strength','Volume']].dropna()
if len(features) < 80:
    st.error("Not enough data for AI model.")
    st.stop()

# -------- AI PREP --------

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

window = 30
X, y = [], []
for i in range(window, len(scaled)):
    X.append(scaled[i-window:i])
    y.append(scaled[i,0])
X, y = np.array(X), np.array(y)

# -------- MODEL --------
model_path = "ai_model.keras"
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

# -------- SIGNAL --------

last_seq = X[-1].reshape(1, window, X.shape[2])
pred_scaled = model.predict(last_seq, verbose=0)
pred_price = scaler.inverse_transform(np.concatenate([pred_scaled,np.zeros((1,8))],axis=1))[0][0]
current_price = features['Close'].iloc[-1]

ai_signal = "BUY" if pred_price > current_price else "SELL"
confidence = min(abs(pred_price-current_price)/current_price*100*5,95)

st.subheader("🧠 AI Signal")
st.write(f"Signal: **{ai_signal}** | Confidence: **{confidence:.2f}%**")

# -------- SMART ALERT --------

state_file = "last_signal.txt"
last_signal = open(state_file).read().strip() if os.path.exists(state_file) else "NONE"

trend_ok = (ai_signal=="BUY" and data['EMA50'].iloc[-1]>data['EMA200'].iloc[-1]) or \
           (ai_signal=="SELL" and data['EMA50'].iloc[-1]<data['EMA200'].iloc[-1])

if ai_signal != last_signal and confidence > 55 and trend_ok:
    send_telegram_alert(f"🚨 {symbol} {ai_signal} | Conf: {confidence:.1f}% | Price: {current_price:.2f}")
    with open(state_file,"w") as f:
        f.write(ai_signal)

# -------- JOURNAL --------

log_file="trade_log.csv"
entry=pd.DataFrame([{"Date":datetime.now(),"Stock":symbol,"Price":current_price,
                     "Prediction":pred_price,"Signal":ai_signal,"Confidence":confidence}])
entry.to_csv(log_file,mode='a',header=not os.path.exists(log_file),index=False)
st.subheader("📒 Trade Journal")
st.dataframe(pd.read_csv(log_file).tail())

# -------- PERFORMANCE ANALYTICS --------

st.subheader("📊 Strategy Performance")

journal = pd.read_csv(log_file)

if len(journal) > 5:
    journal['Result'] = np.where(
        (journal['Signal'] == "BUY") & (journal['Prediction'] > journal['Price']), "Win",
        np.where((journal['Signal'] == "SELL") & (journal['Prediction'] < journal['Price']), "Win", "Loss")
    )

    wins = (journal['Result'] == "Win").sum()
    losses = (journal['Result'] == "Loss").sum()
    accuracy = wins / (wins + losses) * 100 if wins + losses > 0 else 0

    st.metric("Total Trades", len(journal))
    st.metric("Wins", wins)
    st.metric("Losses", losses)
    st.metric("Accuracy", f"{accuracy:.2f}%")
else:
    st.info("Not enough trades yet for performance stats.")

# -------- RISK MANAGEMENT ENGINE --------
st.subheader("🛡 Risk Management")

capital = st.number_input("Account Capital ($)", value=10000)
risk_percent = st.slider("Risk per Trade (%)", 1, 5, 2)

risk_amount = capital * (risk_percent / 100)

atr = data['ATR'].iloc[-1]
stop_loss = current_price - atr if ai_signal == "BUY" else current_price + atr
take_profit = current_price + atr*2 if ai_signal == "BUY" else current_price - atr*2

position_size = risk_amount / abs(current_price - stop_loss)

st.metric("Risk Amount", f"${risk_amount:.2f}")
st.metric("Position Size (shares)", f"{position_size:.2f}")
st.metric("Stop Loss", f"{stop_loss:.2f}")
st.metric("Take Profit", f"{take_profit:.2f}")


# -------- MULTI-STOCK SCANNER --------

st.subheader("📡 AI Market Scanner")
scan_list = st.sidebar.text_area(
    "Stocks to Scan",
    "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"
).split(",")

results=[]
for s in scan_list:
    try:
        df=yf.download(s.strip(),period="2y",interval="1d",progress=False)
        if df.empty: continue
        if isinstance(df.columns,pd.MultiIndex):
            df.columns=df.columns.get_level_values(0)

        df['RSI']=ta.momentum.RSIIndicator(df['Close']).rsi()
        df['EMA50']=ta.trend.EMAIndicator(df['Close'],50).ema_indicator()
        df['EMA200']=ta.trend.EMAIndicator(df['Close'],200).ema_indicator()
        df['MACD']=ta.trend.MACD(df['Close']).macd()
        df['ATR']=ta.volatility.AverageTrueRange(df['High'],df['Low'],df['Close']).average_true_range()
        df['Log_Return']=np.log(df['Close']/df['Close'].shift(1))
        df['Trend_Strength']=df['EMA50']-df['EMA200']

        f=df[['Close','RSI','EMA50','EMA200','MACD','ATR','Log_Return','Trend_Strength','Volume']].dropna()
        if len(f)<60: continue

        scaled_f=scaler.transform(f)
        seq=scaled_f[-window:].reshape(1,window,scaled_f.shape[1])
        p=model.predict(seq,verbose=0)
        pred_p=scaler.inverse_transform(np.concatenate([p,np.zeros((1,8))],axis=1))[0][0]
        cur_p=f['Close'].iloc[-1]
        sig="BUY" if pred_p>cur_p else "SELL"
        conf=min(abs(pred_p-cur_p)/cur_p*100*5,95)
        results.append([s,sig,f"{conf:.1f}%",round(cur_p,2)])
    except: pass

st.dataframe(pd.DataFrame(results,columns=["Stock","Signal","Confidence","Price"]))
