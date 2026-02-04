import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide")
st.title("🚀 AI Trading Terminal")

# =========================
# 📌 SIDEBAR SETTINGS
# =========================
symbol = st.sidebar.text_input("Asset (e.g. NSE:RELIANCE)", "NSE:RELIANCE")
interval = st.sidebar.selectbox("Timeframe", ["1", "5", "15", "60", "D"])

col1, col2 = st.columns([2,1])

# =========================
# 📈 LEFT — REAL TRADINGVIEW CHART
# =========================
with col1:
    tv_chart = f"""
    <iframe 
        src="https://www.tradingview.com/widgetembed/?symbol={symbol}&interval={interval}&theme=dark&style=1&toolbarbg=1e1e1e&hide_top_toolbar=false&hide_legend=false&save_image=false&studies=RSI%40tv-basicstudies%2CMACD%40tv-basicstudies%2CVolume%40tv-basicstudies"
        width="100%" 
        height="700" 
        frameborder="0" 
        allowtransparency="true" 
        scrolling="no">
    </iframe>
    """
    st.components.v1.html(tv_chart, height=700)

# =========================
# 🧠 RIGHT — AI ENGINE
# =========================
with col2:
    st.subheader("🧠 AI Signal Engine")

    # Convert TradingView symbol to Yahoo format
    yf_symbol = symbol.split(":")[-1] + ".NS" if "NSE:" in symbol else symbol

    @st.cache_data(ttl=300)
    def load_data(sym):
        df = yf.download(sym, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

  ai_data = load_data(yf_symbol)

if ai_data.empty:
    st.error("AI data unavailable for this symbol.")
else:
    if len(ai_data) < 60:
        st.warning("Limited history, AI confidence reduced.")

    # Indicators
    ai_data['RSI'] = ta.momentum.RSIIndicator(ai_data['Close']).rsi()
    ai_data['EMA50'] = ta.trend.EMAIndicator(ai_data['Close'],50).ema_indicator()
    ai_data['EMA200'] = ta.trend.EMAIndicator(ai_data['Close'],200).ema_indicator()
    ai_data['MACD'] = ta.trend.MACD(ai_data['Close']).macd()
    ai_data['ATR'] = ta.volatility.AverageTrueRange(
        ai_data['High'], ai_data['Low'], ai_data['Close']
    ).average_true_range()

    features = ai_data[['Close','RSI','EMA50','EMA200','MACD','ATR','Volume']].dropna()


    # Indicators
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['EMA50'] = ta.trend.EMAIndicator(data['Close'],50).ema_indicator()
    data['EMA200'] = ta.trend.EMAIndicator(data['Close'],200).ema_indicator()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['ATR'] = ta.volatility.AverageTrueRange(
        data['High'], data['Low'], data['Close']
    ).average_true_range()

    features = data[['Close','RSI','EMA50','EMA200','MACD','ATR','Volume']].dropna()

    if st.button("Run AI Analysis"):
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

        current_price = ai_data['Close'].iloc[-1]

        signal = "BUY" if pred_price > current_price else "SELL"
        confidence = min(abs(pred_price-current_price)/current_price*100*5, 95)

        st.success(f"Signal: {signal}")
        st.write(f"Confidence: {confidence:.1f}%")
        st.write(f"Predicted Price: {pred_price:.2f}")
        st.write(f"Current Price: {current_price:.2f}")

# =========================
# 🛡 RISK MANAGEMENT PANEL
# =========================
st.subheader("🛡 Risk Management")

capital = st.number_input("Capital", 10000)
risk_percent = st.slider("Risk % per trade",1,5,2)

risk_amount = capital * (risk_percent/100)
atr = data['ATR'].iloc[-1]
position_size = risk_amount / atr

colA, colB = st.columns(2)
colA.metric("Risk Amount", f"${risk_amount:.2f}")
colB.metric("Position Size", f"{position_size:.2f} shares")
