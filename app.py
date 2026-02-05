import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide")
st.title("🌍 AI Market Intelligence Terminal")

# =====================================================
# 🔧 FIX YFINANCE SHAPE BUG
# =====================================================
def fix_yf_data(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    for col in ['Open','High','Low','Close','Volume']:
        if col in df.columns:
            df[col] = pd.Series(df[col].to_numpy().flatten(), index=df.index)

    return df

# =====================================================
# 📌 SIDEBAR
# =====================================================
symbol = st.sidebar.text_input("Asset (TradingView format)", "NASDAQ:AAPL")
interval = st.sidebar.selectbox("Chart Timeframe", ["1", "5", "15", "60", "D"])

def convert_symbol(tv_symbol):
    if "NSE:" in tv_symbol:
        return tv_symbol.split(":")[1] + ".NS"
    elif "BSE:" in tv_symbol:
        return tv_symbol.split(":")[1] + ".BO"
    else:
        return tv_symbol.split(":")[1]

yf_symbol = convert_symbol(symbol)

col1, col2 = st.columns([2, 1])

# =====================================================
# 📈 TRADINGVIEW CHART
# =====================================================
with col1:
    tv_chart = f"""
    <iframe 
        src="https://www.tradingview.com/widgetembed/?symbol={symbol}&interval={interval}&theme=dark&style=1&toolbarbg=1e1e1e&studies=RSI%40tv-basicstudies%2CMACD%40tv-basicstudies%2CVolume%40tv-basicstudies"
        width="100%" height="700" frameborder="0"></iframe>
    """
    st.components.v1.html(tv_chart, height=700)

# =====================================================
# 🧠 AI SIGNAL ENGINE
# =====================================================
with col2:
    st.subheader("🧠 AI Signal Engine")

    @st.cache_data(ttl=300)
    def load_data(sym):
        df = yf.download(sym, period="1y", interval="1d", progress=False)
        return fix_yf_data(df)

    data = load_data(yf_symbol)

    if not data.empty:
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['EMA50'] = ta.trend.EMAIndicator(data['Close'], 50).ema_indicator()
        data['EMA100'] = ta.trend.EMAIndicator(data['Close'], 100).ema_indicator()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['ATR'] = ta.volatility.AverageTrueRange(
            data['High'], data['Low'], data['Close']
        ).average_true_range()

        features = data[['Close','RSI','EMA50','EMA100','MACD','ATR','Volume']].dropna()

        if st.button("Run AI Analysis"):
            if len(features) < 25:
                st.warning("Not enough data for AI model.")
            else:
                with st.spinner("AI analyzing..."):
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(features)

                    window = 15
                    X, y = [], []
                    for i in range(window, len(scaled)):
                        X.append(scaled[i-window:i])
                        y.append(scaled[i,0])
                    X, y = np.array(X), np.array(y)

                    model = Sequential([LSTM(32, input_shape=(X.shape[1], X.shape[2])), Dense(1)])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X, y, epochs=1, verbose=0)

                    pred_scaled = model.predict(X[-1].reshape(1, window, X.shape[2]), verbose=0)
                    pred_price = scaler.inverse_transform(
                        np.concatenate([pred_scaled, np.zeros((1,6))], axis=1)
                    )[0][0]

                    current_price = data['Close'].iloc[-1]
                    signal = "BUY" if pred_price > current_price else "SELL"
                    confidence = min(abs(pred_price-current_price)/current_price*100*5, 95)

                    st.success(f"Signal: {signal}")
                    st.write(f"Confidence: {confidence:.1f}%")

# =====================================================
# 📡 MULTI-STOCK AI SCANNER
# =====================================================
st.subheader("📡 Multi-Stock AI Scanner")

assets = [
    "NASDAQ:AAPL","NASDAQ:TSLA","NASDAQ:MSFT",
    "NSE:RELIANCE","NSE:TCS","NSE:HDFCBANK"
]

if st.button("Run Market Scanner"):
    with st.spinner("Scanning markets..."):
        results = []

        for asset in assets:
            yf_symbol = convert_symbol(asset)
            df = yf.download(yf_symbol, period="6mo", interval="1d", progress=False)
            df = fix_yf_data(df)

            if df.empty or len(df) < 60:
                continue

            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['EMA50'] = ta.trend.EMAIndicator(df['Close'],50).ema_indicator()
            df['EMA100'] = ta.trend.EMAIndicator(df['Close'],100).ema_indicator()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['ATR'] = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close']
            ).average_true_range()

            features = df[['Close','RSI','EMA50','EMA100','MACD','ATR','Volume']].dropna()

            if len(features) < 25:
                continue

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features)

            window = 15
            X = np.array([scaled[i-window:i] for i in range(window, len(scaled))])

            model = Sequential([LSTM(16, input_shape=(X.shape[1], X.shape[2])), Dense(1)])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, scaled[window:,0], epochs=1, verbose=0)

            pred_scaled = model.predict(X[-1].reshape(1, window, X.shape[2]), verbose=0)
            pred_price = scaler.inverse_transform(
                np.concatenate([pred_scaled, np.zeros((1,6))], axis=1)
            )[0][0]

            current_price = features['Close'].iloc[-1]
            signal = "BUY" if pred_price > current_price else "SELL"
            confidence = round(abs(pred_price-current_price)/current_price*100*5, 1)

            results.append([asset, signal, confidence])

        if results:
            scan_df = pd.DataFrame(results, columns=["Asset","Signal","Confidence %"])
            st.dataframe(scan_df.sort_values("Confidence %", ascending=False), use_container_width=True)
        else:
            st.warning("No stocks had sufficient AI data.")

# =====================================================
# 🛡 RISK PANEL
# =====================================================
st.subheader("🛡 Risk Management")

capital = st.number_input("Capital", 10000)
risk_percent = st.slider("Risk % per trade", 1, 5, 2)

risk_amount = capital * (risk_percent / 100)
atr = data['ATR'].iloc[-1] if not data.empty else 1
position_size = risk_amount / atr

colA, colB = st.columns(2)
colA.metric("Risk Amount", f"${risk_amount:.2f}")
colB.metric("Position Size", f"{position_size:.2f}")
