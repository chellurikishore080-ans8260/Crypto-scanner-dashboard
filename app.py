import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta

st.set_page_config(page_title="Binance Crypto AI Scanner", layout="wide", page_icon="⚡")

@st.cache_resource
def get_exchange():
    return ccxt.binance({'enableRateLimit': True})

exchange = get_exchange()

st.title("⚡ Binance Crypto Signals: EMA + RSI + ATR Scanner")

tf = st.selectbox("Select Timeframe:", ['15m', '1h', '4h', '1d'], index=1)

coins_to_scan = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'NEAR/USDT', 'BNB/USDT', 
    'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'SUI/USDT',
    'LINK/USDT', 'PEPE/USDT', 'FET/USDT', 'RENDER/USDT'
]

def analyze_coin(symbol, timeframe):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=80)
        df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        curr = df.iloc[-1]
        close = curr['close']
        atr = curr['ATR'] if not pd.isna(curr['ATR']) else close * 0.02
        rsi = curr['RSI']
        ema20 = curr['EMA_20']
        ema50 = curr['EMA_50']
        
        signal = "WAIT / NEUTRAL ⚪"
        sl = 0.0
        tp = 0.0
        
        if ema20 > ema50 and rsi > 52:
            signal = "BUY 🟢"
            sl = round(close - (1.5 * atr), 4)
            tp = round(close + (2.5 * atr), 4)
        elif ema20 < ema50 and rsi < 48:
            signal = "SELL 🔴"
            sl = round(close + (1.5 * atr), 4)
            tp = round(close - (2.5 * atr), 4)
            
        return {
            'Coin': symbol,
            'Price ($)': close,
            'Signal': signal,
            'RSI': round(rsi, 2) if not pd.isna(rsi) else 50,
            'Stop Loss ($)': sl,
            'Target ($)': tp
        }
    except:
        return None

if st.button("🚀 Scan Coins Now"):
    with st.spinner("Binance live market data scan chesthundi..."):
        results = []
        for s in coins_to_scan:
            res = analyze_coin(s, tf)
            if res:
                results.append(res)
        
        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
        else:
            st.error("Data load kaledhu, malli try cheyandi.")
else:
    st.info("Paina unna '🚀 Scan Coins Now' button click chesi live setups check cheyandi.")
