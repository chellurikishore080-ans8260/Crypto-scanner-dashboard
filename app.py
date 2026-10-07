import streamlit as st
import ccxt
import pandas as pd

st.set_page_config(page_title="Binance Crypto AI Scanner", layout="wide", page_icon="⚡")


@st.cache_resource
def get_exchange(name):
    cls = getattr(ccxt, name)
    return cls({
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {'defaultType': 'spot', 'fetchCurrencies': False}
    })


st.title("⚡ Crypto Signals: EMA + RSI + ATR Scanner")

col1, col2 = st.columns(2)
with col1:
    exchange_name = st.selectbox(
        "Exchange:", ['binance', 'okx', 'kucoin', 'gateio', 'bybit'], index=0
    )
with col2:
    tf = st.selectbox("Select Timeframe:", ['15m', '1h', '4h', '1d'], index=1)

exchange = get_exchange(exchange_name)

coins_to_scan = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'NEAR/USDT', 'BNB/USDT',
    'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'SUI/USDT',
    'LINK/USDT', 'PEPE/USDT', 'FET/USDT', 'RENDER/USDT'
]


def calc_ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_atr(high, low, close, length=14):
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def analyze_coin(symbol, timeframe):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=80)
        df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'vol'])

        df['EMA_20'] = calc_ema(df['close'], 20)
        df['EMA_50'] = calc_ema(df['close'], 50)
        df['RSI'] = calc_rsi(df['close'], 14)
        df['ATR'] = calc_atr(df['high'], df['low'], df['close'], 14)

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
        }, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:300]}"


if st.button("🚀 Scan Coins Now"):
    with st.spinner(f"{exchange_name} live market data scan chesthundi..."):
        results = []
        errors = {}
        for s in coins_to_scan:
            res, err = analyze_coin(s, tf)
            if res:
                results.append(res)
            else:
                errors[s] = err

    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.error("Data load kaledhu. Asali error kinda chudandi.")

    if errors:
        all_errs = " ".join(errors.values()).lower()
        if "451" in all_errs or "restricted" in all_errs or "location" in all_errs:
            st.warning(
                "Ee exchange mee server location ni block chesthundi "
                "(Streamlit Cloud US servers). Paina Exchange ni okx / kucoin / gateio ki marchandi."
            )
        with st.expander(f"⚠️ {len(errors)} coins fail ayyayi — details"):
            for coin, e in errors.items():
                st.write(f"**{coin}** → {e}")
else:
    st.info("Paina unna '🚀 Scan Coins Now' button click chesi live setups check cheyandi.")
