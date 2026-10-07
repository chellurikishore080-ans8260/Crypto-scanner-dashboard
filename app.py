import streamlit as st
import ccxt
import pandas as pd

st.set_page_config(page_title="Crypto AI Scanner", layout="wide", page_icon="⚡")


@st.cache_resource
def get_exchange(name):
    cls = getattr(ccxt, name)
    return cls({
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {'defaultType': 'spot', 'fetchCurrencies': False}
    })


STABLES = {'USDC', 'FDUSD', 'TUSD', 'USDP', 'DAI', 'EUR', 'AEUR', 'USDE',
           'PYUSD', 'XUSD', 'BUSD', 'USD1', 'UST', 'USDT'}

MY_LIST = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'NEAR/USDT', 'BNB/USDT',
    'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'SUI/USDT',
    'LINK/USDT', 'PEPE/USDT', 'FET/USDT', 'RENDER/USDT'
]


@st.cache_data(ttl=300, show_spinner=False)
def get_all_usdt_pairs(exchange_name):
    """Exchange lo unna anni USDT pairs + 24h volume (volume batti sorted)."""
    ex = get_exchange(exchange_name)
    tickers = ex.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith('/USDT'):
            continue
        base = sym.split('/')[0]
        if base in STABLES:
            continue
        rows.append((sym, float(t.get('quoteVolume') or 0)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


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


def analyze_coin(exchange, symbol, timeframe, volume):
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
            sl = round(close - (1.5 * atr), 6)
            tp = round(close + (2.5 * atr), 6)
        elif ema20 < ema50 and rsi < 48:
            signal = "SELL 🔴"
            sl = round(close + (1.5 * atr), 6)
            tp = round(close - (2.5 * atr), 6)

        return {
            'Coin': symbol,
            'Price ($)': close,
            'Signal': signal,
            'RSI': round(rsi, 2) if not pd.isna(rsi) else 50,
            'Stop Loss ($)': sl,
            'Target ($)': tp,
            '24h Vol ($M)': round(volume / 1_000_000, 2)
        }, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:300]}"


st.title("⚡ Crypto Signals: EMA + RSI + ATR Scanner")

c1, c2, c3 = st.columns(3)
with c1:
    exchange_name = st.selectbox(
        "Exchange:", ['binance', 'okx', 'kucoin', 'gateio', 'bybit'], index=0
    )
with c2:
    tf = st.selectbox("Select Timeframe:", ['15m', '1h', '4h', '1d'], index=1)
with c3:
    source = st.radio(
        "Coins:", ["All USDT coins", "Top by volume", "My list (14)"]
    )

top_n = 50
if source == "Top by volume":
    top_n = st.slider("Entha coins (volume batti top):", 10, 200, 50, step=10)

min_vol = 0
if source == "All USDT coins":
    min_vol = st.number_input(
        "Minimum 24h volume ($) — dead/illiquid coins skip cheyadaniki (0 = anni coins):",
        min_value=0, value=0, step=100000
    )

exchange = get_exchange(exchange_name)

if st.button("🚀 Scan Coins Now"):
    try:
        all_pairs = get_all_usdt_pairs(exchange_name)
    except Exception as e:
        st.error(f"Coin list load kaledhu: {type(e).__name__}: {str(e)[:300]}")
        st.stop()

    vol_map = dict(all_pairs)

    if source == "All USDT coins":
        symbols = [s for s, v in all_pairs if v >= min_vol]
    elif source == "Top by volume":
        symbols = [s for s, _ in all_pairs[:top_n]]
    else:
        symbols = [s for s in MY_LIST if s in vol_map]

    st.write(f"**{exchange_name}** lo {len(symbols)} coins scan avtunnayi...")

    results = []
    errors = {}
    progress = st.progress(0.0, text="Scan start avtundi...")
    for i, s in enumerate(symbols):
        progress.progress((i + 1) / len(symbols), text=f"{s} ({i + 1}/{len(symbols)})")
        res, err = analyze_coin(exchange, s, tf, vol_map.get(s, 0))
        if res:
            results.append(res)
        else:
            errors[s] = err
    progress.empty()

    st.session_state['results'] = results
    st.session_state['errors'] = errors
    st.session_state['total'] = len(symbols)
    st.session_state['exchange'] = exchange_name
    st.session_state['tf'] = tf

if 'results' in st.session_state:
    results = st.session_state['results']
    errors = st.session_state['errors']
    total = st.session_state['total']

    only_signals = st.checkbox("Only BUY / SELL chupinchu", value=False)

    if results:
        res_df = pd.DataFrame(results)
        buys = int(res_df['Signal'].str.startswith("BUY").sum())
        sells = int(res_df['Signal'].str.startswith("SELL").sum())
        st.caption(
            f"{st.session_state['exchange']} | {st.session_state['tf']} | "
            f"{len(results)}/{total} coins scan ayyayi | BUY: {buys} | SELL: {sells}"
        )
        if only_signals:
            res_df = res_df[~res_df['Signal'].str.startswith("WAIT")]
        st.dataframe(res_df, use_container_width=True, height=600)
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
