"""
Crypto Signals: EMA + RSI + ATR Scanner  (Kishore)

* Data  : Binance SPOT via data-api.binance.vision (primary) -> api.binance.com -> okx / kucoin / gateio / bybit (ccxt)
* Signal: EMA20/EMA50 + RSI(14) state, ATR(14) based SL / Target. A signal STARTS when the state flips to BUY or SELL.
* Views : Fresh signals (one place) | Running | Completed (SUCCESS = target hit) | Backtest | All coins
* Candles: closed candles only (no flicker / repaint). Times in IST.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Crypto AI Scanner", layout="wide", page_icon="⚡")

# ═════════════════════════════ constants ═════════════════════════════
BINANCE_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
FALLBACK_EXCHANGES = ["okx", "kucoin", "gateio", "bybit"]
TF_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
BINANCE_MODE = "Binance Spot (Vision) + auto fallback"

STABLES = {'USDC', 'FDUSD', 'TUSD', 'USDP', 'DAI', 'EUR', 'AEUR', 'USDE',
           'PYUSD', 'XUSD', 'BUSD', 'USD1', 'UST', 'USDT'}

MY_LIST = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'NEAR/USDT', 'BNB/USDT',
    'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'SUI/USDT',
    'LINK/USDT', 'PEPE/USDT', 'FET/USDT', 'RENDER/USDT'
]

STATUS_TXT = {
    "TP": "SUCCESS ✅ target hit",
    "SL": "SL HIT ❌",
    "REV": "CLOSED ↔ opposite signal",
    "RUN": "RUNNING ⏳",
}


@dataclass(frozen=True)
class Strat:
    rsi_buy: float = 52.0
    rsi_sell: float = 48.0
    sl_atr: float = 1.5
    tp_atr: float = 2.5
    exit_on_reverse: bool = True


@dataclass
class Ctx:
    """Everything a worker thread needs (no Streamlit calls inside threads)."""
    chain: list
    hosts: list
    exs: dict = field(default_factory=dict)
    locks: dict = field(default_factory=dict)


# ═════════════════════════════ data layer ═════════════════════════════
@st.cache_resource
def get_exchange(name):
    cls = getattr(ccxt, name)
    return cls({
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {'defaultType': 'spot', 'fetchCurrencies': False}
    })


def probe_hosts():
    """Binance hosts that answer right now (Streamlit Cloud US servers can get 451 on some)."""
    ok = []
    for h in BINANCE_HOSTS:
        try:
            requests.get(h + "/api/v3/ping", timeout=6).raise_for_status()
            ok.append(h)
        except Exception:
            pass
    return ok


def _http_json(host, path, params, timeout=12):
    for attempt in range(2):
        r = requests.get(host + path, params=params, timeout=timeout)
        if r.status_code in (418, 429) and attempt == 0:
            time.sleep(1.5)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("rate limited")


def binance_json(hosts, path, params):
    last = "no Binance host reachable"
    for host in hosts:
        try:
            return _http_json(host, path, params), host
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                raise ValueError("symbol Binance lo ledu")     # not listed -> go to the next source
            last = f"{host.replace('https://', '')}: HTTP {getattr(e.response, 'status_code', '?')}"
        except Exception as e:
            last = f"{host.replace('https://', '')}: {type(e).__name__}"
    raise RuntimeError(last)


def _closed(df, now_ms):
    df = df[df["close_time"] <= now_ms]
    return df.reset_index(drop=True)


def klines_binance(ctx, symbol, tf, limit):
    sym = symbol.replace("/", "")
    rows, end, host = [], None, None
    while len(rows) < limit:
        lim = min(1000, limit - len(rows))
        params = {"symbol": sym, "interval": tf, "limit": lim}
        if end is not None:
            params["endTime"] = end
        chunk, host = binance_json(ctx.hosts, "/api/v3/klines", params)
        if not chunk:
            break
        rows = chunk + rows
        end = chunk[0][0] - 1
        if len(chunk) < lim:
            break
    if not rows:
        raise RuntimeError("empty klines")
    df = pd.DataFrame([r[:6] for r in rows], columns=["time", "open", "high", "low", "close", "vol"]).astype(float)
    df["time"] = df["time"].astype("int64")
    df["close_time"] = df["time"] + TF_MS[tf]
    df = _closed(df.drop_duplicates("time").sort_values("time"), int(time.time() * 1000))
    return df, ("binance-vision" if "vision" in host else "binance")


def klines_ccxt(ctx, name, symbol, tf, limit):
    ex = ctx.exs[name]
    with ctx.locks[name]:
        bars = ex.fetch_ohlcv(symbol, timeframe=tf, limit=min(limit, 1000))
    if not bars:
        raise RuntimeError("empty ohlcv")
    df = pd.DataFrame([b[:6] for b in bars], columns=["time", "open", "high", "low", "close", "vol"]).astype(float)
    df["time"] = df["time"].astype("int64")
    df["close_time"] = df["time"] + TF_MS[tf]
    df = _closed(df.drop_duplicates("time").sort_values("time"), int(time.time() * 1000))
    return df, name


def fetch_candles(symbol, tf, limit, ctx):
    errs = []
    for src in ctx.chain:
        try:
            if src == "binance":
                return klines_binance(ctx, symbol, tf, limit)
            return klines_ccxt(ctx, src, symbol, tf, limit)
        except Exception as e:
            errs.append(f"{src}: {type(e).__name__}: {str(e)[:80]}")
    raise RuntimeError(" | ".join(errs) if errs else "no data source")


def load_pairs(ctx):
    """All USDT spot pairs with 24h quote volume, from the first source that works."""
    errs = []
    for src in ctx.chain:
        try:
            rows = []
            if src == "binance":
                data, _ = binance_json(ctx.hosts, "/api/v3/ticker/24hr", {})
                for t in data:
                    s = t.get("symbol", "")
                    if not s.endswith("USDT"):
                        continue
                    base = s[:-4]
                    vol = float(t.get("quoteVolume") or 0)
                    if base and base not in STABLES and vol > 0:
                        rows.append((f"{base}/USDT", vol))
            else:
                with ctx.locks[src]:
                    tickers = ctx.exs[src].fetch_tickers()
                for sym, t in tickers.items():
                    if not sym.endswith('/USDT') or sym.split('/')[0] in STABLES:
                        continue
                    rows.append((sym, float(t.get('quoteVolume') or 0)))
            if rows:
                rows.sort(key=lambda x: x[1], reverse=True)
                return rows, src
        except Exception as e:
            errs.append(f"{src}: {type(e).__name__}: {str(e)[:80]}")
    raise RuntimeError(" | ".join(errs) if errs else "no data source")


# ═════════════════════════════ indicators ═════════════════════════════
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


# ═════════════════════════════ signal engine ═════════════════════════════
def ms_ts(ms):
    return pd.Timestamp(int(ms), unit="ms", tz="UTC")


def simulate(i, d, sl, tp, hi, lo, cl, state, exit_on_reverse):
    """
    Walk forward from the signal candle i (entry = its close).
    Returns (status, exit_index, exit_price). status: TP | SL | REV | RUN.
    If SL and TP are both touched inside one candle, SL is assumed first (conservative).
    """
    h, l = hi[i + 1:], lo[i + 1:]
    sl_hit = (l <= sl) if d == 1 else (h >= sl)
    tp_hit = (h >= tp) if d == 1 else (l <= tp)
    cands = []
    if sl_hit.any():
        cands.append((int(sl_hit.argmax()), 0, "SL", sl))
    if tp_hit.any():
        cands.append((int(tp_hit.argmax()), 1, "TP", tp))
    if exit_on_reverse:
        rev = state[i + 1:] == -d
        if rev.any():
            k = int(rev.argmax())
            cands.append((k, 2, "REV", cl[i + 1 + k]))
    if not cands:
        return "RUN", None, None
    k, _, kind, px = min(cands, key=lambda c: (c[0], c[1]))
    return kind, i + 1 + k, float(px)


def build_signals(df, symbol, volume, S, src):
    """df: closed candles. Returns (coin_summary, events)."""
    close = df["close"]
    ema20 = calc_ema(close, 20).to_numpy()
    ema50 = calc_ema(close, 50).to_numpy()
    rsi = calc_rsi(close, 14).to_numpy()
    atr = calc_atr(df["high"], df["low"], close, 14).to_numpy()
    hi, lo, cl = df["high"].to_numpy(), df["low"].to_numpy(), close.to_numpy()
    close_ms = df["close_time"].to_numpy(dtype="int64")

    valid = ~(np.isnan(ema50) | np.isnan(rsi) | np.isnan(atr))
    buy = valid & (ema20 > ema50) & (rsi > S.rsi_buy)
    sell = valid & (ema20 < ema50) & (rsi < S.rsi_sell)
    state = np.where(buy, 1, np.where(sell, -1, 0)).astype(int)
    prev = np.concatenate([[0], state[:-1]])
    first_valid = int(np.argmax(valid)) if valid.any() else len(valid)
    last = len(cl) - 1

    events = []
    for i in np.flatnonzero((state != 0) & (state != prev)):
        i = int(i)
        if i <= first_valid:                       # state may have started before our data: not a real start
            continue
        d = int(state[i])
        entry = float(cl[i])
        sl = entry - d * S.sl_atr * float(atr[i])
        tp = entry + d * S.tp_atr * float(atr[i])
        kind, j, px = simulate(i, d, sl, tp, hi, lo, cl, state, S.exit_on_reverse)
        events.append(dict(
            coin=symbol, side=d, idx=i, signal_time=ms_ts(close_ms[i]), entry=entry, sl=float(sl), tp=float(tp),
            status=kind,
            exit_time=ms_ts(close_ms[j]) if j is not None else pd.NaT,
            exit_price=px if j is not None else float(cl[last]),
            candles_ago=last - i, vol=volume, rsi=float(rsi[i]),
        ))

    cur = int(state[last])
    run_start = last
    while cur != 0 and run_start > 0 and state[run_start - 1] == cur:
        run_start -= 1
    cur_ev = next((e for e in events if e["idx"] == run_start and e["side"] == cur), None)
    summary = dict(
        coin=symbol, vol=volume, src=src, price=float(cl[last]), last_time=ms_ts(close_ms[last]),
        state=cur, rsi=float(rsi[last]) if not np.isnan(rsi[last]) else np.nan,
        since=ms_ts(close_ms[run_start]) if cur != 0 else pd.NaT,
        since_price=float(cl[run_start]) if cur != 0 else np.nan,
        cur_event=cur_ev,
    )
    return summary, events


def analyze_coin(symbol, volume, tf, bars, ctx, S):
    df, src = fetch_candles(symbol, tf, bars, ctx)
    if len(df) < 70:
        raise RuntimeError(f"candles takkuva ({len(df)})")
    return build_signals(df, symbol, volume, S, src)


# ═════════════════════════════ backtest (portfolio) ═════════════════════════════
def run_portfolio(ev, capital, pos_pct, fee_pct, compound, buy_only):
    """
    All coins share ONE capital. Every signal uses pos_pct % of capital (fixed) or of current equity (compound).
    A signal is skipped when there is not enough free capital (50 % per trade -> max 2 trades open together).
    Returns (trades_df, summary_dict, equity_df).
    """
    if buy_only:
        ev = ev[ev["side"] == 1]
    ev = ev.sort_values(["signal_time", "vol"], ascending=[True, False])
    fee = fee_pct / 100.0
    state = {"realized": float(capital), "open": []}
    taken, skipped = [], 0

    def release(upto):
        keep = []
        for t in state["open"]:
            if t["status"] != "RUN" and t["exit_time"] <= upto:
                state["realized"] += t["pnl"]
            else:
                keep.append(t)
        state["open"] = keep

    for r in ev.itertuples():
        release(r.signal_time)
        reserved = sum(t["size"] for t in state["open"])
        base = state["realized"] if compound else float(capital)
        size = pos_pct / 100.0 * base
        if size <= 0 or state["realized"] - reserved < size - 1e-9:
            skipped += 1
            continue
        ret = r.side * (r.exit_price - r.entry) / r.entry - 2 * fee
        tr = dict(coin=r.coin, side=r.side, signal_time=r.signal_time, entry=r.entry, exit_time=r.exit_time,
                  exit_price=r.exit_price, status=r.status, size=size, pnl=size * ret, ret_pct=ret * 100)
        state["open"].append(tr)
        taken.append(tr)
    release(pd.Timestamp("2262-01-01", tz="UTC"))

    tdf = pd.DataFrame(taken)
    summ = dict(trades=0, closed=0, wins=0, win_rate=0.0, realized=0.0, open_pnl=0.0, final=float(capital),
                ret_pct=0.0, pf=0.0, max_dd=0.0, tp=0, sl=0, rev=0, running=0, skipped=skipped,
                avg_win=0.0, avg_loss=0.0)
    if tdf.empty:
        return tdf, summ, pd.DataFrame(columns=["time", "equity"])
    closed = tdf[tdf["status"] != "RUN"].sort_values("exit_time")
    running = tdf[tdf["status"] == "RUN"]
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]
    eq = float(capital) + closed["pnl"].cumsum().to_numpy()
    curve = pd.DataFrame({"time": closed["exit_time"].to_numpy(), "equity": eq})
    if len(eq):
        peak = np.maximum.accumulate(np.concatenate([[float(capital)], eq]))[1:]
        max_dd = float(((peak - eq) / peak).max() * 100)
    else:
        max_dd = 0.0
    gw, gl = float(wins["pnl"].sum()), -float(losses["pnl"].sum())
    summ.update(
        trades=len(tdf), closed=len(closed), wins=len(wins),
        win_rate=(len(wins) / len(closed) * 100) if len(closed) else 0.0,
        realized=float(closed["pnl"].sum()), open_pnl=float(running["pnl"].sum()),
        pf=(gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0), max_dd=max_dd,
        tp=int((closed["status"] == "TP").sum()), sl=int((closed["status"] == "SL").sum()),
        rev=int((closed["status"] == "REV").sum()), running=len(running),
        avg_win=float(wins["pnl"].mean()) if len(wins) else 0.0,
        avg_loss=float(losses["pnl"].mean()) if len(losses) else 0.0,
    )
    summ["final"] = float(capital) + summ["realized"] + summ["open_pnl"]
    summ["ret_pct"] = (summ["final"] - float(capital)) / float(capital) * 100
    return tdf, summ, curve


# ═════════════════════════════ UI helpers ═════════════════════════════
def fp(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:.4f}"
    if ax >= 0.01:
        return f"{x:.5f}"
    return f"{x:.8f}"


def ist(ts):
    if ts is None or pd.isna(ts):
        return "-"
    return ts.tz_convert("Asia/Kolkata").strftime("%d %b %H:%M")


def side_txt(s):
    return "BUY 🟢" if s == 1 else "SELL 🔴"


# ═════════════════════════════ UI ═════════════════════════════
st.title("⚡ Crypto Signals: EMA + RSI + ATR Scanner")

c1, c2, c3 = st.columns(3)
with c1:
    source_mode = st.selectbox("Data source:", [BINANCE_MODE, "okx", "kucoin", "gateio", "bybit"], index=0)
with c2:
    tf = st.selectbox("Select Timeframe:", ["5m", "15m", "30m", "1h", "4h", "1d"], index=3)
with c3:
    source = st.radio("Coins:", ["All USDT coins", "Top by volume", "My list (14)"])

top_n = 50
if source == "Top by volume":
    top_n = st.slider("Entha coins (volume batti top):", 10, 200, 50, step=10)

min_vol = 0
if source == "All USDT coins":
    min_vol = st.number_input(
        "Minimum 24h volume ($) — dead/illiquid coins skip cheyadaniki (0 = anni coins):",
        min_value=0, value=0, step=100000
    )

with st.expander("⚙️ Strategy settings (marchithe malli Scan cheyyali)"):
    s1, s2, s3, s4 = st.columns(4)
    rsi_buy = s1.number_input("BUY: RSI >", 40.0, 80.0, 52.0, 1.0)
    rsi_sell = s2.number_input("SELL: RSI <", 20.0, 60.0, 48.0, 1.0)
    sl_atr = s3.number_input("Stop Loss (× ATR)", 0.5, 5.0, 1.5, 0.25)
    tp_atr = s4.number_input("Target (× ATR)", 0.5, 10.0, 2.5, 0.25)
    s5, s6 = st.columns(2)
    exit_rev = s5.checkbox("Opposite signal vachhaka trade close", value=True)
    bars = s6.selectbox("History candles (backtest ki)", [300, 500, 1000], index=1)
S = Strat(rsi_buy=rsi_buy, rsi_sell=rsi_sell, sl_atr=sl_atr, tp_atr=tp_atr, exit_on_reverse=exit_rev)

if st.button("🚀 Scan Coins Now"):
    hosts = probe_hosts() if source_mode == BINANCE_MODE else []
    if source_mode == BINANCE_MODE:
        chain = (["binance"] if hosts else []) + FALLBACK_EXCHANGES
    else:
        chain = [source_mode]
    ctx = Ctx(chain=chain, hosts=hosts,
              exs={n: get_exchange(n) for n in chain if n != "binance"},
              locks={n: threading.Lock() for n in chain if n != "binance"})
    if source_mode == BINANCE_MODE and not hosts:
        st.warning("Binance (Vision) reach avvatledu — okx / kucoin / gateio / bybit nundi data testunnanu.")

    try:
        all_pairs, pairs_src = load_pairs(ctx)
    except Exception as e:
        st.error(f"Coin list load kaledhu: {str(e)[:300]}")
        st.stop()

    vol_map = dict(all_pairs)
    if source == "All USDT coins":
        symbols = [s for s, v in all_pairs if v >= min_vol]
    elif source == "Top by volume":
        symbols = [s for s, _ in all_pairs[:top_n]]
    else:
        symbols = [s for s in MY_LIST if s in vol_map]
    if not symbols:
        st.error("Scan cheyyadaniki coins levu (filters check cheyyandi).")
        st.stop()

    st.write(f"**{pairs_src}** nundi {len(symbols)} coins scan avtunnayi...")
    summaries, events, errors, sources = [], [], {}, {}
    progress = st.progress(0.0, text="Scan start avtundi...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(analyze_coin, s, vol_map.get(s, 0), tf, bars, ctx, S): s for s in symbols}
        for n, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            progress.progress(n / len(symbols), text=f"{s} ({n}/{len(symbols)})")
            try:
                summ, evs = fut.result()
                summaries.append(summ)
                events += evs
                sources[summ["src"]] = sources.get(summ["src"], 0) + 1
            except Exception as e:
                errors[s] = f"{type(e).__name__}: {str(e)[:250]}"
    progress.empty()

    st.session_state['scan'] = dict(
        summaries=summaries, events=events, errors=errors, total=len(symbols), tf=tf, bars=bars,
        sources=sources, at=pd.Timestamp.now(tz="UTC"), S=S,
    )

scan = st.session_state.get('scan')
if not scan:
    st.info("Paina unna '🚀 Scan Coins Now' button click chesi live setups check cheyandi.")
    st.stop()

summaries, errors, total = scan['summaries'], scan['errors'], scan['total']
if not summaries:
    st.error("Data load kaledhu. Asali error kinda chudandi.")
else:
    src_txt = ", ".join(f"{k} {v}" for k, v in scan['sources'].items())
    st.caption(f"{scan['tf']} | {len(summaries)}/{total} coins | data: {src_txt} | "
               f"scanned {ist(scan['at'])} IST | closed candles only")

    f1, f2 = st.columns(2)
    fresh_n = f1.slider("Fresh = signal vachhi entha candles lopu", 1, 10, 3)
    hist_n = f2.slider("Signal history (candles)", 5, 100, 30)

    ev = pd.DataFrame(scan['events'])
    if not ev.empty:
        price_now = {s["coin"]: s["price"] for s in summaries}
        ev["now"] = ev["coin"].map(price_now)
        ev["move"] = ev["side"] * (ev["now"] - ev["entry"]) / ev["entry"] * 100
        ev["result"] = ev["side"] * (ev["exit_price"] - ev["entry"]) / ev["entry"] * 100

    def live_table(df, completed=False):
        out = pd.DataFrame({
            "Coin": df["coin"],
            "Signal": df["side"].map(side_txt),
            "Signal time (IST)": df["signal_time"].map(ist),
            "Signal price": df["entry"].map(fp),
            "Stop Loss": df["sl"].map(fp),
            "Target": df["tp"].map(fp),
        })
        if completed:
            out["Result"] = df["status"].map(STATUS_TXT)
            out["Closed (IST)"] = df["exit_time"].map(ist)
            out["Exit price"] = df["exit_price"].map(fp)
            out["P/L %"] = df["result"].round(2)
        else:
            out["Price now"] = df["now"].map(fp)
            out["Move %"] = df["move"].round(2)
            out["Candles ago"] = df["candles_ago"]
        out["RSI"] = df["rsi"].round(1)
        out["24h Vol ($M)"] = (df["vol"] / 1_000_000).round(2)
        return out

    if ev.empty:
        run = fresh = done = ev
    else:
        run = ev[(ev["status"] == "RUN") & (ev["candles_ago"] <= hist_n)].sort_values("signal_time", ascending=False)
        fresh = run[run["candles_ago"] <= fresh_n]
        done = ev[(ev["status"] != "RUN") & (ev["candles_ago"] <= hist_n)].sort_values("exit_time", ascending=False)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🆕 Fresh BUY", int((fresh["side"] == 1).sum()) if len(fresh) else 0)
    m2.metric("🆕 Fresh SELL", int((fresh["side"] == -1).sum()) if len(fresh) else 0)
    m3.metric("⏳ Running", len(run))
    m4.metric("✅ Target hit", int((done["status"] == "TP").sum()) if len(done) else 0)
    m5.metric("❌ SL hit", int((done["status"] == "SL").sum()) if len(done) else 0)

    t_fresh, t_run, t_done, t_bt, t_all = st.tabs(
        ["🆕 Fresh signals", "⏳ Running", "✅ Completed", "🧪 Backtest", "📋 All coins"])

    with t_fresh:
        if len(fresh):
            st.dataframe(live_table(fresh), hide_index=True, height=min(600, 60 + 36 * len(fresh)))
        else:
            st.info(f"Last {fresh_n} candles lo kotha signal ledu. Fresh signals ippudu levu.")
        st.caption("Fresh = signal vachhi, inka target/SL hit avvaledu. Target hit ayithe 'Completed' tab ki veltundi.")

    with t_run:
        if len(run):
            st.dataframe(live_table(run), hide_index=True, height=min(600, 60 + 36 * len(run)))
        else:
            st.info("Running signals levu.")

    with t_done:
        if len(done):
            st.dataframe(live_table(done, completed=True), hide_index=True, height=min(600, 60 + 36 * len(done)))
        else:
            st.info("Ee history lo completed signals levu.")

    with t_bt:
        st.markdown("**Backtest** — scan chesina anni coins signals ni oke capital tho simulate chestundi.")
        b1, b2, b3, b4 = st.columns(4)
        capital = b1.number_input("Capital", min_value=1000.0, value=100000.0, step=10000.0, format="%.0f")
        pos_pct = b2.slider("Trade ki capital %", 5, 100, 50, 5)
        fee_pct = b3.number_input("Fee per side (%)", 0.0, 1.0, 0.10, 0.01)
        mode = b4.radio("Trades", ["BUY + SELL", "BUY only (spot)"])
        compound = st.checkbox("Compounding (profit tho size peruguthundi)", value=False)
        if ev.empty:
            st.info("Backtest ki signals levu.")
        else:
            tdf, sm, curve = run_portfolio(ev, capital, pos_pct, fee_pct, compound, mode.startswith("BUY only"))
            per_trade = pos_pct / 100 * capital
            st.caption(f"Capital {capital:,.0f} → trade ki {per_trade:,.0f} ({pos_pct}%). "
                       f"Free capital lekapothe signal skip (max {int(100 // pos_pct)} trades okesari). "
                       f"Last {scan['bars']} × {scan['tf']} candles, {len(summaries)} coins.")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Closed trades", sm["closed"])
            k2.metric("Win rate", f"{sm['win_rate']:.1f}%")
            k3.metric("Realized PnL", f"{sm['realized']:,.0f}")
            k4.metric("Return", f"{sm['ret_pct']:.2f}%")
            k5.metric("Final capital", f"{sm['final']:,.0f}")
            k6, k7, k8, k9, k10 = st.columns(5)
            k6.metric("Profit factor", "∞" if sm["pf"] == float("inf") else f"{sm['pf']:.2f}")
            k7.metric("Max drawdown", f"{sm['max_dd']:.2f}%")
            k8.metric("Target / SL / Reverse", f"{sm['tp']} / {sm['sl']} / {sm['rev']}")
            k9.metric("Still running", f"{sm['running']} ({sm['open_pnl']:,.0f})")
            k10.metric("Skipped (capital busy)", sm["skipped"])
            if not curve.empty:
                x = curve["time"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                fig = go.Figure(go.Scatter(x=x, y=curve["equity"], mode="lines+markers", line=dict(color="#00c853")))
                fig.add_hline(y=capital, line_dash="dot", line_color="gray")
                fig.update_layout(height=320, title="Equity curve (closed trades)", margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")
            if not tdf.empty:
                tbl = pd.DataFrame({
                    "Coin": tdf["coin"], "Side": tdf["side"].map(side_txt),
                    "Signal time (IST)": tdf["signal_time"].map(ist), "Entry": tdf["entry"].map(fp),
                    "Closed (IST)": tdf["exit_time"].map(ist), "Exit": tdf["exit_price"].map(fp),
                    "Result": tdf["status"].map(STATUS_TXT), "Size": tdf["size"].round(0),
                    "PnL": tdf["pnl"].round(0), "Return %": tdf["ret_pct"].round(2),
                })
                st.dataframe(tbl.iloc[::-1], hide_index=True, height=500)
            st.caption("SL/Target same candle lo rendu touch ayithe SL mundu anukuntundi (conservative). "
                       "Past results future guarantee kaadu, mundu demo lo forward test cheyyandi.")

    with t_all:
        only_signals = st.checkbox("Only BUY / SELL chupinchu", value=False)
        rows = []
        for s in summaries:
            cur = s["state"]
            e = s["cur_event"]
            rows.append({
                "Coin": s["coin"],
                "Signal now": side_txt(cur) if cur else "WAIT / NEUTRAL ⚪",
                "Signal time (IST)": ist(s["since"]),
                "Signal price": fp(s["since_price"]),
                "Price now": fp(s["price"]),
                "RSI": round(s["rsi"], 1) if not np.isnan(s["rsi"]) else None,
                "Stop Loss": fp(e["sl"]) if e else "-",
                "Target": fp(e["tp"]) if e else "-",
                "Status": STATUS_TXT[e["status"]] if e else "-",
                "24h Vol ($M)": round(s["vol"] / 1_000_000, 2),
            })
        all_df = pd.DataFrame(rows)
        if only_signals:
            all_df = all_df[all_df["Signal now"] != "WAIT / NEUTRAL ⚪"]
        all_df = all_df.sort_values("24h Vol ($M)", ascending=False)
        st.dataframe(all_df, hide_index=True, height=600)

if errors:
    all_errs = " ".join(errors.values()).lower()
    if "451" in all_errs or "restricted" in all_errs or "location" in all_errs:
        st.warning("Konni exchanges mee server location ni block chesthunnayi (Streamlit Cloud US servers). "
                   "Binance mode lo auto-fallback undi; inka fail ayithe Data source ni okx / kucoin ki marchandi.")
    with st.expander(f"⚠️ {len(errors)} coins fail ayyayi — details"):
        for coin, e in errors.items():
            st.write(f"**{coin}** → {e}")
