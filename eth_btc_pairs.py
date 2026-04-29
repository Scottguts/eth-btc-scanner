"""
ETH vs BTC pairs-trading signal
================================

Pulls daily closes for ETHUSDT and BTCUSDT from Binance's public API,
computes the log-ratio log(ETH/BTC), z-scores it against a rolling
window, and emits a simple mean-reversion signal:

    position = +1  (long ETH / short BTC)   when z < -ENTRY_Z
    position = -1  (short ETH / long BTC)   when z > +ENTRY_Z
    position =  0  (flat)                    when |z| < EXIT_Z after being in a trade
    hard stop                                when |z| > STOP_Z

It also runs a simple dollar-neutral backtest on the log-ratio,
prints today's signal, saves the full timeseries to CSV, and
writes a diagnostic chart.

THIS IS NOT FINANCIAL ADVICE. It's a signal generator to study.
No order execution — that's intentional. Paper-trade it first.

Usage:
    # one-shot: print current signal and exit
    python eth_btc_pairs.py

    # live poller (Binance): re-check every 15 minutes, alert on position flips
    python eth_btc_pairs.py --loop
    python eth_btc_pairs.py --loop --interval-minutes 15
    python eth_btc_pairs.py --loop --interval-minutes 5 --kline 1h

    # alternate source: CoinMarketCap (free tier only gives latest quotes,
    # so the script accumulates its own history locally; signal goes live
    # once ROLLING_WINDOW+1 samples are collected)
    export CMC_API_KEY=your_key_here
    python eth_btc_pairs.py --source cmc --loop --interval-minutes 15

    # email (or email-to-SMS gateway) alerts on position flips
    export SMTP_HOST=smtp.gmail.com SMTP_USER=you@gmail.com SMTP_PASS=app_pw
    export ALERT_TO="you@gmail.com,5555551234@tmomail.net"
    python eth_btc_pairs.py --test-alert             # send one test email
    python eth_btc_pairs.py --loop --alert-email     # alert on every flip

    # Telegram alerts on position flips (recommended: easier + more reliable)
    export TELEGRAM_BOT_TOKEN=123:abc...
    python eth_btc_pairs.py --telegram-find-chat-id  # after messaging bot once
    export TELEGRAM_CHAT_ID=582374991
    python eth_btc_pairs.py --test-alert             # send one test
    python eth_btc_pairs.py --loop --alert-telegram  # alert on every flip
"""

from __future__ import annotations

import sys
import os
import time
import math
import json
import signal as _signal
import argparse
from datetime import datetime, timezone, timedelta

import smtplib
import ssl as _ssl
import pickle
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# v2 components live in the analysis/ package alongside training and audit
# scripts. They are imported lazily / optionally so the bot still runs
# (without auto-regime / ML overlay) on a stripped install.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
    from regime import detect_regime, RegimeBands  # type: ignore
    from features import build_features  # type: ignore
    _V2_AVAILABLE = True
except Exception as _v2_err:  # pragma: no cover - best-effort import
    _V2_AVAILABLE = False
    _V2_IMPORT_ERR = _v2_err

# ---------- CONFIG ----------
# Defaults below were chosen by walk-forward backtest across 102 (timeframe x
# window x entry_z x direction) combinations on 14 months of Binance data.
# See analysis/sweep.py and analysis/research_report.md for the full audit.
#
#   Winner:  kline=4h  window=360  entry_z=2.5  direction=momentum
#            walk-forward Sharpe +1.26, total return +157.7%, MaxDD -19.5%.
#
# Mean-reversion configurations all lost money walk-forward on this period
# (Sharpe -0.22 to -3.85). The cointegration test rejects cointegration
# (Engle-Granger p=0.73), which is consistent with momentum > mean-reversion
# on this regime. Past performance is not a guarantee — paper-trade first.

LOOKBACK_LIMIT = 1000         # Binance max candles per request
INTERVAL       = "4h"         # "1h" / "4h" / "1d"
ROLLING_WINDOW = 360          # z-score lookback (bars). 360 4h bars = 60 days.
ENTRY_Z        = 2.5          # enter when |z| > entry_z
EXIT_Z         = 0.3          # exit back toward mean
STOP_Z         = 3.5          # hard stop
FEE_BPS        = 4.0          # 0.04% per leg per trade (perp taker-ish)
MODE           = "auto"       # "auto" (default; regime detector picks),
                              # "momentum", or "mean-revert".
OUT_DIR        = "."          # write outputs next to the script

# v2 model path (saved by analysis/v2_backtest.py). Used only when --use-ml
# is passed.
ML_MODEL_PATH  = os.path.join(os.path.dirname(__file__),
                              "analysis", "model.pkl")
ML_THRESHOLD   = 0.50         # below this we veto an entry (--use-ml only)

# CoinMarketCap (optional alternate source; free tier = latest quotes only)
CMC_BASE       = "https://pro-api.coinmarketcap.com"
CMC_HISTORY    = f"{OUT_DIR}/eth_btc_cmc_history.csv"
# ----------------------------


def fetch_cmc_latest(api_key: str,
                     symbols: tuple[str, ...] = ("BTC", "ETH")) -> dict:
    """Fetch latest USD quotes from CoinMarketCap (free tier friendly).

    Returns {"BTC": price, "ETH": price, "ts": pandas.Timestamp (UTC)}.
    """
    url = f"{CMC_BASE}/v2/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"}
    params  = {"symbol": ",".join(symbols), "convert": "USD"}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status", {}).get("error_code", 0) != 0:
        raise RuntimeError(f"CMC error: {payload['status']}")
    out = {}
    for sym in symbols:
        # /v2 returns a list per symbol; take the first active entry
        entries = payload["data"].get(sym, [])
        if not entries:
            raise RuntimeError(f"CMC returned no data for {sym}")
        out[sym] = float(entries[0]["quote"]["USD"]["price"])
    out["ts"] = pd.Timestamp.now(tz="UTC").floor("s")
    return out


def cmc_append_history(snap: dict, path: str = CMC_HISTORY) -> pd.DataFrame:
    """Append a snapshot to the persistent CMC history CSV and return the full series."""
    row = pd.DataFrame(
        [{"time": snap["ts"], "btc": snap["BTC"], "eth": snap["ETH"]}]
    ).set_index("time")
    if os.path.exists(path):
        prior = pd.read_csv(path, index_col="time", parse_dates=True)
        # tz-normalize just in case
        if prior.index.tz is None:
            prior.index = prior.index.tz_localize("UTC")
        df = pd.concat([prior, row])
    else:
        df = row
    # de-dupe on minute-floor (avoid runaway rows if poller runs often)
    df = df[~df.index.floor("min").duplicated(keep="last")]
    df = df.sort_index()
    df.to_csv(path)
    return df


# Try binance.com first (better liquidity, more history). If a given host
# geo-blocks us with 451/403, fall through to the next one and remember it.
BINANCE_BASES = ["https://api.binance.com", "https://api.binance.us"]


def fetch_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV klines from Binance public REST API (no auth).

    Tries binance.com first; on 451 ("Unavailable For Legal Reasons", the US
    geo-block) or 403, falls back to binance.us. Caches the working host so
    subsequent calls go straight there without paying the retry latency.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_err: Exception | None = None
    for base in list(BINANCE_BASES):
        try:
            r = requests.get(f"{base}/api/v3/klines",
                             params=params, timeout=30)
            r.raise_for_status()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (451, 403, 418):
                last_err = e
                continue  # try next host
            raise
        except requests.RequestException as e:
            last_err = e
            continue
        # Success — remember this host for the rest of the process.
        if BINANCE_BASES[0] != base:
            BINANCE_BASES.remove(base)
            BINANCE_BASES.insert(0, base)
        rows = r.json()
        cols = ["openTime", "open", "high", "low", "close", "volume",
                "closeTime", "qav", "trades", "tbav", "tbqv", "ignore"]
        df = pd.DataFrame(rows, columns=cols)
        df["time"]   = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        # Volume is needed by the v2 ML feature builder; safe to ignore
        # downstream when --use-ml is off.
        return (df.set_index("time")[["close", "volume"]]
                  .rename(columns={"close": symbol,
                                   "volume": f"{symbol}_vol"}))

    raise last_err or RuntimeError("all Binance endpoints failed")


def compute_zscore(ratio: pd.Series, window: int) -> pd.Series:
    log_r = np.log(ratio)
    mean  = log_r.rolling(window).mean()
    std   = log_r.rolling(window).std()
    # Guard against zero std (e.g., a stuck price) producing inf z-scores
    # that would falsely trigger entries / stops.
    std   = std.where(std > 0)
    return (log_r - mean) / std


def generate_positions(
    z: pd.Series,
    entry: float = ENTRY_Z,
    exit_:  float = EXIT_Z,
    stop:   float = STOP_Z,
    mode:  str   = MODE,
    regime: pd.Series | None = None,
) -> pd.Series:
    """State machine. Same |z| > entry triggers an entry in any mode; the
    *direction* of that entry depends on the mode (or, in "auto" mode, on
    the regime label at the bar where we go from flat to active).

    mode = "mean-revert"  (classical pairs-trading; bet on revert to mean):
        z < -entry  -> +1 (long ratio,  long ETH / short BTC)
        z > +entry  -> -1 (short ratio, short ETH / long BTC)
    mode = "momentum"      (bet that the dislocation continues; current default
                            on the audited 32-month sample):
        z < -entry  -> -1 (short ratio, betting it keeps falling)
        z > +entry  -> +1 (long ratio,  betting it keeps rising)
    mode = "auto"          (regime detector picks momentum or mean-revert per
                            entry; "indeterminate" regime stays flat). Requires
                            `regime` series aligned to z.

    Exits use the same |z| < exit_ corridor and |z| > stop hard stop in all
    modes; only the entry direction varies.
    """
    if mode not in ("momentum", "mean-revert", "auto"):
        raise ValueError(f"mode must be 'momentum'/'mean-revert'/'auto', "
                         f"got {mode!r}")
    if mode == "auto" and regime is None:
        raise ValueError("mode='auto' requires a regime series")

    pos = np.zeros(len(z))
    current = 0
    z_vals = z.values
    r_vals = regime.values if regime is not None else None
    # 'last_sign' lets us hold a trade through a regime change in the middle
    # of a position (don't yank a trade just because the regime label flipped
    # mid-bar; only fresh entries respect the new regime).
    for i in range(len(z)):
        zt = z_vals[i]
        if np.isnan(zt):
            pos[i] = current
            continue
        if current == 0:
            if mode == "auto":
                reg = r_vals[i] if r_vals is not None else "indeterminate"
                if reg == "momentum":
                    sign = +1
                elif reg == "mean-revert":
                    sign = -1
                else:
                    pos[i] = 0
                    continue
            else:
                sign = +1 if mode == "momentum" else -1
            if zt >  entry:
                current = +sign
            elif zt < -entry:
                current = -sign
        elif current != 0 and (abs(zt) < exit_ or abs(zt) > stop):
            current = 0
        pos[i] = current
    return pd.Series(pos, index=z.index, name="position")


# ---------- v2 ML gate (optional) ----------

_ML_BUNDLE_CACHE: dict | None = None


def _load_ml_bundle(path: str = ML_MODEL_PATH) -> dict | None:
    """Lazy-load the pickled classifier produced by analysis/v2_backtest.py.
    Returns None if the file is missing — the bot stays usable without it."""
    global _ML_BUNDLE_CACHE
    if _ML_BUNDLE_CACHE is not None:
        return _ML_BUNDLE_CACHE
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            _ML_BUNDLE_CACHE = pickle.load(f)
    except Exception as e:
        print(f"[ml] WARN: failed to load model: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None
    return _ML_BUNDLE_CACHE


def ml_gate_latest(df: pd.DataFrame, z: pd.Series, regime: pd.Series | None,
                   threshold: float = ML_THRESHOLD) -> tuple[bool, float | None]:
    """Score the latest bar with the saved classifier and decide whether a
    fresh entry should be taken. Returns (allow, proba). If the model isn't
    available we return (True, None) so the bot proceeds unchanged."""
    bundle = _load_ml_bundle()
    if bundle is None or not _V2_AVAILABLE:
        return True, None
    model    = bundle["model"]
    feat_cols = bundle["feature_cols"]
    try:
        feats = build_features(df, z=z, regime=regime,
                               hurst=df.get("hurst"),
                               adf_p=df.get("adf_p"))
    except Exception as e:
        print(f"[ml] WARN: feature build failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return True, None
    if not feats.empty:
        last = feats.iloc[[-1]].reindex(columns=feat_cols)
        # HistGradientBoosting handles NaN natively, so we don't refuse to
        # score on missing features. We *do* require the most-load-bearing
        # column ('z') to be finite, since a NaN z means the rolling window
        # hasn't filled yet.
        if "z" in last.columns and not np.isfinite(last["z"].iloc[0]):
            return True, None
        try:
            proba = float(model.predict_proba(last.values)[:, 1][0])
        except Exception as e:
            print(f"[ml] WARN: predict failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return True, None
        return (proba >= threshold), proba
    return True, None


def backtest(eth: pd.Series, btc: pd.Series, pos: pd.Series,
             fee_bps: float = FEE_BPS) -> pd.DataFrame:
    """
    Dollar-neutral: P&L = position_{t-1} * d log(ETH/BTC)_t
    Fees charged on every position *change*, applied across 2 legs.
    """
    log_ratio = np.log(eth / btc)
    ratio_ret = log_ratio.diff().fillna(0.0)

    gross = pos.shift(1).fillna(0.0) * ratio_ret

    # Turnover = |pos_t - pos_{t-1}|, fee hits 2 legs
    turnover = pos.diff().abs().fillna(0.0)
    fee_cost = turnover * (fee_bps / 1e4) * 2.0

    net = gross - fee_cost
    equity = net.cumsum().apply(np.exp)
    return pd.DataFrame({
        "ratio_ret":  ratio_ret,
        "gross_ret":  gross,
        "fee":        fee_cost,
        "net_ret":    net,
        "equity":     equity,
    })


def extract_trades(df: pd.DataFrame, fee_bps: float = FEE_BPS) -> pd.DataFrame:
    """Walk the position series and return one row per pairs trade.

    A trade starts when `pos` flips away from 0, and ends when `pos` returns to
    0 (or flips sign — a flip-without-flat is split into two trades). The last
    trade is left open if `pos` is non-zero at the end of the series.

    Returned columns:
      entry_time, exit_time, position (+1/-1), bars_held,
      entry_eth, entry_btc, entry_ratio, entry_z,
      exit_eth,  exit_btc,  exit_ratio,  exit_z,
      log_pnl, pct_pnl, fees_bps_total, status ("CLOSED" or "OPEN")
    """
    rows = []
    pos = df["pos"].fillna(0).astype(int).values
    times = df.index.to_list()
    eth = df["eth"].values
    btc = df["btc"].values
    ratio = df["ratio"].values
    z = df["z"].values

    open_trade = None  # dict, accumulates while a trade is live

    def _close(idx: int, status: str = "CLOSED") -> None:
        nonlocal open_trade
        if open_trade is None:
            return
        side = open_trade["position"]
        log_pnl = side * (math.log(ratio[idx]) - math.log(open_trade["entry_ratio"]))
        # Two legs in, two legs out -> 4x bps; keep gross + fee separate.
        fees_total_bps = fee_bps * 4.0
        pct_pnl = (math.exp(log_pnl) - 1.0) * 100.0
        rows.append({
            **open_trade,
            "exit_time":  times[idx],
            "exit_eth":   float(eth[idx]),
            "exit_btc":   float(btc[idx]),
            "exit_ratio": float(ratio[idx]),
            "exit_z":     float(z[idx]),
            "bars_held":  idx - open_trade["_entry_idx"],
            "log_pnl":    float(log_pnl),
            "pct_pnl":    float(pct_pnl),
            "fees_bps_total": fees_total_bps,
            "status":     status,
        })
        open_trade = None

    for i in range(1, len(pos)):
        prev, cur = pos[i - 1], pos[i]
        if prev == 0 and cur != 0:           # entry from flat
            open_trade = {
                "_entry_idx":  i,
                "entry_time":  times[i],
                "position":    int(cur),
                "entry_eth":   float(eth[i]),
                "entry_btc":   float(btc[i]),
                "entry_ratio": float(ratio[i]),
                "entry_z":     float(z[i]),
            }
        elif prev != 0 and cur == 0:         # exit to flat
            _close(i, status="CLOSED")
        elif prev != 0 and cur != 0 and prev != cur:
            # direction flip without going flat — close at this bar, open new
            _close(i, status="CLOSED")
            open_trade = {
                "_entry_idx":  i,
                "entry_time":  times[i],
                "position":    int(cur),
                "entry_eth":   float(eth[i]),
                "entry_btc":   float(btc[i]),
                "entry_ratio": float(ratio[i]),
                "entry_z":     float(z[i]),
            }

    if open_trade is not None:
        # Mark the live trade as still open, using the last bar as a snapshot.
        last = len(pos) - 1
        side = open_trade["position"]
        log_pnl = side * (math.log(ratio[last]) - math.log(open_trade["entry_ratio"]))
        rows.append({
            **open_trade,
            "exit_time":  None,
            "exit_eth":   None,
            "exit_btc":   None,
            "exit_ratio": None,
            "exit_z":     None,
            "bars_held":  last - open_trade["_entry_idx"],
            "log_pnl":    float(log_pnl),
            "pct_pnl":    float((math.exp(log_pnl) - 1.0) * 100.0),
            "fees_bps_total": fee_bps * 2.0,  # only entry legs paid so far
            "status":     "OPEN",
        })

    if not rows:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "position", "bars_held",
            "entry_eth", "entry_btc", "entry_ratio", "entry_z",
            "exit_eth", "exit_btc", "exit_ratio", "exit_z",
            "log_pnl", "pct_pnl", "fees_bps_total", "status",
        ])
    out = pd.DataFrame(rows).drop(columns=["_entry_idx"])
    return out


def perf_stats(net_ret: pd.Series, pos: pd.Series,
               periods_per_year: float) -> dict:
    """Compute summary stats. ``pos`` is needed to count real trades and to
    restrict hit-rate to bars where we were actually in a trade (otherwise
    flat bars dilute the metric and make it look much worse than it is).
    """
    r = net_ret.dropna()
    if r.std() == 0 or len(r) < 2:
        return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0,
                "hit_rate": 0.0, "n_trades": 0, "n_bars_active": 0,
                "pct_time_in_trade": 0.0}
    mu   = r.mean() * periods_per_year
    sig  = r.std() * math.sqrt(periods_per_year)
    eq   = np.exp(r.cumsum())
    peak = eq.cummax()
    dd   = (eq / peak - 1).min()

    active = pos.shift(1).fillna(0.0).reindex(r.index).fillna(0.0).abs() > 0
    n_bars_active = int(active.sum())
    hit_rate = float((r[active] > 0).mean()) if n_bars_active else 0.0
    # A "trade" is a non-zero run of position; count entries from a flat bar.
    p = pos.fillna(0.0).astype(int).values
    n_trades = 0
    prev = 0
    for v in p:
        if v != 0 and prev == 0:
            n_trades += 1
        prev = v
    return {
        "sharpe":            mu / sig if sig else 0.0,
        "cagr":              float(np.exp(mu) - 1),
        "max_dd":            float(dd),
        "hit_rate":          hit_rate,
        "n_trades":          n_trades,
        "n_bars_active":     n_bars_active,
        "pct_time_in_trade": float(active.mean()),
    }


def describe_signal(z_now: float, pos_now: float, mode: str = MODE) -> str:
    """One-line summary of what the bot thinks you should be doing right now.
    The position interpretation is identical across modes (+1 = long ETH /
    short BTC); only the *reasoning* differs."""
    if pos_now == +1:
        if mode == "momentum":
            return (f"IN TRADE: long ETH / short BTC (ratio breaking up, "
                    f"z={z_now:+.2f}). Hold until z falls back inside the "
                    f"exit band or hits the stop.")
        return (f"IN TRADE: long ETH / short BTC (ETH cheap vs BTC, "
                f"z={z_now:+.2f}). Hold until z returns toward 0.")
    if pos_now == -1:
        if mode == "momentum":
            return (f"IN TRADE: short ETH / long BTC (ratio breaking down, "
                    f"z={z_now:+.2f}). Hold until z rises back inside the "
                    f"exit band or hits the stop.")
        return (f"IN TRADE: short ETH / long BTC (ETH rich vs BTC, "
                f"z={z_now:+.2f}). Hold until z returns toward 0.")
    if z_now >  ENTRY_Z:
        side = "LONG ETH / SHORT BTC" if mode == "momentum" else "SHORT ETH / LONG BTC"
        return (f"FLAT, watching: z={z_now:+.2f} above +{ENTRY_Z}. "
                f"Next bar may trigger {side} ({mode}).")
    if z_now < -ENTRY_Z:
        side = "SHORT ETH / LONG BTC" if mode == "momentum" else "LONG ETH / SHORT BTC"
        return (f"FLAT, watching: z={z_now:+.2f} below -{ENTRY_Z}. "
                f"Next bar may trigger {side} ({mode}).")
    return (f"FLAT, no edge: z={z_now:+.2f} inside +/-{ENTRY_Z}. "
            f"Wait for divergence ({mode} mode).")


STATE_FILE = f"{OUT_DIR}/eth_btc_state.json"


# ========================= Telegram alerts ============================

TELEGRAM_BASE = "https://api.telegram.org"


def _load_telegram_config() -> dict | None:
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    cid = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok:
        return None
    return {"token": tok, "chat_id": cid}


def send_telegram_alert(subject: str, body: str, cfg: dict,
                        timeout: float = 20.0) -> None:
    """Send a Markdown-formatted message via Telegram Bot API."""
    if not cfg.get("chat_id"):
        raise RuntimeError("TELEGRAM_CHAT_ID not set. Run "
                           "`python eth_btc_pairs.py --telegram-find-chat-id` "
                           "after messaging your bot once.")
    # Escape minimal Markdown chars that would blow up the parser in our payload.
    # We use the legacy "Markdown" mode to avoid MarkdownV2's aggressive escaping.
    text = f"*{subject}*\n\n```\n{body}```"
    url  = f"{TELEGRAM_BASE}/bot{cfg['token']}/sendMessage"
    r = requests.post(url, json={
        "chat_id": cfg["chat_id"],
        "text":    text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }, timeout=timeout)
    if r.status_code != 200 or not r.json().get("ok"):
        raise RuntimeError(f"Telegram API error: {r.status_code} {r.text[:200]}")


def telegram_find_chat_id(token: str, timeout: float = 20.0) -> dict[int, str]:
    """Call getUpdates and return {chat_id: label} for any chats that have
    messaged the bot recently. User must have tapped Start / sent /start first.
    """
    url = f"{TELEGRAM_BASE}/bot{token}/getUpdates"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    chats: dict[int, str] = {}
    for u in r.json().get("result", []):
        msg = u.get("message") or u.get("edited_message") or u.get("channel_post") or {}
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        if cid is None:
            continue
        label = (f"{chat.get('first_name','')} {chat.get('last_name','')}".strip()
                 or chat.get("username") or chat.get("title") or "unnamed")
        chats[cid] = label
    return chats


# ========================= email / SMS alerts =========================

def _load_alert_config() -> dict | None:
    """Build alert config from env vars. Returns None if not configured.

    Required env vars:
      SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_TO
    Optional:
      SMTP_PORT (default 587)
      SMTP_SSL  ('1' => SMTPS; else STARTTLS on port 587)
      ALERT_FROM (defaults to SMTP_USER)

    ALERT_TO is comma-separated. Entries can be regular emails or
    email-to-SMS gateway addresses such as:
        5555551234@tmomail.net     (T-Mobile)
        5555551234@vtext.com       (Verizon — flaky; many carriers
                                    have been deprecating these)
        5555551234@msg.fi.google.com (Google Fi)
    Delivery to carrier gateways is unreliable by 2024-2026 — prefer
    a normal email inbox on your phone if at all possible.
    """
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    pwd  = os.environ.get("SMTP_PASS")
    to   = os.environ.get("ALERT_TO")
    if not (host and user and pwd and to):
        return None
    return {
        "host":       host,
        "port":       int(os.environ.get("SMTP_PORT", "587")),
        "user":       user,
        "password":   pwd,
        "use_ssl":    os.environ.get("SMTP_SSL", "0") == "1",
        "sender":     os.environ.get("ALERT_FROM", user),
        "recipients": [r.strip() for r in to.split(",") if r.strip()],
    }


def send_email_alert(subject: str, body: str, cfg: dict,
                     timeout: float = 20.0) -> None:
    """Send a short alert. Subject is kept brief so it fits in an SMS gateway.

    Raises on failure — caller should wrap in try/except in a hot loop.
    """
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject[:120]                 # gateway-friendly
    msg["From"]    = cfg["sender"]
    msg["To"]      = ", ".join(cfg["recipients"])
    msg["Date"]    = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()

    if cfg["use_ssl"] or cfg["port"] == 465:
        ctx = _ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"],
                              timeout=timeout, context=ctx) as srv:
            srv.login(cfg["user"], cfg["password"])
            srv.send_message(msg)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=timeout) as srv:
            srv.ehlo()
            srv.starttls(context=_ssl.create_default_context())
            srv.ehlo()
            srv.login(cfg["user"], cfg["password"])
            srv.send_message(msg)


def _format_alert(snap: dict, prev_pos: int) -> tuple[str, str]:
    """Format an alert: short subject (SMS/Telegram-friendly) and a fuller body
    that tells the reader what to do, at what prices, and where the targets and
    stops sit. Body is plain text so it renders cleanly inside Telegram code
    blocks and email-to-SMS gateways.
    """
    pos = snap["position"]
    tag = {1: "LONG ETH / SHORT BTC",
           -1: "SHORT ETH / LONG BTC",
           0: "CLOSE TRADE / FLAT"}[pos]
    direction = {1: "OPEN", -1: "OPEN", 0: "CLOSE"}[pos]
    subject = f"ETH/BTC {direction}: {tag} | z={snap['zscore']:+.2f}"

    eth   = snap["eth"]
    btc   = snap["btc"]
    ratio = snap["ratio"]
    z     = snap["zscore"]
    target = snap.get("target_ratio", float("nan"))
    stop_up = snap.get("stop_up_ratio", float("nan"))
    stop_dn = snap.get("stop_dn_ratio", float("nan"))
    move_pct = snap.get("pct_to_mean", float("nan"))

    # Body format — fixed-width so it lines up in monospace (Telegram code block
    # / email-to-SMS).
    lines = []
    lines.append(f"SIGNAL FLIP: {prev_pos:+d}  ->  {pos:+d}   ({tag})")
    lines.append(f"Time:        {snap['asof']}")
    lines.append("")

    # `effective_mode` is the actual direction the bot took at this bar
    # (auto-mode resolves to momentum or mean-revert via the regime detector).
    mode = snap.get("effective_mode") or snap.get("mode", MODE)
    if pos == +1:
        if mode == "momentum":
            lines.append("WHY:   ETH/BTC ratio is breaking up; betting on continuation.")
        else:
            lines.append("WHY:   ETH looks cheap vs BTC (z below entry).")
        lines.append("DO:    Buy ETH, sell BTC, equal dollars per leg.")
    elif pos == -1:
        if mode == "momentum":
            lines.append("WHY:   ETH/BTC ratio is breaking down; betting on continuation.")
        else:
            lines.append("WHY:   ETH looks rich vs BTC (z above entry).")
        lines.append("DO:    Sell ETH, buy BTC, equal dollars per leg.")
    else:
        if prev_pos == +1:
            lines.append("WHY:   z came back inside exit band (or stop hit).")
            lines.append("DO:    Close ETH long, close BTC short.")
        elif prev_pos == -1:
            lines.append("WHY:   z came back inside exit band (or stop hit).")
            lines.append("DO:    Close ETH short, close BTC long.")
        else:
            lines.append("DO:    Flatten any open pairs trade.")
    lines.append("")

    lines.append("PRICES")
    lines.append(f"  ETH         ${eth:>12,.2f}")
    lines.append(f"  BTC         ${btc:>12,.2f}")
    lines.append(f"  ETH/BTC     {ratio:>13.6f}")
    lines.append(f"  Z-score     {z:>+13.3f}")
    lines.append("")

    if pos != 0:
        lines.append("TARGETS  (where to act)")
        lines.append(f"  Take profit  ratio ~ {target:.6f}   "
                     f"(z = 0,  {move_pct:+.2f}% from now)")
        if pos == +1:
            lines.append(f"  Hard stop    ratio < {stop_dn:.6f}   "
                         f"(z = {-STOP_Z:+.1f})")
        else:
            lines.append(f"  Hard stop    ratio > {stop_up:.6f}   "
                         f"(z = {+STOP_Z:+.1f})")
        lines.append(f"  Soft exit    |z| < {EXIT_Z}")
        lines.append("")

    lines.append(f"Source: {snap.get('source', 'binance')} "
                 f"({snap.get('kline', 'n/a')} bars, {ROLLING_WINDOW}-bar window)")
    lines.append("This is a signal, not an order. Review before trading.")

    body = "\n".join(lines) + "\n"
    return subject, body


# ======================================================================


def run_once(kline: str = INTERVAL, source: str = "binance",
             cmc_api_key: str | None = None,
             verbose: bool = True, write_artifacts: bool = True,
             mode: str = MODE,
             use_ml: bool = False) -> dict:
    """Fetch latest data, compute signal, optionally write CSV/PNG/JSON.

    source = "binance": full historical klines (instant backtest).
    source = "cmc":     polls CMC latest, accumulates local CSV history.
                        Needs ROLLING_WINDOW samples before the signal goes live.

    Returns a dict snapshot of the latest row + stats.
    """
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')

    if source == "cmc":
        if not cmc_api_key:
            raise RuntimeError("CMC source requires CMC_API_KEY env var or --cmc-key")
        if verbose:
            print(f"[{now_str}] Fetching latest BTC+ETH from CoinMarketCap...")
        snap = fetch_cmc_latest(cmc_api_key)
        df = cmc_append_history(snap)
        if verbose:
            print(f"CMC history samples: {len(df)} "
                  f"(need {ROLLING_WINDOW+1} for z-score)")
    else:
        if verbose:
            print(f"[{now_str}] Fetching BTCUSDT + ETHUSDT from Binance ({kline})...")
        btc = fetch_klines("BTCUSDT", kline, LOOKBACK_LIMIT)
        eth = fetch_klines("ETHUSDT", kline, LOOKBACK_LIMIT)
        df = pd.concat([btc, eth], axis=1).dropna(subset=["BTCUSDT", "ETHUSDT"])
        df = df.rename(columns={
            "BTCUSDT":     "btc", "BTCUSDT_vol": "btc_vol",
            "ETHUSDT":     "eth", "ETHUSDT_vol": "eth_vol",
        })

    df["ratio"]   = df["eth"] / df["btc"]
    log_r         = np.log(df["ratio"])
    df["log_mu"]  = log_r.rolling(ROLLING_WINDOW).mean()
    df["log_sd"]  = log_r.rolling(ROLLING_WINDOW).std().where(
        log_r.rolling(ROLLING_WINDOW).std() > 0)
    df["z"]       = (log_r - df["log_mu"]) / df["log_sd"]

    # --- v2 regime detection (only if auto-regime is requested) ---
    regime_series = None
    effective_mode = mode
    if mode == "auto":
        if not _V2_AVAILABLE:
            if verbose:
                print(f"[regime] WARN: v2 modules not available "
                      f"({_V2_IMPORT_ERR!r}); falling back to MODE='momentum'.")
            effective_mode = "momentum"
        else:
            try:
                # The live bot keeps `ratio` rather than `log_r`; the regime
                # detector wants the log spread.
                log_r_for_regime = np.log(df["ratio"])
                rg = detect_regime(log_r_for_regime, hurst_window=240,
                                   adf_window=240, adf_step=4,
                                   bands=RegimeBands(
                                       h_trend=0.53, h_meanrev=0.47,
                                       adf_trend=0.15, adf_meanrev=0.08,
                                       hysteresis_bars=4,
                                       initial_regime="momentum"))
                df["hurst"]  = rg["hurst"]
                df["adf_p"]  = rg["adf_p"]
                df["regime"] = rg["regime"]
                regime_series = rg["regime"]
            except Exception as e:
                if verbose:
                    print(f"[regime] WARN: detection failed "
                          f"({type(e).__name__}: {e}); falling back to "
                          f"MODE='momentum'.")
                effective_mode = "momentum"

    df["pos"]     = generate_positions(df["z"], mode=effective_mode,
                                       regime=regime_series)

    # If we ran in auto-mode, surface the direction the regime detector picked
    # at the latest bar so downstream output (and `effective_mode` in the snap)
    # is always a concrete momentum/mean-revert label rather than "auto".
    if mode == "auto" and regime_series is not None and len(regime_series):
        latest_regime = str(regime_series.iloc[-1])
        if latest_regime in ("momentum", "mean-revert"):
            effective_mode = latest_regime
        elif latest_regime == "indeterminate":
            effective_mode = "indeterminate"

    # --- v2 ML gate (opt-in; vetoes the most recent fresh entry only) ---
    ml_proba = None
    ml_vetoed = False
    if use_ml:
        last_idx = df.index[-1] if len(df) else None
        if last_idx is not None and len(df) >= 2:
            prev_pos = float(df["pos"].iloc[-2]) if len(df) >= 2 else 0.0
            cur_pos  = float(df["pos"].iloc[-1])
            if cur_pos != 0 and prev_pos == 0:    # fresh entry on this bar
                allow, ml_proba = ml_gate_latest(df, df["z"], regime_series)
                if not allow:
                    ml_vetoed = True
                    df.loc[last_idx, "pos"] = 0.0   # veto
                    if verbose:
                        print(f"[ml] entry VETOED (proba={ml_proba:.3f} "
                              f"< {ML_THRESHOLD})")
                elif verbose and ml_proba is not None:
                    print(f"[ml] entry allowed (proba={ml_proba:.3f})")

    # Backtest only makes sense when we have enough history and a clear bar cadence.
    if source == "cmc" and len(df) < ROLLING_WINDOW + 2:
        bootstrap = {
            "asof":       str(df.index[-1]) if len(df) else now_str,
            "btc":        float(df["btc"].iloc[-1]) if len(df) else float("nan"),
            "eth":        float(df["eth"].iloc[-1]) if len(df) else float("nan"),
            "ratio":      float(df["ratio"].iloc[-1]) if len(df) else float("nan"),
            "zscore":     float("nan"),
            "position":   0,
            "signal":     (f"Bootstrapping local history: {len(df)}/{ROLLING_WINDOW+1} "
                          f"samples — signal goes live once window is full."),
            "stats":      {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0,
                           "hit_rate": 0.0, "n_trades": 0,
                           "n_bars_active": 0, "pct_time_in_trade": 0.0},
            "kline":      kline,
            "source":     source,
        }
        if verbose:
            print("=" * 58)
            print(bootstrap["signal"])
            print("=" * 58)
        return bootstrap

    bt = backtest(df["eth"], df["btc"], df["pos"])
    df = df.join(bt)

    ppy   = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24,
             "15m": 365 * 24 * 4, "5m": 365 * 24 * 12}.get(kline, 365)
    stats = perf_stats(df["net_ret"], df["pos"], periods_per_year=ppy)
    latest = df.dropna().iloc[-1]

    mu_ln  = float(latest["log_mu"])
    sd_ln  = float(latest["log_sd"])
    target_ratio = float(np.exp(mu_ln))                       # z = 0
    stop_up      = float(np.exp(mu_ln + STOP_Z * sd_ln))      # z = +stop
    stop_dn      = float(np.exp(mu_ln - STOP_Z * sd_ln))      # z = -stop
    cur_ratio    = float(latest["ratio"])
    pct_to_mean  = (target_ratio / cur_ratio - 1.0) * 100.0   # signed

    snap = {
        "asof":          str(latest.name),
        "btc":           float(latest["btc"]),
        "eth":           float(latest["eth"]),
        "ratio":         cur_ratio,
        "zscore":        float(latest["z"]),
        "position":      int(latest["pos"]),
        "target_ratio":  target_ratio,
        "stop_up_ratio": stop_up,
        "stop_dn_ratio": stop_dn,
        "pct_to_mean":   pct_to_mean,
        "signal":        describe_signal(latest["z"], latest["pos"],
                                          mode=effective_mode),
        "stats":         stats,
        "kline":         kline,
        "source":        source,
        "mode":          mode,
        "effective_mode": effective_mode if mode == "auto" else mode,
        "regime":        (str(regime_series.iloc[-1])
                          if regime_series is not None else None),
        "hurst":         (float(df["hurst"].iloc[-1])
                          if "hurst" in df else None),
        "adf_p":         (float(df["adf_p"].iloc[-1])
                          if "adf_p" in df else None),
        "ml_proba":      ml_proba,
        "ml_vetoed":     ml_vetoed,
    }

    if verbose:
        print("=" * 64)
        # Walk-forward audit numbers from analysis/sweep.py (4h/360/2.5/momentum
        # on 14 months of Binance data). These are the REAL out-of-sample
        # performance numbers; the in-sample backtest below is on a shorter
        # window and is shown only as a sanity-check, not as a return forecast.
        print("Walk-forward audit (4h/360/2.5/momentum, 14mo OOS):")
        print("    Sharpe +1.26 | Total return +157.7% | MaxDD -19.5% | "
              "41 trades")
        print("    Mean-reversion variants all negative (Sharpe -0.22 to -3.85).")
        print("    See analysis/research_report.md for the full sweep.")
        print("-" * 64)
        if mode == "auto":
            r = snap.get("regime") or "n/a"
            h = snap.get("hurst")
            ap = snap.get("adf_p")
            h_s = f"{h:+.3f}" if h is not None else "n/a"
            ap_s = f"{ap:.3f}" if ap is not None else "n/a"
            print(f"Mode:           {mode.upper():>13s}   "
                  f"(regime: {r}, Hurst {h_s}, ADF p {ap_s})")
            print(f"Effective:      {effective_mode.upper():>13s}")
        else:
            print(f"Mode:           {mode.upper():>13s}   "
                  f"(walk-forward winner; see analysis/research_report.md)")
        if use_ml:
            if ml_proba is None:
                print("ML gate:        not active this bar (no entry to score)")
            elif ml_vetoed:
                print(f"ML gate:        VETOED entry (proba {ml_proba:.3f} "
                      f"< {ML_THRESHOLD})")
            else:
                print(f"ML gate:        ALLOWED entry (proba {ml_proba:.3f})")
        print(f"As of:          {latest.name}")
        print(f"BTC close:      ${latest['btc']:>12,.2f}")
        print(f"ETH close:      ${latest['eth']:>12,.2f}")
        print(f"ETH/BTC ratio:  {cur_ratio:>13.6f}   "
              f"(mean {target_ratio:.6f})")
        print(f"Z-score ({ROLLING_WINDOW:>2}):  {latest['z']:+13.3f}   "
              f"(entry +/-{ENTRY_Z}, exit +/-{EXIT_Z}, stop +/-{STOP_Z})")
        print(f"Position:       {latest['pos']:+13.0f}   "
              f"(+1 = long ETH/short BTC, -1 = opposite, 0 = flat)")
        print(f"Mean reverts:   {pct_to_mean:+12.2f}%   "
              f"(ratio move from now back to z=0)")
        print(f"Signal:         {snap['signal']}")
        print("-" * 64)
        print(f"In-sample backtest on last {len(df.dropna())} {kline} bars "
              f"(NOT the audited number above):")
        print(f"  ({kline}): "
              f"Sharpe {stats['sharpe']:.2f} | CAGR {stats['cagr']*100:.1f}% "
              f"| MaxDD {stats['max_dd']*100:.1f}% | "
              f"trades {stats['n_trades']} | "
              f"hit-rate {stats['hit_rate']*100:.1f}% "
              f"(in-trade) | "
              f"time-in-trade {stats['pct_time_in_trade']*100:.1f}%")
        print("=" * 64)

    trades = extract_trades(df)

    if verbose and len(trades):
        recent = trades.tail(5)
        print("Recent trades (most recent last):")
        print(f"  {'entry':19s}  {'exit':19s}  side  bars  pct_pnl   status")
        for _, t in recent.iterrows():
            entry = str(t["entry_time"])[:19]
            exit_ = str(t["exit_time"])[:19] if t["exit_time"] is not None else "(open)"
            side  = "L_ETH" if t["position"] == 1 else "S_ETH"
            print(f"  {entry:19s}  {exit_:19s}  {side:5s} "
                  f"{int(t['bars_held']):4d}  "
                  f"{t['pct_pnl']:+7.2f}%  {t['status']}")
        print("=" * 64)

    if write_artifacts:
        df.to_csv(f"{OUT_DIR}/eth_btc_pairs.csv")
        trades.to_csv(f"{OUT_DIR}/eth_btc_trades.csv", index=False)
        with open(f"{OUT_DIR}/eth_btc_latest_signal.json", "w") as f:
            json.dump(snap, f, indent=2, default=str)

        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        axes[0].plot(df.index, df["ratio"], lw=1.2)
        axes[0].set_title("ETH / BTC price ratio")
        axes[0].grid(alpha=0.3)
        axes[1].plot(df.index, df["z"], lw=1.0, color="#444")
        axes[1].axhline( ENTRY_Z, color="red",   ls="--", alpha=0.7,
                        label=f"+entry ({ENTRY_Z})")
        axes[1].axhline(-ENTRY_Z, color="green", ls="--", alpha=0.7,
                        label=f"-entry (-{ENTRY_Z})")
        axes[1].axhline( STOP_Z,  color="red",   ls=":",  alpha=0.5,
                        label=f"stop ({STOP_Z})")
        axes[1].axhline(-STOP_Z,  color="green", ls=":",  alpha=0.5)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_title(f"Rolling {ROLLING_WINDOW}-bar z-score of log(ETH/BTC)")
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[2].plot(df.index, df["equity"], lw=1.2, color="#1a6")
        axes[2].set_title("Strategy equity curve (dollar-neutral, net of fees)")
        axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/eth_btc_pairs.png", dpi=110)
        plt.close(fig)

    return snap


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _flip_alert_and_save(snap: dict, last_pos: int | None,
                         email_cfg: dict | None,
                         tg_cfg: dict | None) -> int:
    """Compare snap position to last_pos, alert on flip, persist new state.
    Returns the new position (an int, never None)."""
    pos_now = snap["position"]
    flipped = last_pos is not None and pos_now != last_pos
    if flipped:
        print("\n" + "!" * 58)
        print(f"!!  POSITION FLIP: {last_pos:+d}  ->  {pos_now:+d}")
        print(f"!!  {snap['signal']}")
        print("!" * 58 + "\n")
        subj, body = _format_alert(snap, last_pos)
        if email_cfg:
            try:
                send_email_alert(subj, body, email_cfg)
                print(f"[email] sent to {len(email_cfg['recipients'])} recipient(s)")
            except Exception as ex:
                print(f"[email] FAILED: {type(ex).__name__}: {ex}")
        if tg_cfg:
            try:
                send_telegram_alert(subj, body, tg_cfg)
                print(f"[telegram] sent to chat {tg_cfg['chat_id']}")
            except Exception as ex:
                print(f"[telegram] FAILED: {type(ex).__name__}: {ex}")
    _save_state({
        "position": pos_now,
        "asof":     snap["asof"],
        "zscore":   snap["zscore"],
        "updated":  datetime.now(timezone.utc).isoformat(),
    })
    return pos_now


def run_once_with_alerts(kline: str, source: str = "binance",
                         cmc_api_key: str | None = None,
                         alert_email: bool = False,
                         alert_telegram: bool = False,
                         mode: str = MODE,
                         use_ml: bool = False) -> int:
    """One-shot run for CI / cron: fetch, compute, save state, alert on flip,
    exit. Returns 0 on success, non-zero on misconfiguration. Always tries to
    succeed if data fetch fails (logs and exits 0) so the cron schedule
    doesn't go red on a transient outage."""
    email_cfg = _load_alert_config()    if alert_email    else None
    tg_cfg    = _load_telegram_config() if alert_telegram else None
    if alert_email and not email_cfg:
        print("ERROR: --alert-email set but SMTP_* / ALERT_TO env vars are "
              "not all set.", file=sys.stderr)
        return 2
    if alert_telegram and (not tg_cfg or not tg_cfg.get("chat_id")):
        print("ERROR: --alert-telegram needs both TELEGRAM_BOT_TOKEN and "
              "TELEGRAM_CHAT_ID env vars.", file=sys.stderr)
        return 2

    state = _load_state()
    last_pos = state.get("position")
    if last_pos is not None:
        print(f"Loaded prior state: last position = {last_pos:+d}")

    try:
        snap = run_once(kline=kline, source=source, cmc_api_key=cmc_api_key,
                        verbose=True, write_artifacts=True, mode=mode,
                        use_ml=use_ml)
    except Exception as e:
        print(f"[WARN] fetch failed: {type(e).__name__}: {e}. Skipping.")
        return 0  # transient failure -> still exit cleanly so cron stays green

    _flip_alert_and_save(snap, last_pos, email_cfg, tg_cfg)
    return 0


def run_loop(interval_minutes: int, kline: str,
             source: str = "binance",
             cmc_api_key: str | None = None,
             alert_email: bool = False,
             alert_telegram: bool = False,
             mode: str = MODE,
             use_ml: bool = False) -> int:
    """Poll every N minutes. Prints heartbeat; shouts when position flips."""
    stop = {"flag": False}
    def _handler(signum, frame):
        print("\nCaught signal, shutting down cleanly...")
        stop["flag"] = True
    _signal.signal(_signal.SIGINT,  _handler)
    _signal.signal(_signal.SIGTERM, _handler)

    email_cfg = _load_alert_config()    if alert_email    else None
    tg_cfg    = _load_telegram_config() if alert_telegram else None
    if alert_email and not email_cfg:
        print("ERROR: --alert-email set but SMTP_HOST/SMTP_USER/SMTP_PASS/"
              "ALERT_TO env vars are not all set. See --help for details.",
              file=sys.stderr)
        return 2
    if alert_telegram and not tg_cfg:
        print("ERROR: --alert-telegram set but TELEGRAM_BOT_TOKEN env var "
              "is not set.", file=sys.stderr)
        return 2
    if alert_telegram and tg_cfg and not tg_cfg.get("chat_id"):
        print("ERROR: TELEGRAM_CHAT_ID not set. After messaging your bot, "
              "run: python eth_btc_pairs.py --telegram-find-chat-id",
              file=sys.stderr)
        return 2

    state = _load_state()
    last_pos = state.get("position")
    print(f"Starting poller: kline={kline}, every {interval_minutes} min. "
          f"Ctrl-C to stop.")
    if email_cfg:
        print(f"Email alerts:    ENABLED -> {', '.join(email_cfg['recipients'])}")
    if tg_cfg:
        print(f"Telegram alerts: ENABLED -> chat_id {tg_cfg['chat_id']}")
    if last_pos is not None:
        print(f"Resuming from saved state: last position = {last_pos:+d}")

    while not stop["flag"]:
        try:
            snap = run_once(kline=kline, source=source,
                            cmc_api_key=cmc_api_key,
                            verbose=True, write_artifacts=True,
                            mode=mode, use_ml=use_ml)
        except Exception as e:
            print(f"[WARN] fetch failed: {type(e).__name__}: {e}. "
                  f"Will retry in {interval_minutes} min.")
            snap = None

        if snap is not None:
            last_pos = _flip_alert_and_save(snap, last_pos, email_cfg, tg_cfg)

        # sleep in small chunks so Ctrl-C responds quickly
        slept = 0.0
        total = interval_minutes * 60.0
        while slept < total and not stop["flag"]:
            time.sleep(min(1.0, total - slept))
            slept += 1.0

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ETH/BTC pairs-trading signal")
    p.add_argument("--loop", action="store_true",
                   help="poll continuously instead of printing once and exiting")
    p.add_argument("--interval-minutes", type=int, default=15,
                   help="polling interval in minutes (default 15). Ignored without --loop.")
    p.add_argument("--kline", default=INTERVAL,
                   help=f"Binance kline interval: 1d/4h/1h/15m/5m (default {INTERVAL})")
    p.add_argument("--mode", choices=["auto", "momentum", "mean-revert"],
                   default=MODE,
                   help=f"signal direction. 'auto' (default) uses the v2 "
                        f"regime detector (Hurst + rolling ADF) to pick "
                        f"momentum or mean-revert per entry. 'momentum' / "
                        f"'mean-revert' override. The walk-forward audit on "
                        f"32 months of 4h data shows 100%% momentum-regime "
                        f"so 'auto' currently behaves as 'momentum'; the "
                        f"detector is shipped for future regime changes.")
    p.add_argument("--use-ml", action="store_true",
                   help="OPT-IN: gate fresh entries with the trained "
                        "HistGradientBoosting classifier saved at "
                        "analysis/model.pkl. Walk-forward AUC on the audited "
                        "sample is 0.57-0.60; not enough to dominate the bare "
                        "momentum signal in our backtests, so it is OFF by "
                        "default. Useful as a 'second opinion' veto if you "
                        "want fewer / higher-conviction trades.")
    p.add_argument("--source", choices=["binance", "cmc"], default="binance",
                   help="data source: Binance historical klines (default) or "
                        "CoinMarketCap latest quotes accumulating locally")
    p.add_argument("--cmc-key", default=os.environ.get("CMC_API_KEY"),
                   help="CoinMarketCap API key (or set CMC_API_KEY env var)")
    p.add_argument("--alert-email", action="store_true",
                   help="email alerts on position flips. Requires SMTP_HOST, "
                        "SMTP_USER, SMTP_PASS, ALERT_TO env vars. Optional: "
                        "SMTP_PORT (587), SMTP_SSL (0), ALERT_FROM. ALERT_TO "
                        "accepts comma-separated emails or email-to-SMS "
                        "gateway addresses.")
    p.add_argument("--alert-telegram", action="store_true",
                   help="Telegram alerts on position flips. Requires "
                        "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars. "
                        "Find your chat id with --telegram-find-chat-id.")
    p.add_argument("--telegram-find-chat-id", action="store_true",
                   help="print chat IDs that have messaged your bot "
                        "(via getUpdates). Requires TELEGRAM_BOT_TOKEN env var. "
                        "Message your bot first, then run this.")
    p.add_argument("--once-alert", action="store_true",
                   help="run a single poll, save state, alert on flip, then "
                        "exit. Designed for cron / GitHub Actions schedules. "
                        "Mutually exclusive with --loop.")
    p.add_argument("--test-alert", action="store_true",
                   help="send a test through every configured channel "
                        "(email if SMTP_* env vars set, Telegram if "
                        "TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID set). Use this "
                        "to verify credentials before running --loop.")
    args = p.parse_args(argv)

    if args.telegram_find_chat_id:
        tok = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not tok:
            print("ERROR: TELEGRAM_BOT_TOKEN env var not set.", file=sys.stderr)
            return 2
        try:
            chats = telegram_find_chat_id(tok)
        except Exception as ex:
            print(f"FAILED: {type(ex).__name__}: {ex}", file=sys.stderr)
            return 1
        if not chats:
            print("No chats found. Open t.me/<your_bot_username>, tap Start "
                  "(or send /start), then run this again.")
            return 1
        print("Chats that have messaged your bot:")
        for cid, label in chats.items():
            print(f"  chat_id = {cid}   ({label})")
        print("\nExport the chat id you want alerts on, e.g.:")
        for cid in chats:
            print(f"  export TELEGRAM_CHAT_ID={cid}")
            break
        return 0

    if args.test_alert:
        email_cfg = _load_alert_config()
        tg_cfg    = _load_telegram_config()
        if not email_cfg and not tg_cfg:
            print("ERROR: no alert channels configured. Set SMTP_* env vars "
                  "and/or TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID. See --help.",
                  file=sys.stderr)
            return 2
        subj = "ETH/BTC pairs: test alert"
        body = ("This is a test from eth_btc_pairs.py. If you can read this, "
                "alerting is configured correctly and you will receive live "
                "signal flips from --loop.\n")
        ok_any = False
        if email_cfg:
            try:
                send_email_alert(subj, body, email_cfg)
                print(f"[email]    sent to {', '.join(email_cfg['recipients'])}")
                ok_any = True
            except Exception as ex:
                print(f"[email]    FAILED: {type(ex).__name__}: {ex}",
                      file=sys.stderr)
        if tg_cfg:
            if not tg_cfg.get("chat_id"):
                print("[telegram] SKIPPED: TELEGRAM_CHAT_ID not set. "
                      "Run --telegram-find-chat-id first.", file=sys.stderr)
            else:
                try:
                    send_telegram_alert(subj, body, tg_cfg)
                    print(f"[telegram] sent to chat {tg_cfg['chat_id']}")
                    ok_any = True
                except Exception as ex:
                    print(f"[telegram] FAILED: {type(ex).__name__}: {ex}",
                          file=sys.stderr)
        return 0 if ok_any else 1

    if args.source == "cmc" and not args.cmc_key:
        print("ERROR: --source cmc requires CMC_API_KEY env var or --cmc-key",
              file=sys.stderr)
        return 2

    if args.loop and args.once_alert:
        print("ERROR: --once-alert and --loop are mutually exclusive.",
              file=sys.stderr)
        return 2

    if args.once_alert:
        return run_once_with_alerts(
            args.kline, source=args.source, cmc_api_key=args.cmc_key,
            alert_email=args.alert_email, alert_telegram=args.alert_telegram,
            mode=args.mode, use_ml=args.use_ml)

    if args.loop:
        return run_loop(args.interval_minutes, args.kline,
                        source=args.source, cmc_api_key=args.cmc_key,
                        alert_email=args.alert_email,
                        alert_telegram=args.alert_telegram,
                        mode=args.mode, use_ml=args.use_ml)
    run_once(kline=args.kline, source=args.source,
             cmc_api_key=args.cmc_key,
             verbose=True, write_artifacts=True,
             mode=args.mode, use_ml=args.use_ml)
    print(f"\nWrote: eth_btc_pairs.csv, eth_btc_pairs.png, "
          f"eth_btc_trades.csv, eth_btc_latest_signal.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
