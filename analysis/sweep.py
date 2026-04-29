"""
Parameter sweep for the ETH/BTC pairs strategy.

The walk-forward test in research.py showed every 1h/30-bar variant losing
money in true OOS. The classical pairs-trading literature (Gatev-Goetzmann-
Rouwenhorst 2006, Avellaneda-Lee 2010, Vidyamurthy 2004, Krauss 2017) uses
daily bars and 60-250 bar windows, not hourly bars and 30-bar windows. This
script sweeps timeframe x window x entry_z combinations and reports the top
configurations by walk-forward Sharpe.

Trade-cost note: the current 4bps/leg with 4 legs round-trip is ~16bps per
round trip. With 600+ trades on 1h bars over 14 months, fees alone consume
~100% of capital. Reducing trade count is critical.

Run:
    python analysis/sweep.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR  = Path(__file__).parent
FEE_BPS  = 4.0
EXIT_Z   = 0.3
STOP_Z   = 3.5

BINANCE_BASES = ["https://api.binance.com", "https://api.binance.us"]
KLINE_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


# ---------- pagination-aware fetch (cached on disk) ----------

def _fetch_chunk(symbol: str, interval: str, end_ms: int,
                 limit: int = 1000) -> pd.DataFrame:
    last = None
    for base in list(BINANCE_BASES):
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": symbol, "interval": interval,
                        "limit": limit, "endTime": end_ms},
                timeout=30,
            )
            r.raise_for_status()
        except requests.HTTPError as e:
            sc = e.response.status_code if e.response is not None else None
            if sc in (451, 403, 418):
                last = e
                continue
            raise
        except requests.RequestException as e:
            last = e
            continue
        if BINANCE_BASES[0] != base:
            BINANCE_BASES.remove(base)
            BINANCE_BASES.insert(0, base)
        rows = r.json()
        if not rows:
            return pd.DataFrame()
        cols = ["openTime", "open", "high", "low", "close", "volume",
                "closeTime", "qav", "trades", "tbav", "tbqv", "ignore"]
        df = pd.DataFrame(rows, columns=cols)
        df["time"]  = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
        df["close"] = df["close"].astype(float)
        return df.set_index("time")[["close"]]
    raise last or RuntimeError("all Binance hosts failed")


def fetch(symbol: str, interval: str, target_bars: int) -> pd.DataFrame:
    """Cached on disk so the sweep doesn't re-fetch from Binance."""
    cache = OUT_DIR / f"_cache_{symbol}_{interval}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df) >= target_bars:
            return df.tail(target_bars)
    bar_ms = KLINE_MS[interval]
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    frames = []
    pulled = 0
    while pulled < target_bars:
        chunk = _fetch_chunk(symbol, interval, end_ms, limit=1000)
        if chunk.empty:
            break
        frames.append(chunk)
        pulled += len(chunk)
        oldest = int(chunk.index.min().timestamp() * 1000)
        end_ms = oldest - bar_ms
        time.sleep(0.15)
        if len(chunk) < 1000:
            break
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.columns = [symbol]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df.tail(target_bars)


def load_pair(interval: str, target_bars: int) -> pd.DataFrame:
    btc = fetch("BTCUSDT", interval, target_bars)
    eth = fetch("ETHUSDT", interval, target_bars)
    df = pd.concat([btc, eth], axis=1).dropna()
    df = df.rename(columns={"BTCUSDT": "btc", "ETHUSDT": "eth"})
    df["log_r"]   = np.log(df["eth"] / df["btc"])
    df["ret_r"]   = df["log_r"].diff()
    return df.dropna()


# ---------- walk-forward backtest ----------

def state_machine_positions(z: pd.Series, entry: float, exit_: float,
                            stop: float, momentum: bool = False) -> pd.Series:
    """Mean-reversion (default) or momentum (with sign flipped) state machine.

    Mean-reversion: z above +entry -> short ratio (bet on revert down).
    Momentum:       z above +entry -> long ratio  (bet on continuation up).
    """
    pos = np.zeros(len(z))
    cur = 0
    sign = -1 if not momentum else +1
    for i, zt in enumerate(z.values):
        if np.isnan(zt):
            pos[i] = 0
            continue
        if cur == 0:
            if zt >  entry:
                cur = sign
            elif zt < -entry:
                cur = -sign
        # exits: same |z| < exit_ band exits, |z| > stop is hard stop.
        elif cur == +1 and ((sign == -1 and zt > -exit_) or (sign == +1 and zt < exit_) or abs(zt) > stop):
            cur = 0
        elif cur == -1 and ((sign == -1 and zt <  exit_) or (sign == +1 and zt > -exit_) or abs(zt) > stop):
            cur = 0
        pos[i] = cur
    return pd.Series(pos, index=z.index, name="position").astype(float)


@dataclass
class SweepRow:
    kline: str
    window: int
    entry_z: float
    direction: str          # "mean-revert" or "momentum"
    sharpe: float
    cagr: float
    max_dd: float
    n_trades: int
    pct_time_in_trade: float
    total_return_pct: float
    n_bars: int
    walk_train_bars: int
    walk_test_bars: int


def evaluate(df: pd.DataFrame, window: int, entry_z: float,
             walk_train_bars: int, walk_test_bars: int,
             ppy: int, momentum: bool = False) -> dict:
    """Walk-forward: at each test bar use only train-window mu/sd."""
    log_r = df["log_r"]
    z = pd.Series(np.nan, index=df.index)
    n = len(df)
    start = max(walk_train_bars, window + 5)
    if start + walk_test_bars > n:
        return {"sharpe": float("nan"), "cagr": float("nan"),
                "max_dd": float("nan"), "n_trades": 0,
                "pct_time_in_trade": 0.0, "total_return_pct": 0.0}
    while start + walk_test_bars <= n:
        # Fit z-score parameters on train window only.
        train_log = log_r.iloc[start - walk_train_bars:start]
        # But use a rolling `window`-bar mean/std at each test bar so the
        # z-score adapts within the test window — fit-once parameters lag
        # too much when volatility regime shifts. The "OOS" guarantee comes
        # from the fact that at bar t we use mu/sd on (t-window, t-1).
        test_idx = df.index[start:start + walk_test_bars]
        for t in test_idx:
            i = df.index.get_loc(t)
            if i < window:
                continue
            past = log_r.iloc[i - window:i]
            mu, sd = past.mean(), past.std()
            if sd > 0:
                z.loc[t] = (log_r.loc[t] - mu) / sd
        start += walk_test_bars

    pos = state_machine_positions(z, entry_z, EXIT_Z, STOP_Z, momentum=momentum)
    pos_lag = pos.shift(1).fillna(0.0)
    gross = pos_lag * df["ret_r"]
    turnover = pos.diff().abs().fillna(0.0)
    fees = turnover * (FEE_BPS / 1e4) * 2.0
    net = (gross - fees).fillna(0.0)
    eq = np.exp(net.cumsum())

    r = net.dropna()
    if r.std() == 0 or len(r) < 2:
        return {"sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0, "n_trades": 0,
                "pct_time_in_trade": 0.0, "total_return_pct": 0.0}

    mu  = r.mean() * ppy
    sig = r.std() * math.sqrt(ppy)
    sharpe = float(mu / sig if sig else 0.0)
    cagr   = float(np.exp(mu) - 1)
    peak   = eq.cummax()
    max_dd = float((eq / peak - 1).min())

    p = pos.fillna(0).astype(int).values
    nt, prev = 0, 0
    for v in p:
        if v != 0 and prev == 0:
            nt += 1
        prev = v
    return {
        "sharpe":            sharpe,
        "cagr":              cagr,
        "max_dd":            max_dd,
        "n_trades":          nt,
        "pct_time_in_trade": float((pos_lag.abs() > 0).mean()),
        "total_return_pct":  float((eq.iloc[-1] - 1.0) * 100.0),
    }


def main() -> int:
    # Sweep grid driven by literature.
    # Daily bars: classic pairs-trading window is 60-250 days.
    # 4h bars: window of 60-180 4h bars = 10-30 days.
    # 1h bars: try wider thresholds and longer windows than the original 30/2.0.
    grid = []
    grid += [("1d",  w, ez) for w in [60, 90, 120, 180, 250]
                              for ez in [1.5, 2.0, 2.5, 3.0]]
    grid += [("4h",  w, ez) for w in [60, 120, 180, 360]
                              for ez in [1.5, 2.0, 2.5, 3.0]]
    grid += [("1h",  w, ez) for w in [120, 240, 480, 720]
                              for ez in [2.0, 2.5, 3.0]]

    target_bars_for = {"1d": 1500, "4h": 6000, "1h": 13_000}
    walk_for = {
        "1d": (250, 60),    # 250d train, 60d test
        "4h": (24*30*3, 24*30),  # ~3 months train (in 4h), 1 month test
        "1h": (24*90, 24*30),
    }
    ppy = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24}

    cached: dict[str, pd.DataFrame] = {}
    rows: list[SweepRow] = []

    for kline, window, entry_z in grid:
        if kline not in cached:
            cached[kline] = load_pair(kline, target_bars_for[kline])
        df = cached[kline]
        train_bars, test_bars = walk_for[kline]
        for momentum, label in [(False, "mean-revert"), (True, "momentum")]:
            m = evaluate(df, window=window, entry_z=entry_z,
                         walk_train_bars=train_bars, walk_test_bars=test_bars,
                         ppy=ppy[kline], momentum=momentum)
            rows.append(SweepRow(
                kline=kline, window=window, entry_z=entry_z, direction=label,
                n_bars=len(df),
                walk_train_bars=train_bars, walk_test_bars=test_bars,
                **m,
            ))
            print(f"  {kline:>3s}  win={window:>3d}  ez={entry_z:.1f}  "
                  f"{label:11s}  "
                  f"Sharpe={m['sharpe']:+5.2f}  Ret={m['total_return_pct']:+7.2f}%  "
                  f"trades={m['n_trades']:>3d}  DD={m['max_dd']*100:+5.1f}%")

    rows.sort(key=lambda r: r.sharpe, reverse=True)

    print("\n=== TOP 10 by walk-forward Sharpe (across BOTH directions) ===")
    print(f"  {'kline':>5s}  {'window':>6s}  {'entry_z':>7s}  {'direction':>11s}  "
          f"{'Sharpe':>6s}  {'CAGR%':>6s}  {'MaxDD%':>7s}  "
          f"{'trades':>6s}  {'TotRet%':>8s}")
    for r in rows[:10]:
        print(f"  {r.kline:>5s}  {r.window:>6d}  {r.entry_z:>7.1f}  "
              f"{r.direction:>11s}  "
              f"{r.sharpe:>+6.2f}  {r.cagr*100:>+6.1f}  {r.max_dd*100:>+7.1f}  "
              f"{r.n_trades:>6d}  {r.total_return_pct:>+8.2f}")

    print("\n=== BOTTOM 5 (sanity) ===")
    for r in rows[-5:]:
        print(f"  {r.kline:>5s}  {r.window:>6d}  {r.entry_z:>7.1f}  "
              f"{r.direction:>11s}  "
              f"{r.sharpe:>+6.2f}  Ret={r.total_return_pct:>+7.2f}%")

    out = {
        "sweep": [vars(r) for r in rows],
        "top1":  vars(rows[0]),
    }
    (OUT_DIR / "sweep_results.json").write_text(json.dumps(out, indent=2,
                                                          default=str))
    print(f"\nWrote {OUT_DIR/'sweep_results.json'}")
    print(f"\nBest config: kline={rows[0].kline}  window={rows[0].window}  "
          f"entry_z={rows[0].entry_z}  direction={rows[0].direction}  "
          f"Sharpe={rows[0].sharpe:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
