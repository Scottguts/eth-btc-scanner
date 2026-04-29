"""
v2 mega-backtest: regime-aware direction + z-signal + ML overlay + vol-target.

Pipeline per bar
----------------

1. Compute log-spread features (z, momentum, vol) on a rolling 360-bar 4h window.
2. Compute regime label from Hurst + rolling ADF (analysis/regime.py) with
   hysteresis to prevent flapping.
3. State-machine on z to generate base entries/exits, with the entry direction
   determined by regime:
        regime == "momentum"     -> long ratio when z > +entry_z, short when <
        regime == "mean-revert"  -> short ratio when z > +entry_z, long when <
        regime == "indeterminate" -> stay flat (no edge claimed)
4. ML overlay (HistGradientBoostingClassifier) gates entries: only take a fresh
   trade if the model's probability that the trade will be profitable 24 bars
   forward is >= threshold. Trained walk-forward on a 270-day window, retested
   on the next 30 days, retrained at the end of each fold.
5. Vol-target position sizing: scale per-leg notional so that the expected
   1-day vol of the spread is `target_daily_vol`. Equivalent to dividing
   raw position {-1, 0, +1} by realized 24-bar vol of the log-spread.

Outputs
-------
    analysis/v2_metrics.json      machine-readable summary
    analysis/v2_report.txt        human-readable summary
    analysis/v2_curves.png        equity curves vs v1 baseline
    analysis/model.pkl            trained classifier the live bot uses
    analysis/model_meta.json      training-window stats for the live bot
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from regime import detect_regime, RegimeBands
from features import build_features

# ---------- config ----------
KLINE              = "4h"
TARGET_BARS        = 6_000              # ~3 years of 4h closes
ROLLING_WINDOW     = 360                # z-score window (60 days)
ENTRY_Z, EXIT_Z, STOP_Z = 2.5, 0.3, 3.5
FEE_BPS            = 4.0
WALK_TRAIN_BARS    = 6 * 30 * 3         # 3 months train (4h)   = 540
WALK_TEST_BARS     = 6 * 30             # 1 month test (4h)     = 180
ML_THRESHOLD       = 0.50               # below: veto entry (hard gate variant)
ML_SIZE_FLOOR      = 0.30                # baseline size when proba == 0.5
ML_SIZE_GAIN       = 4.0                 # extra size per +0.10 above 0.5
TARGET_DAILY_VOL   = 0.015               # 1.5% daily vol target on the spread
RANDOM_STATE       = 7
PERIODS_PER_YEAR   = 6 * 365
HURST_WINDOW       = 240
ADF_WINDOW         = 240
ADF_STEP           = 4
OUT_DIR            = Path(__file__).parent
# ----------------------------


# ---------- data fetch (uses sweep cache if present) ----------

BINANCE_BASES = ["https://api.binance.com", "https://api.binance.us"]
KLINE_MS      = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _fetch_chunk(symbol: str, interval: str, end_ms: int,
                 limit: int = 1000) -> pd.DataFrame:
    last = None
    for base in list(BINANCE_BASES):
        try:
            r = requests.get(f"{base}/api/v3/klines",
                params={"symbol": symbol, "interval": interval,
                        "limit": limit, "endTime": end_ms}, timeout=30)
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
        df["time"]   = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df.set_index("time")[["close", "volume"]]
    raise last or RuntimeError("all Binance hosts failed")


def fetch(symbol: str, interval: str, target_bars: int) -> pd.DataFrame:
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
    if "close" in df:
        # Single-symbol from cache; rename for the joined df.
        df.columns = [f"{symbol}_{c}" for c in df.columns]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df.tail(target_bars)


def load_pair(interval: str, target_bars: int) -> pd.DataFrame:
    btc = fetch("BTCUSDT", interval, target_bars)
    eth = fetch("ETHUSDT", interval, target_bars)
    # Cache compatibility with sweep.py which stored single-column ('BTCUSDT').
    if "BTCUSDT" in btc.columns:
        btc = btc.rename(columns={"BTCUSDT": "BTCUSDT_close"})
        btc["BTCUSDT_volume"] = np.nan
    if "ETHUSDT" in eth.columns:
        eth = eth.rename(columns={"ETHUSDT": "ETHUSDT_close"})
        eth["ETHUSDT_volume"] = np.nan
    df = pd.concat([btc, eth], axis=1).dropna(subset=["BTCUSDT_close",
                                                       "ETHUSDT_close"])
    df = df.rename(columns={
        "BTCUSDT_close":  "btc",  "BTCUSDT_volume": "btc_vol",
        "ETHUSDT_close":  "eth",  "ETHUSDT_volume": "eth_vol",
    })
    df["log_eth"] = np.log(df["eth"])
    df["log_btc"] = np.log(df["btc"])
    df["log_r"]   = np.log(df["eth"] / df["btc"])
    df["ret_eth"] = df["log_eth"].diff()
    df["ret_btc"] = df["log_btc"].diff()
    df["ret_r"]   = df["log_r"].diff()
    # Drop rows where the price/return columns are NaN, but allow volume to
    # be NaN (the sweep.py cache only saved closes; we won't refuse to run
    # just because volume features will be missing).
    return df.dropna(subset=["eth", "btc", "log_r", "ret_r",
                              "ret_eth", "ret_btc"])


# ---------- core machinery ----------

def zscore_oos(log_r: pd.Series, train_bars: int, test_bars: int,
               window: int) -> pd.Series:
    """Walk-forward z-score: at each test bar use a `window`-bar mean/std on
    bars that strictly precede it. Equivalent to the live bot's behavior."""
    z = pd.Series(np.nan, index=log_r.index)
    n = len(log_r)
    start = max(train_bars, window + 1)
    while start + test_bars <= n:
        for i in range(start, start + test_bars):
            past = log_r.iloc[i - window:i]
            mu, sd = past.mean(), past.std()
            if sd > 0:
                z.iloc[i] = (log_r.iloc[i] - mu) / sd
        start += test_bars
    return z


def regime_aware_positions(z: pd.Series, regime: pd.Series,
                           entry: float = ENTRY_Z, exit_: float = EXIT_Z,
                           stop:  float = STOP_Z) -> pd.Series:
    """State machine that maps z + regime label -> position. Indeterminate
    regime forces flat. Direction flips with regime — but only when the
    bot is currently flat (a regime change does NOT yank a live trade)."""
    pos = np.zeros(len(z))
    cur = 0
    cur_regime = "indeterminate"
    z_vals = z.values
    r_vals = regime.values
    for i in range(len(z)):
        zt, reg = z_vals[i], r_vals[i]
        if cur == 0:
            cur_regime = reg
        if not np.isfinite(zt):
            pos[i] = cur
            continue
        if cur == 0:
            if cur_regime == "indeterminate":
                pos[i] = 0
                continue
            sign = +1 if cur_regime == "momentum" else -1
            if zt >  entry:
                cur = +sign
            elif zt < -entry:
                cur = -sign
        elif cur != 0 and (abs(zt) < exit_ or abs(zt) > stop):
            cur = 0
        pos[i] = cur
    return pd.Series(pos, index=z.index, name="position").astype(float)


def vol_target_size(positions: pd.Series, ret_r: pd.Series,
                    target_daily_vol: float = TARGET_DAILY_VOL,
                    vol_window: int = 24, max_size: float = 5.0
                    ) -> pd.Series:
    """Scale {-1, 0, +1} positions so the *expected* daily vol of the spread
    return matches `target_daily_vol`. Crucially: size is set ONCE at entry
    and held for the duration of the trade, NOT rebalanced bar-by-bar — that
    avoids spurious turnover and the fee drag we observed in the first
    iteration (Sharpe dropped from 1.04 -> 0.46 when size was floating).
    """
    bar_vol = ret_r.rolling(vol_window).std()
    daily_vol = (bar_vol * np.sqrt(6)).replace(0, np.nan).bfill().ffill()
    raw_size = (target_daily_vol / daily_vol).clip(upper=max_size).shift(1)

    sized = pd.Series(0.0, index=positions.index)
    held_size = 1.0
    pos_vals  = positions.values
    sz_vals   = raw_size.fillna(1.0).values
    last_pos  = 0.0
    for i in range(len(positions)):
        cur = pos_vals[i]
        if cur != 0 and last_pos == 0:                # entry: lock size now
            held_size = float(sz_vals[i])
        if cur == 0:                                  # flat between trades
            held_size = 1.0
        sized.iloc[i] = cur * held_size
        last_pos = cur
    return sized.rename("sized_pos")


@dataclass
class BTResult:
    name: str
    sharpe: float
    cagr: float
    max_dd: float
    n_trades: int
    pct_time_in_trade: float
    total_return_pct: float
    equity: pd.Series


def run_backtest(df: pd.DataFrame, pos: pd.Series, name: str,
                 fee_bps: float = FEE_BPS) -> BTResult:
    pos_lag = pos.shift(1).fillna(0.0)
    gross   = pos_lag * df["ret_r"]
    turnover = pos.diff().abs().fillna(0.0)
    fees    = turnover * (fee_bps / 1e4) * 2.0
    net     = (gross - fees).fillna(0.0)
    eq      = np.exp(net.cumsum())
    r = net.dropna()
    if r.std() == 0 or len(r) < 2:
        return BTResult(name, 0., 0., 0., 0, 0., 0., eq)
    mu  = r.mean() * PERIODS_PER_YEAR
    sig = r.std() * math.sqrt(PERIODS_PER_YEAR)
    sharpe = float(mu / sig if sig else 0.0)
    cagr   = float(np.exp(mu) - 1)
    peak   = eq.cummax()
    max_dd = float((eq / peak - 1).min())

    sign_pos = np.sign(pos.fillna(0).values)
    nt, prev = 0, 0
    for v in sign_pos:
        if v != 0 and prev == 0:
            nt += 1
        prev = v
    return BTResult(
        name=name, sharpe=sharpe, cagr=cagr, max_dd=max_dd,
        n_trades=nt,
        pct_time_in_trade=float((pos_lag.abs() > 0).mean()),
        total_return_pct=float((eq.iloc[-1] - 1.0) * 100.0),
        equity=eq,
    )


# ---------- ML overlay (walk-forward, save final fold's model) ----------

def label_forward(df: pd.DataFrame, raw_pos: pd.Series,
                  horizon: int = 24) -> pd.Series:
    """Label = 1 if a position taken in `raw_pos` direction at this bar would
    be profitable `horizon` bars forward (gross of fees). Labelled on EVERY
    bar where raw_pos != 0 (continuous labels), not just entry bars — that
    gives the classifier ~10-20x more training data and a smoother target.

    The classifier still gates ENTRIES only at decision time; the broader
    training set just teaches it 'in similar market conditions, was the
    direction-X trade profitable N bars later?'.
    """
    fwd = df["log_r"].shift(-horizon) - df["log_r"]
    pnl = raw_pos * fwd
    y = pd.Series(np.nan, index=df.index)
    mask = (raw_pos != 0) & pnl.notna()
    y[mask] = (pnl[mask] > 0).astype(float)
    return y


def walk_forward_ml(df: pd.DataFrame, z: pd.Series, raw_pos: pd.Series,
                    feat: pd.DataFrame,
                    train_bars: int, test_bars: int,
                    threshold: float = ML_THRESHOLD,
                    size_floor: float = ML_SIZE_FLOOR,
                    size_gain:  float = ML_SIZE_GAIN
                    ) -> tuple[pd.Series, pd.Series, list[float],
                               HistGradientBoostingClassifier | None,
                               list[str], pd.Series]:
    """Walk-forward classifier training. Returns:
        gated_pos    - hard-gated positions (entry vetoed if proba < threshold)
        sized_pos    - probability-scaled positions (continuous in [0, max])
        oos_aucs     - OOS AUC per fold
        last_model   - the most recent fold's fitted classifier (saved to pickle
                       for the live bot)
        feature_cols - column ordering the live bot must replicate at inference
        proba_series - the OOS probabilities, useful for diagnostics
    """
    labels = label_forward(df, raw_pos, horizon=24)
    gated  = raw_pos.copy()
    sized  = raw_pos.copy().astype(float)
    proba_series = pd.Series(np.nan, index=df.index)
    aucs: list[float] = []
    last_model = None
    feat_cols = list(feat.columns)
    n = len(df)
    start = train_bars
    held_size = 0.0  # locked size during a live trade for `sized` variant
    last_pos  = 0
    # We walk the gating logic per-fold below; sized_pos is set in a second
    # pass after the full proba_series is built.
    while start + test_bars <= n:
        train_idx = df.index[start - train_bars:start]
        test_idx  = df.index[start:start + test_bars]

        Xtr = feat.loc[train_idx].dropna()
        ytr = labels.loc[Xtr.index].dropna()
        Xtr = Xtr.loc[ytr.index]
        if len(ytr) < 50 or ytr.nunique() < 2:
            start += test_bars
            continue

        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.04, max_iter=400,
            l2_regularization=1.0, min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )
        clf.fit(Xtr.values, ytr.values)
        last_model = clf

        Xte = feat.loc[test_idx].dropna()
        if Xte.empty:
            start += test_bars
            continue
        proba = clf.predict_proba(Xte.values)[:, 1]
        proba_s = pd.Series(proba, index=Xte.index)
        proba_series.loc[Xte.index] = proba_s

        # Hard-gate variant: veto an entry if proba < threshold.
        prev = raw_pos.shift(1).fillna(0).reindex(test_idx).fillna(0)
        cur  = raw_pos.reindex(test_idx).fillna(0)
        entry_mask = (prev == 0) & (cur != 0)
        for ts, is_entry in entry_mask.items():
            if not is_entry or ts not in proba_s.index:
                continue
            if proba_s.loc[ts] < threshold:
                gated.loc[ts] = 0.0
                j = df.index.get_loc(ts) + 1
                while j < len(df):
                    rp = raw_pos.iloc[j]
                    if rp == 0 or np.sign(rp) != np.sign(cur.loc[ts]):
                        break
                    gated.iloc[j] = 0.0
                    j += 1

        yte = labels.loc[Xte.index].dropna()
        if len(yte) >= 20 and yte.nunique() == 2:
            aucs.append(float(roc_auc_score(
                yte.values, proba_s.loc[yte.index].values)))

        start += test_bars

    # ---- second pass: continuous probability-scaled position size ----
    # size = max(size_floor, size_floor + size_gain * (proba - 0.5)) at entry,
    # then HELD constant for the duration of the trade. proba is shifted by 1
    # bar to preserve no-lookahead.
    proba_lag = proba_series.shift(1).fillna(0.5)
    sized_vals = np.zeros(len(raw_pos))
    last_pos = 0
    held_size = 0.0
    for i in range(len(raw_pos)):
        cur = raw_pos.iloc[i]
        if cur != 0 and last_pos == 0:
            p = float(proba_lag.iloc[i])
            held_size = max(0.0, size_floor + size_gain * (p - 0.5))
        if cur == 0:
            held_size = 0.0
        sized_vals[i] = cur * held_size
        last_pos = cur
    sized = pd.Series(sized_vals, index=raw_pos.index, name="sized_pos")

    return gated, sized, aucs, last_model, feat_cols, proba_series


# ---------- main ----------

def main() -> int:
    print(f"Loading {TARGET_BARS} {KLINE} bars from Binance (cached if seen)...")
    df = load_pair(KLINE, TARGET_BARS)
    print(f"Got {len(df)} rows {df.index[0]} -> {df.index[-1]}")

    print("Computing regime (Hurst + rolling ADF) ...")
    # Slightly looser bands than the defaults so the detector gets a fair
    # chance to flag mean-reverting episodes if any exist in the sample.
    bands = RegimeBands(
        h_trend=0.53, h_meanrev=0.47,
        adf_trend=0.15, adf_meanrev=0.08,
        hysteresis_bars=4, initial_regime="momentum",
    )
    rg = detect_regime(df["log_r"], hurst_window=HURST_WINDOW,
                       adf_window=ADF_WINDOW, adf_step=ADF_STEP, bands=bands)
    df = df.join(rg)
    counts = df["regime"].value_counts(dropna=False)
    print("  Regime distribution (bars):")
    for k, v in counts.items():
        print(f"    {str(k):>14s}  {int(v):>5d}  ({v/len(df)*100:.1f}%)")

    # Walk-forward z-score (true OOS).
    z = zscore_oos(df["log_r"], train_bars=WALK_TRAIN_BARS,
                   test_bars=WALK_TEST_BARS, window=ROLLING_WINDOW)
    df["z"] = z

    # ----- v1 baseline (momentum, single direction, dollar-weighted) -----
    def _v1_momentum_pos(z: pd.Series) -> pd.Series:
        """Same state machine as the live bot's MODE='momentum' branch."""
        pos = np.zeros(len(z))
        cur = 0
        for i, zt in enumerate(z.values):
            if not np.isfinite(zt):
                pos[i] = cur
                continue
            if cur == 0:
                if zt >  ENTRY_Z:
                    cur = +1
                elif zt < -ENTRY_Z:
                    cur = -1
            elif abs(zt) < EXIT_Z or abs(zt) > STOP_Z:
                cur = 0
            pos[i] = cur
        return pd.Series(pos, index=z.index, name="position").astype(float)
    base_mom = _v1_momentum_pos(z)
    res_v1   = run_backtest(df, base_mom, "v1: momentum / dollar / no ML")

    # ----- v2: regime-aware -----
    raw_pos_v2 = regime_aware_positions(z, df["regime"],
                                        ENTRY_Z, EXIT_Z, STOP_Z)
    res_v2_raw = run_backtest(df, raw_pos_v2,
                              "v2a: regime-aware / dollar / no ML")

    # ----- v2 + ML overlay -----
    print("Building ML features ...")
    feat = build_features(df, z=z, regime=df["regime"],
                          hurst=df["hurst"], adf_p=df["adf_p"])
    print(f"  features: {feat.shape[1]} columns")

    print("Training walk-forward ML overlay ...")
    gated_pos, ml_sized_pos, aucs, last_model, feat_cols, proba_oos = (
        walk_forward_ml(df, z, raw_pos_v2, feat,
                        train_bars=WALK_TRAIN_BARS, test_bars=WALK_TEST_BARS,
                        threshold=ML_THRESHOLD))

    res_v2_ml = run_backtest(df, gated_pos,
                             "v2b: regime / dollar / ML hard-gate")

    # v2c: ML probability sizes the position continuously (no veto)
    res_v2_size = run_backtest(df, ml_sized_pos,
                               "v2c: regime / ML proba-sized position")

    # v2d: full stack — ML proba-sized * vol-target (size locked at entry)
    sized_full = vol_target_size(ml_sized_pos, df["ret_r"],
                                 target_daily_vol=TARGET_DAILY_VOL)
    res_v2_full = run_backtest(df, sized_full,
                               "v2d: regime / ML / vol-target")

    print(f"\n  ML OOS AUC across {len(aucs)} folds: "
          f"mean={np.mean(aucs) if aucs else float('nan'):.3f} "
          f"median={np.median(aucs) if aucs else float('nan'):.3f}")

    results = [res_v1, res_v2_raw, res_v2_ml, res_v2_size, res_v2_full]

    print("\n=== v1 vs v2 walk-forward results ===")
    print(f"  {'variant':50s}  Sharpe  CAGR%  MaxDD%  Trades  Time%  TotRet%")
    for r in results:
        print(f"  {r.name:50s}  {r.sharpe:>+5.2f}  {r.cagr*100:>+5.1f}  "
              f"{r.max_dd*100:>+6.1f}  {r.n_trades:>5d}  "
              f"{r.pct_time_in_trade*100:>4.1f}  {r.total_return_pct:>+7.2f}")

    winner = max(results, key=lambda r: r.sharpe)
    print(f"\n  WINNER: {winner.name}  (Sharpe {winner.sharpe:+.2f})")

    # ----- artifacts -----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for r in results:
        eq = r.equity.dropna()
        if len(eq):
            ax.plot(eq.index, eq.values, label=f"{r.name} (Sh {r.sharpe:+.2f})")
    ax.set_title("v1 vs v2 walk-forward equity curves")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v2_curves.png", dpi=110)
    plt.close(fig)

    metrics = {
        "data": {"kline": KLINE, "n_bars": int(len(df)),
                 "from": str(df.index[0]), "to": str(df.index[-1])},
        "config": {
            "rolling_window": ROLLING_WINDOW, "entry_z": ENTRY_Z,
            "exit_z": EXIT_Z, "stop_z": STOP_Z, "fee_bps": FEE_BPS,
            "ml_threshold": ML_THRESHOLD,
            "target_daily_vol": TARGET_DAILY_VOL,
            "walk_train_bars": WALK_TRAIN_BARS,
            "walk_test_bars": WALK_TEST_BARS,
            "hurst_window": HURST_WINDOW, "adf_window": ADF_WINDOW,
        },
        "regime_counts": {str(k): int(v) for k, v in counts.items()},
        "ml": {"oos_aucs": aucs,
               "auc_mean":   float(np.mean(aucs)) if aucs else None,
               "auc_median": float(np.median(aucs)) if aucs else None},
        "results": {r.name: {
            "sharpe":            r.sharpe,  "cagr": r.cagr,
            "max_dd":            r.max_dd,  "n_trades": r.n_trades,
            "pct_time_in_trade": r.pct_time_in_trade,
            "total_return_pct":  r.total_return_pct,
        } for r in results},
        "winner": winner.name,
    }
    (OUT_DIR / "v2_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str))

    # Save the FINAL fold's model for the live bot. Train on the most-recent
    # full window so the online prediction is as fresh as we can make it.
    final_train_idx = df.index[len(df) - WALK_TRAIN_BARS:]
    Xtr_final = feat.loc[final_train_idx].dropna()
    ytr_final = label_forward(df, raw_pos_v2, horizon=24).loc[Xtr_final.index].dropna()
    Xtr_final = Xtr_final.loc[ytr_final.index]
    if len(ytr_final) >= 20 and ytr_final.nunique() == 2:
        clf_final = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.04, max_iter=400,
            l2_regularization=1.0, min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )
        clf_final.fit(Xtr_final.values, ytr_final.values)
        with (OUT_DIR / "model.pkl").open("wb") as f:
            pickle.dump({"model": clf_final, "feature_cols": feat_cols}, f)
        meta = {
            "trained_on_bars": int(len(ytr_final)),
            "trained_on_from":  str(Xtr_final.index[0]),
            "trained_on_to":    str(Xtr_final.index[-1]),
            "ml_threshold":     ML_THRESHOLD,
            "feature_cols":     feat_cols,
            "kline":            KLINE,
            "rolling_window":   ROLLING_WINDOW,
        }
        (OUT_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2,
                                                             default=str))
        print(f"  Saved model.pkl ({len(ytr_final)} training rows) and model_meta.json")
    else:
        print("  Not enough training data for the final-fold model; "
              "model.pkl NOT updated.")

    # Plain-text report.
    lines = ["v2 walk-forward results", "=" * 60]
    lines += [f"Data: {len(df)} {KLINE} bars, {df.index[0]} -> {df.index[-1]}",
              ""]
    lines += ["Regime distribution:"]
    for k, v in counts.items():
        lines.append(f"  {str(k):>14s}  {int(v):>5d}  ({v/len(df)*100:.1f}%)")
    lines += ["", "Walk-forward backtest:",
              f"  {'variant':50s}  Sharpe  CAGR%  MaxDD%  Trades  Time%  TotRet%"]
    for r in results:
        lines.append(f"  {r.name:50s}  {r.sharpe:>+5.2f}  {r.cagr*100:>+5.1f}  "
                     f"{r.max_dd*100:>+6.1f}  {r.n_trades:>5d}  "
                     f"{r.pct_time_in_trade*100:>4.1f}  "
                     f"{r.total_return_pct:>+7.2f}")
    if aucs:
        lines += ["", f"ML OOS AUC: mean={np.mean(aucs):.3f} "
                       f"median={np.median(aucs):.3f} ({len(aucs)} folds)"]
    lines += ["", f"Winner: {winner.name}", ""]
    (OUT_DIR / "v2_report.txt").write_text("\n".join(lines))
    print(f"\nWrote: {OUT_DIR/'v2_curves.png'}")
    print(f"       {OUT_DIR/'v2_metrics.json'}")
    print(f"       {OUT_DIR/'v2_report.txt'}")
    print(f"       {OUT_DIR/'model.pkl'}  (live-bot scoring artifact)")
    print(f"       {OUT_DIR/'model_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
