"""
ETH/BTC pairs-trading research script
======================================

Runs the empirical work that decides what should ship into the live bot:

  1. Pull ~14 months of 1h ETHUSDT / BTCUSDT closes from Binance, paginated.
  2. Stationarity / cointegration tests on log(ETH/BTC) and log(ETH) vs log(BTC).
  3. Rolling hedge-ratio estimation (beta from log returns; cointegration
     coefficient from rolling regression).
  4. Walk-forward backtest comparing four variants:
        - Z-score, dollar-weighted (the current bot)
        - Z-score, beta-weighted (rolling beta from log returns)
        - Z-score, cointegration-weighted (rolling EG coefficient)
        - Z + ML gating (HistGradientBoostingClassifier filters entries)
  5. Print a final recommendation table the live bot can be wired against.

We use sklearn's HistGradientBoostingClassifier rather than LightGBM because:
  - It is a histogram-based gradient booster (same family as LightGBM/XGBoost)
    so the modelling capacity is comparable for tabular financial features.
  - It ships with sklearn, no libomp / OpenMP requirement.
  - The empirical asset-pricing literature (Gu, Kelly & Xiu 2020;
    Krauss, Do & Huck 2017) finds that boosted trees match or beat LSTMs on
    return-prediction tasks of this size, especially on short panels — and
    this dataset is short by ML standards (~10k bars).

Run:
    python analysis/research.py

Outputs:
    analysis/research_report.txt    (human-readable summary)
    analysis/research_metrics.json  (machine-readable for the live bot)
    analysis/research_curves.png    (equity curves)
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ----------------- config -----------------
KLINE              = "1h"
TARGET_BARS        = 13_000          # ~1.5 years of 1h closes
ROLLING_WINDOW     = 30              # bars used for z-score + bands
ENTRY_Z, EXIT_Z, STOP_Z = 2.0, 0.3, 3.5
FEE_BPS            = 4.0             # per leg, taker
WALK_TRAIN_BARS    = 24 * 90         # 90 days train
WALK_TEST_BARS     = 24 * 30         # 30 days test
ML_PROB_THRESHOLD  = 0.55            # only enter when classifier > this
RANDOM_STATE       = 7
PERIODS_PER_YEAR   = 24 * 365
OUT_DIR            = Path(__file__).parent
# ------------------------------------------


# ---------- data fetch (paginated) ----------

BINANCE_BASES = ["https://api.binance.com", "https://api.binance.us"]
KLINE_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
            "4h": 14_400_000, "1d": 86_400_000}


def _fetch_chunk(symbol: str, interval: str, end_ms: int,
                 limit: int = 1000) -> pd.DataFrame:
    """Fetch up to `limit` klines ending at `end_ms` (inclusive). Tries .com
    first, falls back to .us on geo-block."""
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
        df["time"]   = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df.set_index("time")[["close", "volume"]]
    raise last or RuntimeError("all Binance hosts failed")


def fetch_long_history(symbol: str, interval: str,
                       target_bars: int) -> pd.DataFrame:
    """Walk backwards in time pulling 1000-bar chunks until target_bars are
    accumulated. Deduplicates and sorts ascending."""
    bar_ms = KLINE_MS[interval]
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    frames: list[pd.DataFrame] = []
    pulled = 0
    while pulled < target_bars:
        chunk = _fetch_chunk(symbol, interval, end_ms, limit=1000)
        if chunk.empty:
            break
        frames.append(chunk)
        pulled += len(chunk)
        # Step back: oldest bar's openTime - one bar
        oldest = int(chunk.index.min().timestamp() * 1000)
        end_ms = oldest - bar_ms
        time.sleep(0.15)  # be polite
        if len(chunk) < 1000:
            break
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.columns = [f"{symbol}_close", f"{symbol}_volume"]
    return df.tail(target_bars)


def load_data() -> pd.DataFrame:
    print(f"Pulling {TARGET_BARS} {KLINE} bars of BTCUSDT and ETHUSDT...")
    btc = fetch_long_history("BTCUSDT", KLINE, TARGET_BARS)
    eth = fetch_long_history("ETHUSDT", KLINE, TARGET_BARS)
    df = pd.concat([btc, eth], axis=1).dropna()
    df = df.rename(columns={
        "BTCUSDT_close":  "btc",  "BTCUSDT_volume": "btc_vol",
        "ETHUSDT_close":  "eth",  "ETHUSDT_volume": "eth_vol",
    })
    df["ratio"]   = df["eth"] / df["btc"]
    df["log_eth"] = np.log(df["eth"])
    df["log_btc"] = np.log(df["btc"])
    df["log_r"]   = np.log(df["ratio"])
    df["ret_eth"] = df["log_eth"].diff()
    df["ret_btc"] = df["log_btc"].diff()
    df["ret_r"]   = df["log_r"].diff()
    df = df.dropna()
    print(f"Got {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    return df


# ---------- statistical tests ----------

def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller. Reject null (non-stationary) when p < 0.05."""
    s = series.dropna()
    stat, pv, _, _, crit, _ = adfuller(s, autolag="AIC")
    return {"adf_stat": float(stat), "p_value": float(pv),
            "crit_5pct": float(crit["5%"]),
            "stationary_5pct": bool(pv < 0.05)}


def engle_granger(eth_log: pd.Series, btc_log: pd.Series) -> dict:
    """Engle-Granger two-step cointegration test.

    Step 1: regress log(ETH) on a constant + log(BTC) -> hedge ratio gamma.
    Step 2: ADF on the residual. Reject null (no cointegration) when p < 0.05.
    """
    a, b = eth_log.dropna(), btc_log.dropna()
    a, b = a.align(b, join="inner")
    score, pv, _ = coint(a, b)
    X = add_constant(b.values)
    model = OLS(a.values, X).fit()
    alpha, gamma = float(model.params[0]), float(model.params[1])
    resid = a - (alpha + gamma * b)
    return {"eg_stat": float(score), "p_value": float(pv),
            "cointegrated_5pct": bool(pv < 0.05),
            "alpha": alpha, "gamma": gamma,
            "resid_std": float(resid.std())}


# ---------- hedge-ratio estimators ----------

def rolling_beta_logret(eth_ret: pd.Series, btc_ret: pd.Series,
                        window: int) -> pd.Series:
    """β from rolling cov / var on log returns (no intercept)."""
    cov = eth_ret.rolling(window).cov(btc_ret)
    var = btc_ret.rolling(window).var()
    return (cov / var).rename("beta_logret")


def rolling_cointegration_gamma(eth_log: pd.Series, btc_log: pd.Series,
                                window: int) -> pd.Series:
    """Rolling EG hedge ratio: regress log(ETH) ~ a + γ·log(BTC) on each
    rolling window. Uses centered/de-meaned slope which is identical to OLS
    slope under a constant term."""
    e = eth_log - eth_log.rolling(window).mean()
    b = btc_log - btc_log.rolling(window).mean()
    cov = (e * b).rolling(window).mean()
    var = (b * b).rolling(window).mean()
    return (cov / var).rename("gamma_eg")


# ---------- signal & backtest ----------

def zscore(log_r: pd.Series, window: int) -> pd.Series:
    mu = log_r.rolling(window).mean()
    sd = log_r.rolling(window).std().where(log_r.rolling(window).std() > 0)
    return (log_r - mu) / sd


def state_machine_positions(z: pd.Series,
                            entry: float, exit_: float, stop: float) -> pd.Series:
    pos = np.zeros(len(z))
    cur = 0
    for i, zt in enumerate(z.values):
        if np.isnan(zt):
            pos[i] = 0
            continue
        if cur == 0:
            if zt >  entry:
                cur = -1
            elif zt < -entry:
                cur = +1
        elif cur == +1 and (zt > -exit_ or zt < -stop):
            cur = 0
        elif cur == -1 and (zt <  exit_ or zt >  stop):
            cur = 0
        pos[i] = cur
    return pd.Series(pos, index=z.index, name="position").astype(float)


@dataclass
class BacktestResult:
    name: str
    sharpe: float
    cagr: float
    max_dd: float
    hit_rate: float
    n_trades: int
    pct_time_in_trade: float
    total_return_pct: float
    equity: pd.Series = field(repr=False)


def backtest(eth_ret: pd.Series, btc_ret: pd.Series, ratio_ret: pd.Series,
             pos: pd.Series, hedge: pd.Series | None,
             fee_bps: float = FEE_BPS,
             name: str = "") -> BacktestResult:
    """Bar-level backtest.

    If `hedge` is None: dollar-neutral on the ratio. P&L = pos[t-1] * Δlog(ratio).
    If `hedge` is given: P&L = pos[t-1] * (ret_eth[t] - hedge[t-1] * ret_btc[t]).
    Fees are charged on every position change, two legs each side (so 2x for
    flat->in or in->flat, 4x for in->flip-in).
    """
    pos_lag   = pos.shift(1).fillna(0.0)
    if hedge is None:
        gross = pos_lag * ratio_ret
    else:
        h_lag = hedge.shift(1).fillna(0.0)
        gross = pos_lag * (eth_ret - h_lag * btc_ret)

    turnover = pos.diff().abs().fillna(0.0)
    fees     = turnover * (fee_bps / 1e4) * 2.0
    net      = (gross - fees).fillna(0.0)
    eq       = np.exp(net.cumsum())

    r = net.dropna()
    if r.std() == 0 or len(r) < 2:
        return BacktestResult(name, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, eq)

    mu  = r.mean() * PERIODS_PER_YEAR
    sig = r.std() * math.sqrt(PERIODS_PER_YEAR)
    sharpe = mu / sig if sig else 0.0
    cagr   = float(np.exp(mu) - 1)
    peak   = eq.cummax()
    max_dd = float((eq / peak - 1).min())
    active = pos_lag.abs() > 0
    hit    = float((r[active.reindex(r.index, fill_value=False)] > 0).mean()) \
                if active.sum() else 0.0
    p = pos.fillna(0).astype(int).values
    nt, prev = 0, 0
    for v in p:
        if v != 0 and prev == 0:
            nt += 1
        prev = v
    return BacktestResult(
        name=name,
        sharpe=float(sharpe),
        cagr=cagr,
        max_dd=max_dd,
        hit_rate=hit,
        n_trades=nt,
        pct_time_in_trade=float(active.mean()),
        total_return_pct=float((eq.iloc[-1] - 1.0) * 100.0),
        equity=eq,
    )


# ---------- ML overlay ----------

def build_features(df: pd.DataFrame, z: pd.Series, hedge: pd.Series) -> pd.DataFrame:
    """Engineered features for the gating classifier. All values are at-bar
    only (no lookahead). Lags applied where we use 'past' returns."""
    f = pd.DataFrame(index=df.index)
    f["z"]        = z
    f["z_diff"]   = z.diff()
    f["z_abs"]    = z.abs()
    f["ret_r_1"]  = df["ret_r"].shift(1)
    f["ret_r_3"]  = df["ret_r"].rolling(3).sum().shift(1)
    f["ret_r_12"] = df["ret_r"].rolling(12).sum().shift(1)
    f["vol_r_24"] = df["ret_r"].rolling(24).std().shift(1)
    f["mom_eth"]  = df["ret_eth"].rolling(24).sum().shift(1)
    f["mom_btc"]  = df["ret_btc"].rolling(24).sum().shift(1)
    f["vol_eth"]  = df["ret_eth"].rolling(24).std().shift(1)
    f["vol_btc"]  = df["ret_btc"].rolling(24).std().shift(1)
    f["volr_eth"] = (df["eth_vol"] / df["eth_vol"].rolling(48).mean()).shift(1)
    f["volr_btc"] = (df["btc_vol"] / df["btc_vol"].rolling(48).mean()).shift(1)
    f["hedge"]    = hedge
    f["hedge_d"]  = hedge.diff()
    return f


def label_forward_profit(df: pd.DataFrame, z: pd.Series, hedge: pd.Series,
                         horizon: int = 24) -> pd.Series:
    """Binary label: would a trade opened *now* in the direction implied by z
    be profitable `horizon` bars later, dollar/beta weighted (ignoring fees)?

    direction = -sign(z): a positive z means short ETH / long BTC.
    P&L proxy   = direction * (ret_eth_h - hedge * ret_btc_h)
    Label = 1 if P&L > 0 else 0. Only labels rows where |z| > ENTRY_Z so the
    classifier learns which entries are worth taking.
    """
    fwd_eth = (df["log_eth"].shift(-horizon) - df["log_eth"])
    fwd_btc = (df["log_btc"].shift(-horizon) - df["log_btc"])
    direction = -np.sign(z)
    pnl = direction * (fwd_eth - hedge * fwd_btc)
    y = (pnl > 0).astype(float)
    mask = (z.abs() > ENTRY_Z) & pnl.notna()
    y[~mask] = np.nan
    return y.rename("y")


def ml_gated_positions(df: pd.DataFrame, z: pd.Series, hedge: pd.Series,
                       train_bars: int, test_bars: int,
                       prob_threshold: float = ML_PROB_THRESHOLD
                       ) -> tuple[pd.Series, list[float]]:
    """Walk-forward: roll a (train_bars, test_bars) window across the series.
    Train classifier on completed labels (those with full forward window) in
    train slice, predict on test slice; gate base entries by predicted
    probability >= threshold. Returns the gated position series and OOS AUCs.
    """
    feats  = build_features(df, z, hedge)
    labels = label_forward_profit(df, z, hedge, horizon=24)
    base_pos = state_machine_positions(z, ENTRY_Z, EXIT_Z, STOP_Z)
    ml_pos   = base_pos.copy()

    aucs: list[float] = []
    n = len(df)
    start = train_bars
    while start + test_bars <= n:
        train_idx = df.index[start - train_bars:start]
        test_idx  = df.index[start:start + test_bars]

        # We can only train on rows where the forward window is complete
        # (label not NaN) AND the row has a candidate entry (|z| > ENTRY_Z).
        Xtr = feats.loc[train_idx].dropna()
        ytr = labels.loc[Xtr.index].dropna()
        Xtr = Xtr.loc[ytr.index]

        if len(ytr) < 50 or ytr.nunique() < 2:
            start += test_bars
            continue

        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        )
        clf.fit(Xtr.values, ytr.values)

        Xte_full = feats.loc[test_idx].dropna()
        if Xte_full.empty:
            start += test_bars
            continue

        proba = clf.predict_proba(Xte_full.values)[:, 1]
        # Only gate at *entry* bars: where base_pos transitions from 0 -> ±1.
        bp = base_pos.loc[Xte_full.index].values
        prev_bp = base_pos.shift(1).fillna(0).loc[Xte_full.index].values
        gated = np.array(bp, dtype=float)
        for i in range(len(Xte_full)):
            if prev_bp[i] == 0 and bp[i] != 0 and proba[i] < prob_threshold:
                gated[i] = 0.0
            # Once a gated entry is rejected, propagate flat forward until the
            # base signal goes flat too (so we don't half-enter mid-trade).
            if i > 0 and gated[i - 1] == 0 and bp[i] != 0 and prev_bp[i] != 0:
                gated[i] = 0.0
        ml_pos.loc[Xte_full.index] = gated

        # OOS AUC on rows where we have labels available.
        yte = labels.loc[Xte_full.index].dropna()
        if len(yte) >= 20 and yte.nunique() == 2:
            aucs.append(float(roc_auc_score(
                yte.values, proba[Xte_full.index.isin(yte.index)])))

        start += test_bars

    return ml_pos, aucs


# ---------- walk-forward base / hedge variants ----------

def walk_forward_variant(df: pd.DataFrame, name: str,
                         hedge: pd.Series | None,
                         train_bars: int, test_bars: int) -> BacktestResult:
    """Build a walk-forward signal: at each test bar, recompute the z-score
    using only data up to (but not including) that bar. The state machine
    runs on the full OOS-z series so positions are determined only from
    information available at decision time. Then backtest end-to-end.
    """
    log_r = df["log_r"]
    z = pd.Series(np.nan, index=df.index)
    n = len(df)
    start = train_bars
    while start + test_bars <= n:
        train_log = log_r.iloc[start - train_bars:start]
        test_idx  = df.index[start:start + test_bars]
        # Use only training-period mu/sd to z-score the test slice (true OOS).
        mu = train_log.mean()
        sd = train_log.std()
        if sd > 0:
            z.loc[test_idx] = (log_r.loc[test_idx] - mu) / sd
        start += test_bars

    pos = state_machine_positions(z, ENTRY_Z, EXIT_Z, STOP_Z)
    return backtest(df["ret_eth"], df["ret_btc"], df["ret_r"],
                    pos, hedge=hedge, name=name)


def main() -> int:
    df = load_data()

    print("\n=== Stationarity / cointegration tests (full sample) ===")
    adf_logr = adf_test(df["log_r"])
    eg       = engle_granger(df["log_eth"], df["log_btc"])
    print(f"  ADF on log(ETH/BTC):  stat={adf_logr['adf_stat']:+.3f} "
          f"p={adf_logr['p_value']:.4f} -> "
          f"{'STATIONARY' if adf_logr['stationary_5pct'] else 'non-stationary'} @ 5%")
    print(f"  Engle-Granger:         stat={eg['eg_stat']:+.3f} "
          f"p={eg['p_value']:.4f} gamma={eg['gamma']:.4f} -> "
          f"{'COINTEGRATED' if eg['cointegrated_5pct'] else 'not cointegrated'} @ 5%")

    print("\n=== Hedge-ratio estimators ===")
    beta = rolling_beta_logret(df["ret_eth"], df["ret_btc"],
                               window=24 * 30).bfill().ffill()
    gamma = rolling_cointegration_gamma(df["log_eth"], df["log_btc"],
                                        window=24 * 30).bfill().ffill()
    print(f"  Rolling beta (30d, log returns):  median={beta.median():.3f} "
          f"min={beta.min():.3f} max={beta.max():.3f}")
    print(f"  Rolling gamma (30d, EG slope):    median={gamma.median():.3f} "
          f"min={gamma.min():.3f} max={gamma.max():.3f}")

    print("\n=== Walk-forward backtests ===")
    print(f"  train={WALK_TRAIN_BARS} bars (~{WALK_TRAIN_BARS/24:.0f} days), "
          f"test={WALK_TEST_BARS} bars (~{WALK_TEST_BARS/24:.0f} days)")

    res_dollar = walk_forward_variant(df, "z+dollar (current bot)",
                                      hedge=None,
                                      train_bars=WALK_TRAIN_BARS,
                                      test_bars=WALK_TEST_BARS)

    res_beta   = walk_forward_variant(df, "z+beta-weighted",
                                      hedge=beta,
                                      train_bars=WALK_TRAIN_BARS,
                                      test_bars=WALK_TEST_BARS)

    res_gamma  = walk_forward_variant(df, "z+cointegration-weighted",
                                      hedge=gamma,
                                      train_bars=WALK_TRAIN_BARS,
                                      test_bars=WALK_TEST_BARS)

    # ML overlay on top of the best of (dollar, beta, gamma)
    base_candidates = {"dollar": (None, res_dollar),
                       "beta":   (beta, res_beta),
                       "gamma":  (gamma, res_gamma)}
    best_key = max(base_candidates, key=lambda k: base_candidates[k][1].sharpe)
    best_hedge, best_base = base_candidates[best_key]

    print(f"\n  Best non-ML hedge: '{best_key}'  Sharpe={best_base.sharpe:.2f}")
    print("  Training HistGradientBoostingClassifier overlay walk-forward...")

    # Rebuild z under the same walk-forward scheme so the gating uses the same
    # OOS z series the base backtest saw.
    log_r = df["log_r"]
    z_oos = pd.Series(np.nan, index=df.index)
    start = WALK_TRAIN_BARS
    while start + WALK_TEST_BARS <= len(df):
        train_log = log_r.iloc[start - WALK_TRAIN_BARS:start]
        test_idx  = df.index[start:start + WALK_TEST_BARS]
        mu, sd = train_log.mean(), train_log.std()
        if sd > 0:
            z_oos.loc[test_idx] = (log_r.loc[test_idx] - mu) / sd
        start += WALK_TEST_BARS

    h_for_ml = (best_hedge if best_hedge is not None
                else pd.Series(1.0, index=df.index))
    ml_pos, aucs = ml_gated_positions(df, z_oos, h_for_ml,
                                      train_bars=WALK_TRAIN_BARS,
                                      test_bars=WALK_TEST_BARS)
    res_ml = backtest(df["ret_eth"], df["ret_btc"], df["ret_r"],
                      ml_pos, hedge=best_hedge,
                      name=f"z+{best_key}+ML-gate")
    auc_str = (f"  HistGBM OOS AUC across {len(aucs)} folds: "
               f"mean={np.mean(aucs):.3f} median={np.median(aucs):.3f}"
               if aucs else "  (insufficient labels for AUC)")
    print(auc_str)

    results = [res_dollar, res_beta, res_gamma, res_ml]

    print("\n=== Walk-forward results ===")
    print(f"  {'variant':32s}  Sharpe  CAGR  MaxDD  Hit%  Trades  Time%  TotalRet%")
    for r in results:
        print(f"  {r.name:32s}  {r.sharpe:>5.2f}  {r.cagr*100:>4.1f}  "
              f"{r.max_dd*100:>5.1f}  {r.hit_rate*100:>4.1f}  "
              f"{r.n_trades:>5d}  {r.pct_time_in_trade*100:>4.1f}  "
              f"{r.total_return_pct:>+8.2f}")

    winner = max(results, key=lambda r: r.sharpe)
    print(f"\n  WINNER (by walk-forward Sharpe): {winner.name}")

    # ---------- artifacts ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for r in results:
        eq = r.equity.dropna()
        if len(eq):
            ax.plot(eq.index, eq.values, label=f"{r.name} (Sh {r.sharpe:.2f})")
    ax.set_title("Walk-forward equity curves (dollar-neutral, net of fees)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "research_curves.png", dpi=110)
    plt.close(fig)

    metrics = {
        "data": {
            "kline":   KLINE,
            "n_bars":  int(len(df)),
            "from":    str(df.index[0]),
            "to":      str(df.index[-1]),
        },
        "tests": {
            "adf_log_ratio": adf_logr,
            "engle_granger": eg,
        },
        "hedge_ratios": {
            "rolling_beta_median":  float(beta.median()),
            "rolling_gamma_median": float(gamma.median()),
        },
        "walk_forward": {
            r.name: {
                "sharpe":            r.sharpe,
                "cagr":              r.cagr,
                "max_dd":            r.max_dd,
                "hit_rate":          r.hit_rate,
                "n_trades":          r.n_trades,
                "pct_time_in_trade": r.pct_time_in_trade,
                "total_return_pct":  r.total_return_pct,
            } for r in results
        },
        "ml_aucs": aucs,
        "winner":  winner.name,
        "best_non_ml_hedge": best_key,
    }
    (OUT_DIR / "research_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str))

    report_lines = [
        "ETH/BTC research summary",
        "=" * 60,
        f"Data: {len(df)} {KLINE} bars, {df.index[0]} -> {df.index[-1]}",
        "",
        "Stationarity / cointegration:",
        f"  ADF on log(ETH/BTC):  p={adf_logr['p_value']:.4f}  "
        f"-> {'STATIONARY' if adf_logr['stationary_5pct'] else 'non-stationary'} at 5%",
        f"  Engle-Granger:         p={eg['p_value']:.4f}  gamma={eg['gamma']:.4f}  "
        f"-> {'COINTEGRATED' if eg['cointegrated_5pct'] else 'not cointegrated'} at 5%",
        "",
        "Hedge-ratio summary (30-day rolling):",
        f"  beta (cov/var on log returns): median={beta.median():.3f}",
        f"  gamma (EG slope):              median={gamma.median():.3f}",
        "",
        "Walk-forward backtest results:",
        f"  train={WALK_TRAIN_BARS} bars, test={WALK_TEST_BARS} bars, "
        f"fees={FEE_BPS} bps/leg",
        "",
        f"  {'variant':32s}  Sharpe  CAGR%  MaxDD%  Hit%  Trades  Time%  TotRet%",
    ]
    for r in results:
        report_lines.append(
            f"  {r.name:32s}  {r.sharpe:>5.2f}  {r.cagr*100:>5.1f}  "
            f"{r.max_dd*100:>6.1f}  {r.hit_rate*100:>4.1f}  "
            f"{r.n_trades:>5d}  {r.pct_time_in_trade*100:>4.1f}  "
            f"{r.total_return_pct:>+7.2f}")
    if aucs:
        report_lines += ["",
            f"HistGBM OOS AUC across {len(aucs)} folds: "
            f"mean={np.mean(aucs):.3f} median={np.median(aucs):.3f}"]
    report_lines += ["", f"WINNER (by walk-forward Sharpe): {winner.name}", ""]
    (OUT_DIR / "research_report.txt").write_text("\n".join(report_lines))

    print(f"\nWrote: {OUT_DIR/'research_curves.png'}")
    print(f"       {OUT_DIR/'research_metrics.json'}")
    print(f"       {OUT_DIR/'research_report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
