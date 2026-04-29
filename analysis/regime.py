"""
Regime detector for the ETH/BTC strategy.

Two independent estimators of "is the log-ratio currently trending vs mean-
reverting?" plus a hysteresis layer to prevent the bot from flapping back
and forth on the boundary.

Estimators
----------

1. **Hurst exponent (rescaled-range)**.
     H ~ 0.5  -> random walk (no edge in either direction)
     H >  0.55 -> persistent / trending  -> momentum mode
     H <  0.45 -> anti-persistent / mean-reverting -> mean-revert mode
   Reference: Hurst (1951), "Long-term storage capacity of reservoirs".
   Implementation here is the canonical R/S analysis on log returns of the
   spread.

2. **Rolling ADF p-value**.
     p < 0.05 -> log-spread is stationary on this window  -> mean-revert mode
     p > 0.20 -> non-stationary / trending                -> momentum mode
     0.05-0.20 -> indeterminate (treat as 'no change')
   Computed on a rolling window of the log-ratio itself, not its returns.

Combined regime
---------------

A bar is labelled `momentum` if both estimators agree (H >= 0.55 AND ADF
p > 0.20), `mean-revert` if both agree the other way (H <= 0.45 AND ADF
p < 0.05), otherwise `indeterminate` (we keep the previous regime and
require a *consecutive*-bar streak to confirm a switch — that's the
hysteresis). The hysteresis depth is configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


Regime = Literal["momentum", "mean-revert", "indeterminate"]


# ---------- Hurst (rescaled-range) ----------

def _hurst_rs(x: np.ndarray) -> float:
    """Rescaled-range estimator of Hurst exponent on a 1D array.

    Splits x into 8 dyadic chunk sizes from len(x)/8 up to len(x)/2,
    computes mean(R/S) per chunk-size, fits a log-log line, slope is H.
    """
    n = len(x)
    if n < 32:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    cum = np.cumsum(x)

    # Dyadic chunk sizes.
    sizes = []
    s = 8
    while s <= n // 2:
        sizes.append(s)
        s *= 2
    if not sizes:
        return float("nan")

    rs_vals = []
    for sz in sizes:
        n_chunks = n // sz
        rs_per = []
        for k in range(n_chunks):
            start = k * sz
            block = x[start:start + sz]
            cm = np.cumsum(block - block.mean())
            r = cm.max() - cm.min()
            sd = block.std(ddof=1)
            if sd > 0 and r > 0:
                rs_per.append(r / sd)
        if rs_per:
            rs_vals.append((sz, np.mean(rs_per)))
    if len(rs_vals) < 2:
        return float("nan")

    sizes_arr = np.log(np.array([s for s, _ in rs_vals]))
    rs_arr    = np.log(np.array([r for _, r in rs_vals]))
    slope, _  = np.polyfit(sizes_arr, rs_arr, 1)
    return float(slope)


def rolling_hurst(series: pd.Series, window: int) -> pd.Series:
    """Hurst exponent on a rolling window of values. The series should be
    the log-spread (`log(ETH/BTC)`)."""
    out = np.full(len(series), np.nan)
    vals = series.values
    for i in range(window, len(series)):
        out[i] = _hurst_rs(vals[i - window:i])
    return pd.Series(out, index=series.index, name="hurst")


# ---------- Rolling ADF ----------

def rolling_adf_pvalue(series: pd.Series, window: int,
                       step: int = 1) -> pd.Series:
    """Rolling ADF p-value on the LEVEL of the spread (test for stationarity).
    `step` lets us subsample (ADF is the bottleneck) — we forward-fill between
    steps to keep the index aligned."""
    out = np.full(len(series), np.nan)
    vals = series.values
    for i in range(window, len(series), step):
        block = vals[i - window:i]
        if np.all(np.isfinite(block)) and block.std() > 0:
            try:
                _, pv, *_ = adfuller(block, autolag="AIC")
                out[i] = pv
            except Exception:
                pass
    s = pd.Series(out, index=series.index, name="adf_p")
    if step > 1:
        s = s.ffill()
    return s


# ---------- Combined regime classifier ----------

@dataclass
class RegimeBands:
    h_trend:        float = 0.55     # H >= this -> momentum candidate
    h_meanrev:      float = 0.45     # H <= this -> mean-revert candidate
    adf_meanrev:    float = 0.05     # p < this  -> mean-revert candidate
    adf_trend:      float = 0.20     # p > this  -> momentum candidate
    hysteresis_bars: int  = 6        # consecutive agreeing bars to switch
    initial_regime: Regime = "momentum"   # starting regime before warmup


def classify(hurst: pd.Series, adf_p: pd.Series,
             bands: RegimeBands = RegimeBands()) -> pd.Series:
    """Walks the joint series and returns a regime label per bar with
    hysteresis. Both inputs should be aligned on the same index."""
    n = len(hurst)
    out = np.array([bands.initial_regime] * n, dtype=object)
    current: Regime = bands.initial_regime
    streak: dict[Regime, int] = {"momentum": 0, "mean-revert": 0,
                                  "indeterminate": 0}
    h_vals = hurst.values
    p_vals = adf_p.values

    for i in range(n):
        h, p = h_vals[i], p_vals[i]
        if not (np.isfinite(h) and np.isfinite(p)):
            out[i] = current
            continue

        if h >= bands.h_trend and p > bands.adf_trend:
            candidate: Regime = "momentum"
        elif h <= bands.h_meanrev and p < bands.adf_meanrev:
            candidate = "mean-revert"
        else:
            candidate = "indeterminate"

        # Hysteresis: only switch if we've seen `hysteresis_bars` consecutive
        # signals for the candidate regime. `indeterminate` never switches us
        # out of the current regime — it's the no-info state.
        for k in streak:
            streak[k] = streak[k] + 1 if k == candidate else 0
        if (candidate != "indeterminate" and candidate != current
                and streak[candidate] >= bands.hysteresis_bars):
            current = candidate

        out[i] = current

    return pd.Series(out, index=hurst.index, name="regime")


def detect_regime(log_ratio: pd.Series,
                  hurst_window: int = 240,        # 240 4h bars = 40 days
                  adf_window: int = 240,
                  adf_step: int = 4,
                  bands: RegimeBands = RegimeBands()
                  ) -> pd.DataFrame:
    """One-call regime detection. Returns a DataFrame with columns
    `hurst`, `adf_p`, `regime`."""
    h  = rolling_hurst(log_ratio, hurst_window)
    p  = rolling_adf_pvalue(log_ratio, adf_window, step=adf_step)
    return pd.DataFrame({
        "hurst":  h,
        "adf_p":  p,
        "regime": classify(h, p, bands),
    })
