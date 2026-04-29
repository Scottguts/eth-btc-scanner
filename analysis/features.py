"""
Feature engineering for the v2 ML overlay.

All features are computed with strict no-lookahead semantics: at row t, every
feature value is computable from data available at or before t. The library
expects a DataFrame with at least these columns:

    eth, btc, eth_vol, btc_vol  (close prices and volumes)
    log_eth, log_btc, log_r     (logs and log-spread)
    ret_eth, ret_btc, ret_r     (log returns)

If any are missing they are derived. The classifier (`HistGradientBoosting`)
does not need scaling, but it benefits from features that capture different
horizons of the same phenomenon, so we include several lookback windows.

Reference: Krauss, Do & Huck (2017), "Statistical arbitrage in the U.S.
equities market", Sec. 3.3 — feature panel of lagged returns at multiple
horizons consistently dominates raw price features for tree models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Technical indicators ----------

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index, Wilder smoothing approximation via EWMA."""
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{window}")


def bollinger_pct(close: pd.Series, window: int = 20,
                  k: float = 2.0) -> pd.Series:
    """Position of close within the Bollinger band, in [0, 1]; 0.5 = at SMA."""
    mu = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = mu + k * sd
    lower = mu - k * sd
    out = (close - lower) / (upper - lower)
    return out.rename(f"bb_pct_{window}")


def atr_proxy(close: pd.Series, window: int = 14) -> pd.Series:
    """ATR proxy from close-only data: rolling stddev of log returns scaled by
    sqrt(window). Cheap and robust without high/low."""
    return close.pct_change().rolling(window).std().rename(f"atr_{window}")


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling realized vol assuming returns is per-bar log-returns
    on a 4h frequency (6 bars/day, 365 days/yr -> 2190 bars/yr)."""
    return (returns.rolling(window).std() * np.sqrt(2190)).rename(
        f"rv_{window}")


def volume_ratio(volume: pd.Series, fast: int = 12, slow: int = 96) -> pd.Series:
    """Short-term over long-term volume — a regime indicator. fast=12 (2 days),
    slow=96 (16 days) on 4h bars."""
    return (volume.rolling(fast).mean()
            / volume.rolling(slow).mean()).rename(
        f"volr_{fast}_{slow}")


# ---------- Master feature builder ----------

def ensure_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log_eth" not in out:
        out["log_eth"] = np.log(out["eth"])
    if "log_btc" not in out:
        out["log_btc"] = np.log(out["btc"])
    if "log_r"   not in out:
        out["log_r"]   = np.log(out["eth"] / out["btc"])
    if "ret_eth" not in out:
        out["ret_eth"] = out["log_eth"].diff()
    if "ret_btc" not in out:
        out["ret_btc"] = out["log_btc"].diff()
    if "ret_r"   not in out:
        out["ret_r"]   = out["log_r"].diff()
    return out


def build_features(df: pd.DataFrame, z: pd.Series,
                   regime: pd.Series | None = None,
                   hurst:  pd.Series | None = None,
                   adf_p:  pd.Series | None = None) -> pd.DataFrame:
    """Returns a DataFrame of features aligned to `df.index`. All values are
    .shift(1)-ed where they would otherwise leak the current bar's information
    into the prediction.
    """
    df = ensure_logs(df)
    f = pd.DataFrame(index=df.index)

    # --- z-score features ---
    f["z"]         = z
    f["z_abs"]     = z.abs()
    f["z_diff"]    = z.diff()
    f["z_mom_6"]   = z - z.shift(6)
    f["z_mom_24"]  = z - z.shift(24)

    # --- Spread momentum / vol at multiple horizons ---
    for h in (1, 3, 6, 12, 24, 72):
        f[f"ret_r_{h}"] = df["ret_r"].rolling(h).sum().shift(1)
    for h in (12, 24, 72, 168):
        f[f"vol_r_{h}"] = df["ret_r"].rolling(h).std().shift(1)

    # --- Per-leg momentum / vol ---
    for h in (12, 24, 72):
        f[f"mom_eth_{h}"] = df["ret_eth"].rolling(h).sum().shift(1)
        f[f"mom_btc_{h}"] = df["ret_btc"].rolling(h).sum().shift(1)
        f[f"vol_eth_{h}"] = df["ret_eth"].rolling(h).std().shift(1)
        f[f"vol_btc_{h}"] = df["ret_btc"].rolling(h).std().shift(1)

    # --- Technicals on the spread ratio ---
    ratio_close = (df["eth"] / df["btc"]).rename("ratio")
    f["rsi_r_14"]   = rsi(ratio_close, 14).shift(1)
    f["rsi_r_42"]   = rsi(ratio_close, 42).shift(1)
    f["bb_r_20"]    = bollinger_pct(ratio_close, 20).shift(1)
    f["atr_r_14"]   = atr_proxy(ratio_close, 14).shift(1)

    # --- Volume regime ---
    if "eth_vol" in df:
        f["volr_eth"] = volume_ratio(df["eth_vol"]).shift(1)
    if "btc_vol" in df:
        f["volr_btc"] = volume_ratio(df["btc_vol"]).shift(1)

    # --- Realized vol on each leg (annualised) ---
    f["rv_eth_24"]  = realized_vol(df["ret_eth"], 24).shift(1)
    f["rv_btc_24"]  = realized_vol(df["ret_btc"], 24).shift(1)
    f["rv_r_24"]    = realized_vol(df["ret_r"], 24).shift(1)

    # --- Regime features ---
    if hurst is not None:
        f["hurst"]   = hurst.shift(1)
    if adf_p is not None:
        f["adf_p"]   = adf_p.shift(1)
    if regime is not None:
        # Encode the regime so the model can learn regime-conditional behavior.
        f["is_momentum"]    = (regime.shift(1) == "momentum").astype(float)
        f["is_meanrevert"]  = (regime.shift(1) == "mean-revert").astype(float)

    return f
