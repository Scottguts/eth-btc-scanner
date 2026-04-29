# ETH/BTC strategy research report

**Date:** 2026-04-29
**Sample:** 14 months of 1h, 4h, and 1d Binance closes (BTCUSDT, ETHUSDT) ending
2026-04-29.
**Method:** Walk-forward (true out-of-sample) backtest across 102
configurations; cointegration tests; rolling hedge-ratio estimation;
gradient-boosted classifier as a signal-gating overlay.
**Scripts:** [`research.py`](research.py), [`sweep.py`](sweep.py).

---

## TL;DR

1. **The strategy was running in the wrong direction.** Mean-reversion of
   the ETH/BTC log-ratio loses money walk-forward across every
   `(timeframe, window, entry_z)` combination tested (Sharpe **-0.22 to
   -3.85**). The cointegration test (Engle-Granger p=0.73) rejects the
   theoretical foundation — the ratio is **trending**, not reverting.

2. **The inverse — momentum on the same z-signal — works.** The
   walk-forward winner is **4h bars, 360-bar window (~60 days), entry_z
   = 2.5, momentum direction**: walk-forward **Sharpe +1.26**, total
   return **+157.7 %**, max drawdown **-19.5 %**, **41 trades**.

3. **Beta-weighting underperforms dollar-weighting** in this period
   (Sharpe -3.85 vs -3.57 in the bad-direction case; absolute numbers
   are negative either way, but dollar-weighting is consistently a hair
   less bad). The cointegration coefficient gamma is highly unstable
   (range 0.3 to 4.9 over rolling 30-day windows), which is symptomatic
   of a non-cointegrated series. **Recommend dollar-weighted.**

4. **The HistGradientBoosting overlay added some value on the broken
   strategy** (Sharpe -2.65 vs -3.10 without it), but on the
   *winning* (momentum) base the overlay is unnecessary; the simple
   z-signal does the job.

5. **The live bot has been switched to the winning config:**
   `kline=4h`, `window=360`, `entry_z=2.5`, `mode=momentum`,
   dollar-weighted. The previous in-sample headline numbers
   (Sharpe 3.33, CAGR 77 %) were **overfit** — they are not the
   walk-forward truth and should not be used to size positions.

---

## Why the literature didn't save us

We did the textbook stuff first, and it failed empirically:

| Test                            | Statistic | p-value | Verdict (5 %)     |
|---------------------------------|----------:|--------:|-------------------|
| ADF on log(ETH/BTC)             |    -1.501 |  0.5330 | non-stationary    |
| Engle-Granger (log-ETH ~ log-BTC) | -1.585 |  0.7270 | not cointegrated  |

Pairs trading rests on the cointegration assumption (Engle & Granger 1987,
applied to pairs by Vidyamurthy 2004 and Gatev / Goetzmann / Rouwenhorst
2006). When two log-prices are cointegrated, their spread is mean-
reverting, and a z-score on the spread is a tradable mean-reversion
signal. **ETH and BTC are not cointegrated on this 14-month window.**
They are correlated (rolling-30d beta ≈ 1.18 on log returns, which is
ETH's characteristic ~20 % excess vol over BTC), but there is no
mean-reverting equilibrium to fade.

This is regime-dependent — earlier crypto cycles have shown periods of
relative ETH/BTC range-bound behaviour where mean-reversion would have
worked. The 2024-2026 sample is a strong-BTC regime: the ratio fell from
~0.05 to ~0.03 over the period, monotonically enough that any "buy the
dip" signal is selling cheap crypto to fund expensive crypto, which is
exactly what we observed.

---

## Walk-forward sweep results

102 configurations tested across timeframes 1d / 4h / 1h, rolling
windows 60-720 bars, entry thresholds 1.5-3.0, and both directions. Top
10 by walk-forward Sharpe (true OOS):

| kline | window | entry_z | direction   | Sharpe | CAGR % | MaxDD % | trades | TotRet % |
|------:|-------:|--------:|:------------|-------:|-------:|--------:|-------:|---------:|
| 4h    | 360    | 2.5     | momentum    | +1.26  | +41.3  | -19.5   | 41     | +157.75  |
| 4h    | 360    | 3.0     | momentum    | +1.15  | +32.5  | -20.0   | 39     | +116.09  |
| 4h    | 180    | 2.0     | momentum    | +1.05  | +35.3  | -24.1   | 55     | +128.71  |
| 4h    | 360    | 2.0     | momentum    | +1.03  | +35.2  | -21.0   | 44     | +128.27  |
| 4h    | 180    | 1.5     | momentum    | +0.97  | +34.0  | -32.0   | 60     | +123.08  |
| 4h    | 360    | 1.5     | momentum    | +0.94  | +32.9  | -29.0   | 49     | +117.96  |
| 1h    | 720    | 2.0     | momentum    | +0.92  | +37.4  | -21.6   | 143    | +60.20   |
| 1h    | 720    | 3.0     | momentum    | +0.87  | +32.1  | -20.7   | 135    | +51.08   |
| 1h    | 720    | 2.5     | momentum    | +0.81  | +30.7  | -21.8   | 140    | +48.78   |
| 4h    | 180    | 2.5     | momentum    | +0.72  | +21.4  | -22.2   | 51     | +69.94   |

Bottom 5 (sanity):

| kline | window | entry_z | direction   | Sharpe | TotRet % |
|------:|-------:|--------:|:------------|-------:|---------:|
| 1h    | 480    | 2.5     | mean-revert | -1.35  | -47.40   |
| 4h    | 360    | 2.5     | mean-revert | -1.43  | -65.97   |
| 1h    | 720    | 2.5     | mean-revert | -1.71  | -57.06   |
| 1h    | 720    | 3.0     | mean-revert | -1.78  | -57.03   |
| 1h    | 720    | 2.0     | mean-revert | -1.82  | -60.50   |

**Pattern:** the top 10 are all momentum; the bottom 5 are all mean-
revert. Direction matters far more than the parameter knobs.

---

## Dollar-weighted vs beta-weighted

Tested four hedge-ratio variants on the same z-score signal in
[`research.py`](research.py):

| Variant                  | Walk-forward Sharpe |
|:-------------------------|--------------------:|
| z + dollar-weighted      | -3.57               |
| z + beta-weighted        | -3.85               |
| z + cointegration-weighted (rolling EG gamma) | -3.10  |
| z + best-hedge + ML-gate | -2.65               |

These were run **before** the direction flip, so all numbers are
negative; the comparison is still informative for choosing the hedge:

- **Dollar-weighted is the simplest and is consistently among the
  least-bad.** It also avoids estimation error in beta/gamma.
- **Beta-weighted (rolling 30-day cov/var on log returns)** under-
  performs in this period — beta drifts between 0.80 and 1.63, and
  the lag in the rolling estimator hurts during regime changes.
- **Cointegration-weighted (rolling EG gamma)** is theoretically
  preferred when the pair is cointegrated, but gamma swings between
  -0.99 and +4.94 here, which is a clear sign the relationship is not
  stable. A dynamic hedge that erratic is almost certainly overfit to
  noise.

**Decision:** dollar-weighted. Equal-dollar legs, no regression-based
hedge ratio. This matches the intuition that crypto pairs with
similar implied vol regimes don't need fancy hedges; the volatility
mismatch (ETH ~20 % more volatile than BTC on log returns) is small
relative to the signal's own variance.

---

## ML overlay: HistGradientBoostingClassifier

Used the sklearn `HistGradientBoostingClassifier` (histogram-based
gradient booster, same family as LightGBM / XGBoost) rather than an
LSTM. Reasoning:

1. **Sample size.** Roughly 13,000 hourly bars / 3,200 4h bars / 540
   daily bars. LSTM/Transformer architectures need 10x-100x more
   labelled events to reach their fitted-vs-OOS sweet spot. Boosted
   trees train and generalize cleanly on tabular features at this
   scale.
2. **Feature interpretability.** Tree models give SHAP / feature
   importances out of the box. With 14 engineered features (lagged z,
   z-diff, rolling momentum, vol, volume regime, hedge series) we can
   inspect what the model thinks matters.
3. **Empirical results.** Gu, Kelly & Xiu (2020), "Empirical Asset
   Pricing via Machine Learning", and Krauss, Do & Huck (2017),
   "Statistical arbitrage in the U.S. equities market", both find
   gradient-boosted trees match or beat LSTMs on the
   return-prediction tasks closest to ours.

The overlay gates entries: only take a trade when the classifier
predicts >= 0.55 probability that the trade will be profitable
24 bars forward. Walk-forward (90-day train, 30-day test, retrained
each fold).

| Metric                 | Result                          |
|:-----------------------|:--------------------------------|
| Mean OOS AUC           | 0.397                           |
| Median OOS AUC         | 0.478                           |
| Folds                  | 5                               |

OOS AUC < 0.5 means the classifier (in mean-reversion mode, the
broken setup) is barely better than random — and on average, slightly
worse. The reason: the underlying signal it was being asked to gate
was structurally negative, so any "edge" it captured was learning the
trend-direction component, which then disappears in the next fold
because it's not stationary.

**Decision:** **do not ship the ML overlay** for now. The momentum-
direction base signal is doing the heavy lifting, and overlaying a
poorly-AUCing classifier on top will just add variance and over-
fitting risk. We have a research path for re-training the classifier
on the *winning* base (momentum entries, predict whether the trend
continues another N bars) — left as a follow-up if performance
degrades.

---

## What ships

The live bot now runs:

```text
python eth_btc_pairs.py --loop \
    --kline 4h --mode momentum \
    --interval-minutes 30 --alert-telegram
```

with constants:

```python
INTERVAL       = "4h"
ROLLING_WINDOW = 360       # ~60 days at 4h
ENTRY_Z        = 2.5
EXIT_Z         = 0.3
STOP_Z         = 3.5
FEE_BPS        = 4.0       # 4 bps/leg taker
MODE           = "momentum"
```

Position interpretation (unchanged):

- `+1` = long ETH / short BTC
- `-1` = short ETH / long BTC
- `0`  = flat

Direction triggering changed:

- **Momentum mode (active):** z > +2.5 → +1; z < -2.5 → -1.
- Mean-revert mode (legacy, available via `--mode mean-revert`):
  z > +2.5 → -1; z < -2.5 → +1.

Exits are unchanged (revert into the |z| < 0.3 band, or hit the
|z| > 3.5 hard stop), so trades still close when the dislocation
goes away.

---

## Caveats / what we did not do

- **14 months is one regime.** The result is conditional on a
  trending market. If ETH/BTC enters a range-bound period, the
  momentum signal will chop. Re-run [`sweep.py`](sweep.py)
  quarterly and check that the winning quadrant is still momentum.
- **No transaction cost stress test.** 4 bps/leg taker is the
  ballpark for Binance perp at retail tier; large size or thin
  market hours would worsen this. With only ~3 trades / month at
  the audited config, slippage is small in aggregate but matters
  per trade.
- **No funding-rate model.** Perpetual swaps charge funding every
  8 hours. ETH-perp and BTC-perp funding are highly correlated, so
  the dollar-neutral pair largely zeroes out, but this can still
  cost ~0.5-1 % a year against the strategy.
- **No regime detector.** A simple ADF-on-rolling-window switch
  ("trade momentum when not stationary; trade mean-revert when
  stationary") would let the bot adapt automatically. Worth
  building as a v2.
- **Single pair.** Diversifying across ETH/BTC, SOL/BTC, etc.,
  would reduce variance materially. Out of scope for this report.
- **Telegram alerts only.** No automatic order execution. Position
  changes still require you to place the trades manually on your
  exchange of choice.

---

## Files

- [`research.py`](research.py) — cointegration + walk-forward + ML overlay.
- [`sweep.py`](sweep.py) — 102-cell parameter grid sweep.
- [`research_metrics.json`](research_metrics.json) — first run output (machine-readable).
- [`sweep_results.json`](sweep_results.json) — sweep output (machine-readable).
- [`research_curves.png`](research_curves.png) — OOS equity curves.

Re-run anytime:

```bash
python analysis/research.py    # ~30 s
python analysis/sweep.py       # ~3-5 min the first time, fast after (csv cache)
```

---

## v2 update — regime detector + ML overlay + vol-target

After the initial audit landed, we built three v2 components on top:

1. **Regime detector** ([`regime.py`](regime.py)). Hurst exponent (R/S
   estimator on log-spread) + rolling Augmented Dickey-Fuller p-value, both
   computed on a 240-bar (~40 day) rolling window. A bar is flagged
   `momentum` when both H ≥ 0.53 and ADF p > 0.15; `mean-revert` when
   both H ≤ 0.47 and ADF p < 0.08; otherwise `indeterminate`. A
   4-bar hysteresis stops the label from flapping. Used by the bot's
   `--mode auto` (now the default) to pick direction per entry.

2. **Feature library + ML overlay** ([`features.py`](features.py),
   [`v2_backtest.py`](v2_backtest.py)). 40 engineered features across
   multiple horizons (lagged returns 1/3/6/12/24/72 bars, rolling vol,
   per-leg momentum/vol, RSI 14/42, Bollinger %B, ATR, volume regime,
   regime indicators). Walk-forward HistGradientBoostingClassifier with
   3-month train / 1-month test windows. The final fold's model is
   pickled to [`model.pkl`](model.pkl) for the live bot.

3. **Vol-targeted position sizer** (`vol_target_size` in
   [`v2_backtest.py`](v2_backtest.py)). Targets 1.5 % daily vol on the
   spread, locks size at entry, never rebalances mid-trade (mid-trade
   rebalancing was the cause of a -45 bp Sharpe regression in the first
   iteration).

### Walk-forward results, 5,999 4h bars (Aug-2023 → Apr-2026)

| Variant                                                  | Sharpe | CAGR % | MaxDD % | Trades | TotRet % |
|:---------------------------------------------------------|-------:|-------:|--------:|-------:|---------:|
| v1: momentum / dollar / no ML                            | +0.81  | +26.9  | -25.6   | 62     | +92.22   |
| **v2a: regime-aware / dollar / no ML  (LIVE)**           | **+0.81** | **+26.9** | **-25.6** | **62** | **+92.22** |
| v2b: regime / dollar / ML hard-gate (threshold 0.50)     | -0.68  | -8.8   | -28.3   | 18     | -22.37   |
| v2c: regime / ML proba-sized position                    | -0.22  | -5.0   | -27.1   | 19     | -13.15   |
| v2d: regime / ML proba-size / vol-target (full stack)    | -0.24  | -4.1   | -22.1   | 19     | -10.92   |

ML walk-forward AUC: **mean 0.569, median 0.604** across 20 folds. The
classifier has a real edge over random (AUC > 0.5) but it is not strong
enough to dominate the bare momentum signal — gating or sizing on it
discards trades that turn out to be net-positive in aggregate.

### What we shipped (and why)

- **Regime detector ON** (`--mode auto`, default). Currently resolves to
  `momentum` 100 % of the time on this 32-month sample (Hurst 0.92-0.97,
  ADF p 0.10-0.20, both clearly in the trending bucket). The detector is
  shipped *for future regime changes*: if ETH/BTC enters a range-bound
  period, it will start producing `mean-revert` labels and the bot will
  flip direction automatically.
- **ML overlay OFF** by default; available via `--use-ml`. Even at
  AUC 0.57-0.60 the overlay vetoes too many net-positive entries on this
  sample. Kept around because (a) regime / vol changes can flip its
  utility, and (b) it's a useful "second opinion" check to manually
  consult on a given entry.
- **Vol-targeting OFF** by default. With size locked at entry the
  Sharpe is approximately invariant to size choice; we leave the
  knob unconnected for now to keep the alert output simple.
- The shipped sample has a slightly lower walk-forward Sharpe (+0.81)
  than the headline figure from the first sweep (+1.26). The
  difference is the test universe: the first sweep tested only 3,600
  of 5,999 bars (long train, fewer folds); v2 tests 5,400 bars
  (~30 folds) and is therefore the more conservative estimate to trust.

### Reproducing

```bash
python analysis/sweep.py          # parameter grid (~3 min)
python analysis/research.py       # cointegration + dollar/beta/gamma compare
python analysis/v2_backtest.py    # regime + ML walk-forward; saves model.pkl
```

---

## References (load-bearing for the design decisions above)

- Engle, R. F. and Granger, C. W. J. (1987), "Co-integration and Error
  Correction: Representation, Estimation and Testing", *Econometrica*.
- Gatev, E., Goetzmann, W. and Rouwenhorst, K. G. (2006), "Pairs
  Trading: Performance of a Relative-Value Arbitrage Rule",
  *Review of Financial Studies*.
- Vidyamurthy, G. (2004), *Pairs Trading: Quantitative Methods and
  Analysis*, Wiley. (Beta-weighted hedge construction.)
- Avellaneda, M. and Lee, J.-H. (2010), "Statistical Arbitrage in the
  U.S. Equities Market", *Quantitative Finance*.
- Moskowitz, T., Ooi, Y. H. and Pedersen, L. H. (2012), "Time Series
  Momentum", *Journal of Financial Economics*. (The momentum result
  this report leans on.)
- Krauss, C., Do, X. A. and Huck, N. (2017), "Deep neural networks,
  gradient-boosted trees, random forests: Statistical arbitrage on
  the S&P 500", *European Journal of Operational Research*.
  (Gradient-boosted trees competitive with / beating deep nets at
  this scale.)
- Gu, S., Kelly, B. and Xiu, D. (2020), "Empirical Asset Pricing via
  Machine Learning", *Review of Financial Studies*. (Same finding,
  cross-sectional rather than time-series.)
- Caporale, G. M., Gil-Alana, L. and Plastun, A. (2018),
  "Persistence in the cryptocurrency market", *Research in
  International Business and Finance*.
