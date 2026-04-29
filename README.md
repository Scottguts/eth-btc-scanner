# ETH/BTC pairs signal bot

A small Python bot that watches the ETH/BTC price ratio, detects unusually
large dislocations on a rolling z-score basis, and emits a **signal**
(long ETH / short BTC, or the inverse, or flat). It does **not** execute
orders — it sends Telegram or email alerts and writes a trade log; you
take the trades manually.

**Default mode: `auto`** — the v2 regime detector (Hurst exponent +
rolling Augmented Dickey-Fuller test on the log-spread) picks the entry
direction (momentum vs mean-revert) per entry, so the bot adapts if the
ETH/BTC regime shifts. On the audited 32 months of 4h data the regime
is 100 % momentum, so `auto` currently behaves identically to `momentum`
(walk-forward Sharpe **+0.81**, total return **+92.2 %**, MaxDD
**-25.6 %**, 62 trades). Mean-reversion variants of the same z-signal
were Sharpe **-0.22 to -3.85** in walk-forward — the cointegration test
(Engle-Granger p=0.73) rejects cointegration so the textbook pairs-
trading bet on revert-to-mean has no theoretical foundation in this
period.

A walk-forward HistGradientBoosting classifier (sklearn) is trained on
40 engineered features and shipped as [`analysis/model.pkl`](analysis/model.pkl);
its OOS AUC is **0.57-0.60** across 20 folds — a real edge over random,
but not strong enough to gate or size the base signal profitably on the
audited sample, so it is **opt-in** via `--use-ml`. Full audit:
[`analysis/research_report.md`](analysis/research_report.md).

> Not financial advice. Past walk-forward performance is not a guarantee of
> live results. Paper-trade before risking real money.

---

## What it does each cycle

1. Pulls 4h closes for `BTCUSDT` and `ETHUSDT` from Binance's public REST API.
   (Auto-falls-back to `binance.us` if the primary host geo-blocks.)
2. Computes `log(ETH/BTC)` and z-scores it against a 360-bar (~60 day) rolling
   window.
3. Runs a state machine. **In momentum mode (default):**
   - `z > +2.5` → **+1**: long ETH / short BTC (bet trend continues up)
   - `z < -2.5` → **-1**: short ETH / long BTC (bet trend continues down)
   - `|z| < 0.3` → flat (dislocation cleared)
   - `|z| > 3.5` → hard stop

   In mean-revert mode (legacy, `--mode mean-revert`) the entry directions
   are flipped — see the audit for why momentum was chosen.
4. Backtests the rule dollar-neutral with realistic per-leg fees. The
   displayed in-sample Sharpe is on the recent 1000-bar window only and is
   labelled clearly as **not** the audited number.
5. Writes:
   - `eth_btc_pairs.csv` — full timeseries (price, ratio, z, position, returns)
   - `eth_btc_pairs.png` — three-panel chart (ratio, z-score, equity curve)
   - `eth_btc_trades.csv` — one row per pairs trade (entry/exit/PnL)
   - `eth_btc_latest_signal.json` — current snapshot for downstream tools
   - `eth_btc_state.json` — last-seen position (used to detect flips)
6. If running with `--loop` and `--alert-telegram` (or `--alert-email`), pings
   you on every position change with a concrete, easy-to-read alert.

## Reading the output

The console output and Telegram alerts use a fixed-width layout that tells
you, in plain English:

```
SIGNAL FLIP: 0  ->  +1   (LONG ETH / SHORT BTC)
Time:        2026-04-29 14:00:00+00:00

WHY:   ETH looks cheap vs BTC (z below entry).
DO:    Buy ETH, sell BTC, equal dollars per leg.

PRICES
  ETH         $    2,293.52
  BTC         $   77,041.72
  ETH/BTC          0.029769
  Z-score          -2.130

TARGETS  (where to act)
  Take profit  ratio ~ 0.030200   (z = 0,  +1.45% from now)
  Hard stop    ratio < 0.029400   (z = -3.5)
  Soft exit    |z| < 0.3
```

Every closed pairs trade is also logged to `eth_btc_trades.csv` with entry
time, exit time, side, bars held, percentage P&L, and total fees in bps —
suitable for opening in a spreadsheet to review performance.

---

## Quickstart (local)

```bash
git clone https://github.com/<you>/eth_btc_bot.git
cd eth_btc_bot
pip install -r requirements.txt

# One-shot: print today's signal, write CSV/PNG/JSON, and exit.
python eth_btc_pairs.py

# Live poller every 15 minutes on 1h bars.
python eth_btc_pairs.py --loop --interval-minutes 15 --kline 1h
```

### Adding Telegram alerts

1. Open Telegram, message `@BotFather`, run `/newbot`, save the token.
2. Open `t.me/<your_bot_username>` and tap **Start**.
3. Set the env vars and run the chat-id helper:

   ```bash
   export TELEGRAM_BOT_TOKEN=12345:abc...
   python eth_btc_pairs.py --telegram-find-chat-id
   # -> chat_id = 98765   (Your Name)
   export TELEGRAM_CHAT_ID=98765
   ```

4. Test it:

   ```bash
   python eth_btc_pairs.py --test-alert
   ```

5. Run live:

   ```bash
   python eth_btc_pairs.py --loop --alert-telegram
   ```

### Email / SMS gateway alerts

Set `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, `ALERT_TO` (and optionally
`SMTP_PORT`, `SMTP_SSL`, `ALERT_FROM`) and run with `--alert-email`.
Carrier email-to-SMS gateways have grown unreliable; prefer Telegram or a
real inbox.

---

## Run on GitHub Actions (zero-cost, no laptop)

This is the cleanest way to run the bot if you don't have a server. Each
firing of the cron is a fresh Ubuntu VM; total runtime is ~1 min/run with
pip caching.

[`.github/workflows/poll.yml`](.github/workflows/poll.yml) is preconfigured to:

- Run every 30 minutes via `cron`. With ~1 min/run that's ~1,500 min/month,
  inside the **2,000 min/mo free tier** for personal private repos.
- Install deps with `actions/setup-python` pip cache.
- Restore the prior `eth_btc_state.json` from the Actions cache so flip
  detection persists between runs.
- Run `python eth_btc_pairs.py --once-alert --kline 4h --mode momentum
  --alert-telegram`.
- Upload the snapshot/CSV/PNG as artifacts (30-day retention) for debugging.

To use it:

1. **Push this repo** to GitHub (any visibility — public or private).
2. **Add two repository Secrets** at *Settings -> Secrets and variables ->
   Actions*:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. **Enable Actions** if your account has it disabled, then either wait for
   the next cron tick or trigger the workflow manually via *Actions -> 
   poll-eth-btc -> Run workflow*.
4. Watch the run logs in the Actions tab; Telegram alerts arrive on every
   position flip exactly as they do locally.

**Public vs private:** GitHub Actions minutes are unlimited for public
repos and capped at 2,000/mo free for private. With 30-min cron + pip
cache, this fits the private free tier. The bot has no privileged data
in the repo (secrets are stored in repo Secrets, not committed) so public
is also fine — your call.

## Run locally on macOS (LaunchAgent)

If you just want it running on your laptop, the included `run_bot.sh` and
`com.scottguts.eth-btc-bot.plist` set it up as a user LaunchAgent that:

- starts automatically when you log in,
- restarts itself on crash (with a 30 s back-off),
- wraps the bot in `caffeinate -i -s` so the system won't idle-sleep,
- logs to `~/Library/Logs/eth_btc_bot/bot.log`,
- reads secrets from `bot.env` (mode 0600), not from the plist.

```bash
cp bot.env.example bot.env       # then fill in real Telegram values
chmod 600 bot.env

# Edit the absolute paths in the plist if your repo isn't at
# /Users/<you>/eth_btc_bot, then:
cp com.scottguts.eth-btc-bot.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.scottguts.eth-btc-bot.plist
launchctl enable    gui/$(id -u)/com.scottguts.eth-btc-bot
launchctl kickstart -k gui/$(id -u)/com.scottguts.eth-btc-bot

# Watch it work:
tail -f ~/Library/Logs/eth_btc_bot/bot.log
```

**Important caveat:** macOS sleeps the *whole machine* when the laptop lid
closes. `caffeinate` blocks idle/system sleep but cannot block lid-close
sleep on a bare laptop. If you need true 24/7 operation with the lid
closed, use clamshell mode (external display + keyboard + power) or run
the bot on a small VPS via the systemd installer below.

To stop it:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.scottguts.eth-btc-bot.plist
```

## Deploy on a server (systemd)

See [`install_bot_on_server.sh`](install_bot_on_server.sh). On a fresh
Ubuntu 22.04 / 24.04 box as root:

```bash
# From your laptop:
cp bot.env.example bot.env       # then fill in real tokens
scp eth_btc_pairs.py bot.env install_bot_on_server.sh root@SERVER:/root/

# On the server:
ssh root@SERVER bash /root/install_bot_on_server.sh
```

The installer:

- Adds 1 GB swap if RAM < 900 MB (pandas needs the headroom on a $5 droplet).
- Installs Python deps system-wide.
- Drops the bot into `/root/eth_btc_bot/`.
- Loads secrets from `/root/eth_btc_bot/bot.env` (mode 0600) via systemd's
  `EnvironmentFile=` — secrets never appear in the unit file or the process
  list.
- Registers a systemd service that auto-restarts on crash and starts on boot.

Manage it with the usual `systemctl status / restart / stop eth-btc-bot`.

---

## Configuration

All knobs live at the top of [`eth_btc_pairs.py`](eth_btc_pairs.py). Defaults
are the walk-forward winner:

| Constant         | Default     | Meaning                                       |
|------------------|-------------|-----------------------------------------------|
| `LOOKBACK_LIMIT` | 1000        | Binance candles per request (max 1000).       |
| `INTERVAL`       | `4h`        | Kline interval.                               |
| `ROLLING_WINDOW` | 360         | Z-score lookback in bars (~60 days at 4h).    |
| `ENTRY_Z`        | 2.5         | Open trade when |z| crosses this.             |
| `EXIT_Z`         | 0.3         | Close trade when z reverts back to here.      |
| `STOP_Z`         | 3.5         | Hard stop if z keeps running against you.     |
| `FEE_BPS`        | 4.0         | Per-leg taker fee in basis points.            |
| `MODE`           | `momentum`  | `momentum` (default) or `mean-revert`.        |

CLI flags: `--kline {1d,4h,1h,15m,5m}`, `--mode {momentum,mean-revert}`,
`--source {binance,cmc}`, `--once-alert` (CI/cron one-shot), `--loop`,
`--alert-telegram`, `--alert-email`.

## CLI cheat sheet

```bash
python eth_btc_pairs.py                                    # one-shot, defaults
python eth_btc_pairs.py --loop                             # live poller
python eth_btc_pairs.py --once-alert --alert-telegram      # cron / CI one-shot
python eth_btc_pairs.py --loop --interval-minutes 30       # custom poll cadence
python eth_btc_pairs.py --mode mean-revert                 # use legacy direction
python eth_btc_pairs.py --kline 1h --mode mean-revert      # legacy 1h setup
python eth_btc_pairs.py --source cmc --loop                # CMC fallback
python eth_btc_pairs.py --telegram-find-chat-id            # discover chat id
python eth_btc_pairs.py --test-alert                       # send a test
python eth_btc_pairs.py --loop --alert-telegram            # live + Telegram
```

## Notes & limitations

- **No order execution.** This is a signal generator. Wire the JSON snapshot
  into your own execution layer if you want it automated.
- **Backtest is dollar-neutral on the log-ratio.** Real fills will diverge.
  The 4 bps per leg is a taker-perp ballpark; tune `FEE_BPS` to your venue.
- **Free CMC tier only gives latest quotes**, so the bot accumulates its own
  history. The signal goes live once `ROLLING_WINDOW + 1` samples are
  collected.
- **Telegram alerts go through the public Bot API.** Your bot token grants
  full control of the bot; keep `bot.env` mode 0600 and out of git.

## License

MIT — see headers, do what you want, no warranty.
