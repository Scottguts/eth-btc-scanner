# ETH/BTC pairs-trading signal bot

A small Python bot that watches the ETH/BTC price ratio, detects when ETH gets
unusually cheap or expensive vs BTC on a rolling z-score basis, and emits a
mean-reversion **signal** (long ETH / short BTC, or the inverse, or flat).

It does **not** execute orders. It is a signal generator with a backtest, a
chart, a trade log, and Telegram/email alerts on every position flip.

> Not financial advice. Paper-trade it before risking real money.

---

## What it does each cycle

1. Pulls daily/hourly closes for `BTCUSDT` and `ETHUSDT` from Binance's public
   REST API. (Auto-falls-back to `binance.us` if the primary host geo-blocks.)
2. Computes `log(ETH/BTC)` and z-scores it against a 30-bar rolling window.
3. Runs a state machine:
   - `z < -2.0` → **long ratio** (long ETH, short BTC)
   - `z > +2.0` → **short ratio** (short ETH, long BTC)
   - `|z| < 0.3` → exit toward mean
   - `|z| > 3.5` → hard stop
4. Backtests the rule dollar-neutral with realistic per-leg fees.
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

All knobs live at the top of [`eth_btc_pairs.py`](eth_btc_pairs.py):

| Constant         | Default | Meaning                                       |
|------------------|---------|-----------------------------------------------|
| `LOOKBACK_LIMIT` | 1000    | Binance candles per request (max 1000).       |
| `INTERVAL`       | `1d`    | Default kline interval.                       |
| `ROLLING_WINDOW` | 30      | Z-score lookback in bars.                     |
| `ENTRY_Z`        | 2.0     | Open trade when |z| crosses this.             |
| `EXIT_Z`         | 0.3     | Close trade when z reverts back to here.      |
| `STOP_Z`         | 3.5     | Hard stop if z keeps running against you.     |
| `FEE_BPS`        | 4.0     | Per-leg taker fee in basis points.            |

The CLI also exposes `--kline {1d,4h,1h,15m,5m}` and `--source {binance,cmc}`.

## CLI cheat sheet

```bash
python eth_btc_pairs.py                         # one-shot
python eth_btc_pairs.py --loop                  # live poller
python eth_btc_pairs.py --loop --interval-minutes 5 --kline 1h
python eth_btc_pairs.py --source cmc --loop     # CMC fallback (free tier)
python eth_btc_pairs.py --telegram-find-chat-id # discover chat id
python eth_btc_pairs.py --test-alert            # send a test through every channel
python eth_btc_pairs.py --loop --alert-telegram # live + Telegram alerts
python eth_btc_pairs.py --loop --alert-email    # live + email/SMS alerts
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
