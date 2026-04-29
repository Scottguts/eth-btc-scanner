#!/bin/bash
# Local launcher for the ETH/BTC pairs-trading bot.
# Designed to be invoked from a macOS LaunchAgent so the bot survives
# logout, terminal close, and crashes (LaunchAgent re-spawns it).
#
# Wrapping the python invocation in `caffeinate -i -s` prevents the system
# from going to idle/system sleep while the bot is actively running.
# (Note: `caffeinate` does NOT prevent lid-close sleep on a laptop. If you
# need true 24/7 with the lid closed, run this on a server instead.)

set -euo pipefail

BOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BOT_DIR"

# Pull TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, INTERVAL_MINUTES, KLINE out of
# bot.env (mode 0600). bot.env is gitignored.
if [ -f "$BOT_DIR/bot.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$BOT_DIR/bot.env"
    set +a
else
    echo "ERROR: $BOT_DIR/bot.env not found. Copy bot.env.example to bot.env and fill in your Telegram values." >&2
    exit 1
fi

INTERVAL_MINUTES="${INTERVAL_MINUTES:-15}"
KLINE="${KLINE:-1h}"

PY=/usr/local/bin/python3
[ -x "$PY" ] || PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13
[ -x "$PY" ] || PY=$(command -v python3)

# `caffeinate -i` blocks idle sleep, `-s` blocks system sleep while on power.
# `exec` replaces the shell so launchd correctly tracks the python process.
exec /usr/bin/caffeinate -i -s "$PY" -u "$BOT_DIR/eth_btc_pairs.py" \
    --loop \
    --interval-minutes "$INTERVAL_MINUTES" \
    --kline "$KLINE" \
    --alert-telegram
