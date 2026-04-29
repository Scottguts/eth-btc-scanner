#!/bin/bash
# One-shot installer for the ETH/BTC pairs-trading bot.
# Run on a fresh Ubuntu 22.04 / 24.04 server as root.
#
# Expects two files to already be uploaded to /root/ before this runs:
#   eth_btc_pairs.py         — the bot itself
#   bot.env                  — TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, etc.
#
# Example bot.env:
#     TELEGRAM_BOT_TOKEN=12345:abc...
#     TELEGRAM_CHAT_ID=98765
#     INTERVAL_MINUTES=15
#     KLINE=1h
#
# Upload via:
#     scp eth_btc_pairs.py bot.env root@SERVER:/root/
#     ssh root@SERVER bash /root/install_bot_on_server.sh

set -euo pipefail

BOT_DIR=/root/eth_btc_bot
SERVICE=/etc/systemd/system/eth-btc-bot.service
ENV_FILE_SRC=/root/bot.env
ENV_FILE_DST=$BOT_DIR/bot.env

if [ ! -f "$ENV_FILE_SRC" ] && [ ! -f "$ENV_FILE_DST" ]; then
    echo "ERROR: $ENV_FILE_SRC not found."
    echo
    echo "Create one with at least:"
    echo "    TELEGRAM_BOT_TOKEN=..."
    echo "    TELEGRAM_CHAT_ID=..."
    echo "    INTERVAL_MINUTES=15"
    echo "    KLINE=1h"
    echo
    echo "Then scp it to the server before running this installer."
    exit 1
fi

echo
echo "==> Checking memory and adding swap if needed..."
MEM_MB=$(awk '/^MemTotal/ {print int($2/1024)}' /proc/meminfo)
echo "    RAM: ${MEM_MB} MB"
if [ "$MEM_MB" -lt 900 ] && [ ! -f /swapfile ]; then
    echo "    Creating 1 GB swap file (needed for pandas on small droplets)..."
    fallocate -l 1G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null
    swapon /swapfile
    grep -q '/swapfile' /etc/fstab || echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
    echo "    Swap active."
fi

echo
echo "==> Updating package list and installing Python..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-pip ca-certificates

echo
echo "==> Installing Python libraries (requests, pandas, numpy, matplotlib)..."
pip3 install --break-system-packages --quiet \
    requests pandas numpy matplotlib

echo
echo "==> Setting up bot directory at $BOT_DIR..."
mkdir -p "$BOT_DIR"

# Move the script and env file in (overwriting any older copy).
if [ -f /root/eth_btc_pairs.py ]; then
    mv /root/eth_btc_pairs.py "$BOT_DIR/eth_btc_pairs.py"
fi
if [ -f "$ENV_FILE_SRC" ]; then
    mv "$ENV_FILE_SRC" "$ENV_FILE_DST"
    chmod 600 "$ENV_FILE_DST"
fi
if [ ! -f "$BOT_DIR/eth_btc_pairs.py" ]; then
    echo "ERROR: $BOT_DIR/eth_btc_pairs.py not found."
    echo "Did you scp eth_btc_pairs.py up to /root/ before running this?"
    exit 1
fi
if [ ! -f "$ENV_FILE_DST" ]; then
    echo "ERROR: $ENV_FILE_DST not found."
    exit 1
fi

# Read INTERVAL_MINUTES + KLINE out of the env file (defaults if absent), so
# we can bake them into the systemd ExecStart line without exposing the
# Telegram token there.
# shellcheck disable=SC1090
set -a; source "$ENV_FILE_DST"; set +a
INTERVAL_MINUTES="${INTERVAL_MINUTES:-30}"
KLINE="${KLINE:-4h}"
MODE="${MODE:-auto}"

echo
echo "==> Creating systemd service..."
cat > "$SERVICE" <<SERVICE_EOF
[Unit]
Description=ETH/BTC pairs-trading signal bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$BOT_DIR
EnvironmentFile=$ENV_FILE_DST
ExecStart=/usr/bin/python3 $BOT_DIR/eth_btc_pairs.py --loop --interval-minutes $INTERVAL_MINUTES --kline $KLINE --mode $MODE --alert-telegram
Restart=always
RestartSec=30
StandardOutput=append:$BOT_DIR/bot.log
StandardError=append:$BOT_DIR/bot.log

[Install]
WantedBy=multi-user.target
SERVICE_EOF

echo
echo "==> Enabling and starting the service..."
systemctl daemon-reload
systemctl enable eth-btc-bot >/dev/null 2>&1
systemctl restart eth-btc-bot

echo "    (waiting 8 seconds for first poll to complete)"
sleep 8

echo
echo "==> Service status:"
systemctl is-active eth-btc-bot && echo "    -> ACTIVE" || echo "    -> NOT ACTIVE"
systemctl is-enabled eth-btc-bot && echo "    -> ENABLED (starts on boot)" || true

echo
echo "==> Last 30 log lines:"
echo "------------------------------------------------------------"
tail -30 "$BOT_DIR/bot.log" 2>/dev/null || echo "(no log yet — first poll still running)"
echo "------------------------------------------------------------"

echo
echo "============================================================"
echo "  DONE. Bot is installed as a systemd service."
echo
echo "  Useful commands (run on the server):"
echo "    tail -f $BOT_DIR/bot.log     # live log"
echo "    systemctl status eth-btc-bot # status"
echo "    systemctl restart eth-btc-bot"
echo "    systemctl stop eth-btc-bot"
echo
echo "  Secrets live in $ENV_FILE_DST (mode 0600)."
echo "  Edit that file and 'systemctl restart eth-btc-bot' to change them."
echo
echo "  The bot will auto-restart on crash and auto-start on reboot."
echo "  You can safely close this ssh session — the bot keeps running."
echo "============================================================"
