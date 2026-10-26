#!/usr/bin/env bash
# One-time server setup (Ubuntu 24.04 on EC2). Run on the server:  sudo bash bootstrap.sh your-name.duckdns.org
set -euo pipefail
DOMAIN="${1:?usage: sudo bash bootstrap.sh <domain>}"

apt-get update -y
apt-get install -y python3-venv sqlite3 debian-keyring debian-archive-keyring apt-transport-https curl gnupg
# Caddy from its official repository
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -y && apt-get install -y caddy
fi

id training >/dev/null 2>&1 || useradd --system --home /srv/training-agent --shell /usr/sbin/nologin training
mkdir -p /srv/training-agent/{web,backend,data/backups}
chown -R training:training /srv/training-agent/data
chown -R ubuntu:ubuntu /srv/training-agent/web /srv/training-agent/backend   # deploy.sh uploads as ubuntu

# Secrets live only on the server. The password hash is added later by set-password.sh.
if [ ! -f /etc/training-agent.env ]; then
  umask 077
  cat > /etc/training-agent.env <<ENV
TA_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
TA_DB=/srv/training-agent/data/training.db
TA_SECURE_COOKIE=1
TA_PASSWORD_HASH=
ENV
  chown root:training /etc/training-agent.env && chmod 640 /etc/training-agent.env
fi

sed "s/__DOMAIN__/${DOMAIN}/" "$(dirname "$0")/Caddyfile.template" > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
install -m 644 "$(dirname "$0")/training-agent.service" /etc/systemd/system/training-agent.service
install -m 755 "$(dirname "$0")/set-password.sh" /usr/local/bin/training-agent-set-password

# Daily database backup, keep 14 days
cat > /etc/cron.daily/training-agent-backup <<'CRON'
#!/bin/sh
sqlite3 /srv/training-agent/data/training.db ".backup '/srv/training-agent/data/backups/training-$(date +%F).db'" 2>/dev/null || true
find /srv/training-agent/data/backups -name 'training-*.db' -mtime +14 -delete
CRON
chmod 755 /etc/cron.daily/training-agent-backup

systemctl daemon-reload
systemctl enable caddy training-agent
systemctl reload caddy || systemctl restart caddy
echo "Bootstrap done for ${DOMAIN}. Next: deploy.sh from the laptop, then set the password."
