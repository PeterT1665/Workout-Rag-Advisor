#!/usr/bin/env bash
# Build and upload the app + backend to the server. Run from the project root on the laptop:
#   HOST=ubuntu@<server-ip-or-domain> KEY=~/.ssh/<key>.pem deploy/deploy.sh
# Optional public demo (sample data, no login) on a second name that already points at the server:
#   DOMAIN=<main-domain> DEMO_DOMAIN=<demo-domain> HOST=... KEY=... deploy/deploy.sh
set -euo pipefail
: "${HOST:?set HOST=ubuntu@<server>}"
: "${KEY:?set KEY=path/to/key.pem}"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new"
cd "$(dirname "$0")/.."

echo "== Building app (personal build: private photos are served only after login)"
(cd web && npm ci --silent && npm run build)

if [ -n "${DEMO_DOMAIN:-}" ]; then
  : "${DOMAIN:?set DOMAIN=<main domain> together with DEMO_DOMAIN}"
  echo "== Building public demo"
  (cd web && npm run build:demo)
fi

echo "== Uploading"
rsync -az --delete -e "$SSH" web/dist/ "$HOST:/srv/training-agent/web/"
rsync -az --delete -e "$SSH" --exclude .venv --exclude tests --exclude __pycache__ --exclude '*.db' backend/ "$HOST:/srv/training-agent/backend/"
rsync -az -e "$SSH" deploy/ "$HOST:/tmp/training-agent-deploy/"

if [ -n "${DEMO_DOMAIN:-}" ]; then
  echo "== Public demo"
  $SSH "$HOST" 'sudo mkdir -p /srv/training-agent/demo && sudo chown ubuntu:ubuntu /srv/training-agent/demo'
  rsync -az --delete -e "$SSH" web/dist-demo/ "$HOST:/srv/training-agent/demo/"
  # Only ask for a certificate once the name really points at this server (avoids Let's Encrypt failure limits)
  $SSH "$HOST" "set -e
    MYIP=\$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
    DNSIP=\$(getent hosts '$DEMO_DOMAIN' | awk '{print \$1}' | head -1)
    if [ \"\$MYIP\" != \"\$DNSIP\" ]; then echo \"!! $DEMO_DOMAIN points to '\$DNSIP', not this server (\$MYIP). Skipping HTTPS setup for the demo; point DuckDNS at \$MYIP and run again.\"; exit 0; fi
    cd /tmp/training-agent-deploy
    sudo bash -c \"sed 's/__DOMAIN__/$DOMAIN/' Caddyfile.template > /etc/caddy/Caddyfile && sed 's/__DEMO_DOMAIN__/$DEMO_DOMAIN/' Caddyfile.demo.template >> /etc/caddy/Caddyfile && caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null && systemctl reload caddy\"
    echo 'demo is live at https://$DEMO_DOMAIN'"
fi

echo "== Installing backend dependencies and restarting"
$SSH "$HOST" 'set -e
  cd /srv/training-agent/backend
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  if sudo grep -q "^TA_PASSWORD_HASH=.\+" /etc/training-agent.env; then sudo systemctl restart training-agent; else echo "(password not set yet: run training-agent-set-password)"; fi
  sudo systemctl reload caddy'
echo "== Done"
