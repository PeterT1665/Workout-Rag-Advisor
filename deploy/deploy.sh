#!/usr/bin/env bash
# Build and upload the app + backend to the server. Run from the project root on the laptop:
#   HOST=ubuntu@<server-ip-or-domain> KEY=~/.ssh/<key>.pem deploy/deploy.sh
set -euo pipefail
: "${HOST:?set HOST=ubuntu@<server>}"
: "${KEY:?set KEY=path/to/key.pem}"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new"
cd "$(dirname "$0")/.."

echo "== Building app (personal build: private photos are served only after login)"
(cd web && npm ci --silent && npm run build)

echo "== Uploading"
rsync -az --delete -e "$SSH" web/dist/ "$HOST:/srv/training-agent/web/"
rsync -az --delete -e "$SSH" --exclude .venv --exclude tests --exclude __pycache__ --exclude '*.db' backend/ "$HOST:/srv/training-agent/backend/"
rsync -az -e "$SSH" deploy/ "$HOST:/tmp/training-agent-deploy/"

echo "== Installing backend dependencies and restarting"
$SSH "$HOST" 'set -e
  cd /srv/training-agent/backend
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  if sudo grep -q "^TA_PASSWORD_HASH=.\+" /etc/training-agent.env; then sudo systemctl restart training-agent; else echo "(password not set yet: run training-agent-set-password)"; fi
  sudo systemctl reload caddy'
echo "== Done"
