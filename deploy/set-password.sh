#!/usr/bin/env bash
# Set or change the app password (asks for it; nothing is echoed or logged).
set -euo pipefail
HASH=$(sudo -u training /srv/training-agent/backend/.venv/bin/python -c "
import getpass, sys
sys.path.insert(0, '/srv/training-agent/backend')
from app.auth import hash_password
pw = getpass.getpass('New app password: ')
if len(pw) < 8: sys.exit('Use at least 8 characters.')
if pw != getpass.getpass('Repeat: '): sys.exit('Passwords did not match.')
print(hash_password(pw))")
sudo sed -i "s|^TA_PASSWORD_HASH=.*|TA_PASSWORD_HASH=${HASH}|" /etc/training-agent.env
sudo systemctl restart training-agent
echo "Password set. Log in at your app address."
