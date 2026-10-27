# Deploying Sport Personal Coach (EC2 + Caddy + DuckDNS)

One small server runs everything: Caddy (HTTPS, serves the app, guards private photos) and the FastAPI backend (login + sync, SQLite).

## You do this once (in your browser)

1. **AWS EC2**: launch an instance
   - Ubuntu Server 24.04 LTS, type `t3.micro` (or `t2.micro`), free-tier eligible.
   - Create a key pair (`.pem`), save it on the laptop (e.g. `~/.ssh/training-agent.pem`, then `chmod 400` it).
   - Security group inbound: SSH 22 (My IP only), HTTP 80 (anywhere), HTTPS 443 (anywhere).
   - Allocate an **Elastic IP** and associate it with the instance (so the address never changes).
2. **DuckDNS** (duckdns.org, sign in with GitHub/Google): create a subdomain, e.g. `peter-training`, and set its IP to the Elastic IP.

## Then (Claude can run these for you)

```bash
# 1. copy setup files and run the one-time bootstrap
scp -i KEY deploy/* ubuntu@IP:/tmp/ta/      # (after: ssh ... mkdir -p /tmp/ta)
ssh -i KEY ubuntu@IP 'sudo bash /tmp/ta/bootstrap.sh peter-training.duckdns.org'

# 2. build and upload
HOST=ubuntu@IP KEY=KEY deploy/deploy.sh

# 3. set your password (YOU type it; it is never shown or logged)
ssh -t -i KEY ubuntu@IP training-agent-set-password
```

Open `https://peter-training.duckdns.org`, log in, then on the iPhone: Safari → Share → Add to Home Screen.

## Updating later

`HOST=ubuntu@IP KEY=KEY deploy/deploy.sh`. Data and the password are untouched; the database is backed up daily (14 days kept) in `/srv/training-agent/data/backups`.
