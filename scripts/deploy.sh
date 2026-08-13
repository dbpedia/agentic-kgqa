#!/usr/bin/env bash
set -euo pipefail

HOST="root@research.liberai.org"
REMOTE_DIR="/opt/agentic-kgqa"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Syncing project to ${HOST}:${REMOTE_DIR} ..."
rsync -avzL --delete \
  --exclude '.git' \
  --exclude '.claude' \
  --exclude '.claude-flow' \
  --exclude '.venv' \
  --exclude '.swarm' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude '.DS_Store' \
  --exclude '.mcp.json' \
  --exclude 'Pipfile.lock' \
  --exclude '*.ipynb' \
  --exclude 'data' \
  "${PROJECT_DIR}/" "${HOST}:${REMOTE_DIR}/"

echo "==> Installing dependencies and starting service on remote ..."
ssh "${HOST}" bash -s <<'REMOTE'
set -euo pipefail
cd /opt/agentic-kgqa

# Create venv if missing
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

# Install deps into venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet openai gensim rdflib python-dotenv redis fastapi uvicorn pyyaml pandas

# Install systemd unit
cp scripts/agentic-kgqa.service /etc/systemd/system/agentic-kgqa.service
systemctl daemon-reload
systemctl enable agentic-kgqa
systemctl restart agentic-kgqa

sleep 1
echo "==> Service status:"
systemctl --no-pager status agentic-kgqa || true
echo "==> Deployment complete."
REMOTE
