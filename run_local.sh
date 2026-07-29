#!/bin/bash
# Load .env and start the race UI on http://localhost:8130 (see README).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "copy .env.example to .env and set DO_KEY"; exit 1; }
set -a; . ./.env; set +a
export DATASET="${DATASET:-$(pwd)/evalset.csv}"
exec python3 race_server.py
