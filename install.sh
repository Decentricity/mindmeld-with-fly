#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# On the Decentricity machine, reuse connectome data/cache if present
if [[ ! -e data ]] && [[ -d /home/decentricity/connectome/data ]]; then
  ln -s /home/decentricity/connectome/data data
  echo "Linked data/ -> /home/decentricity/connectome/data"
fi
echo "OK. Run: source .venv/bin/activate && ./scripts/living-caca --graph full"
