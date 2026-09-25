#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if curl -fsS --max-time 2 http://127.0.0.1:8765/health >/dev/null 2>&1; then
  echo "GPU API ya activa en 127.0.0.1:8765"
  exit 0
fi
if tmux has-session -t embedding-gpu 2>/dev/null; then
  echo "La sesión embedding-gpu existe, pero /health no responde. Revísala con: tmux attach -t embedding-gpu" >&2
  exit 1
fi
tmux new-session -d -s embedding-gpu -n api \
  -c "$ROOT_DIR" \
  "bash -lc 'CUDA_VISIBLE_DEVICES=0 .venv/bin/python gpu_server.py; exec bash'"
for attempt in $(seq 1 30); do
  if curl -fsS --max-time 2 http://127.0.0.1:8765/health >/dev/null 2>&1; then
    echo "GPU API lista en 127.0.0.1:8765 (tmux embedding-gpu)"
    exit 0
  fi
  sleep 1
done
echo "La GPU API no respondió; revisa: tmux attach -t embedding-gpu" >&2
exit 1
