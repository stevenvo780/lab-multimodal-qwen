#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_PORT="${LAB_WEB_PORT:-4173}"
GPU_PORT="${LAB_GPU_PORT:-18765}"
API_URL="http://127.0.0.1:${GPU_PORT}"
REMOTE_DIR="/datos/workspaces/personal/conferencias/EmbedingMultimodales"
REMOTE_DIR="${LAB_REMOTE_DIR:-$REMOTE_DIR}"
SSH_HOST="${LAB_SSH_HOST:-kratos}"
TUNNEL_PID=""
WEB_PID=""

cleanup() {
  if [[ -n "$WEB_PID" ]]; then
    kill "$WEB_PID" 2>/dev/null || true
    wait "$WEB_PID" 2>/dev/null || true
  fi
  if [[ -n "$TUNNEL_PID" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

port_open() {
  (echo >"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

gpu_ready() {
  curl --silent --show-error --fail --max-time 3 "$API_URL/health" 2>/dev/null | rg -q '"status"[[:space:]]*:[[:space:]]*"ok"'
}

if gpu_ready \
  && curl --silent --show-error --fail --max-time 2 "http://127.0.0.1:${WEB_PORT}/README.md" 2>/dev/null | rg -q '^# Laboratorio visual de embeddings multimodales' \
  && curl --silent --show-error --fail --max-time 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1 \
  && ps -eo args= | rg -F "$ROOT_DIR/laboratorio.py" | rg -q 'streamlit'; then
  echo "Laboratorio SPA ya activo: http://127.0.0.1:$WEB_PORT"
  exit 0
fi

if ! gpu_ready; then
  echo "Iniciando el procesador GPU en $SSH_HOST…"
  printf -v REMOTE_START_COMMAND 'bash %q' "$REMOTE_DIR/start-gpu.sh"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$SSH_HOST" "$REMOTE_START_COMMAND"

  if ! port_open "$GPU_PORT"; then
    echo "Abriendo túnel SSH local :$GPU_PORT → $SSH_HOST:8765…"
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -L "127.0.0.1:${GPU_PORT}:127.0.0.1:8765" "$SSH_HOST" &
    TUNNEL_PID=$!
  fi

  for _ in {1..30}; do
    if gpu_ready; then break; fi
    if [[ -n "$TUNNEL_PID" ]] && ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
      echo "El túnel SSH terminó antes de que respondiera la GPU." >&2
      exit 1
    fi
    sleep 0.5
  done
fi

if ! gpu_ready; then
  echo "La API GPU no responde en $API_URL. Comprueba ssh $SSH_HOST y tmux embedding-gpu." >&2
  exit 1
fi

if port_open "$WEB_PORT"; then
  if ! curl --silent --show-error --fail --max-time 2 "http://127.0.0.1:${WEB_PORT}/README.md" | rg -q '^# Laboratorio visual de embeddings multimodales'; then
    echo "El puerto del catálogo $WEB_PORT está ocupado por otro servicio." >&2
    exit 1
  fi
else
  python3 -m http.server "$WEB_PORT" --bind 127.0.0.1 --directory "$ROOT_DIR" >/dev/null 2>&1 &
  WEB_PID=$!
fi

if port_open 8501; then
  echo "El puerto 8501 de Streamlit ya está ocupado. Cierra ese servicio antes de continuar." >&2
  exit 1
fi

if [[ ! -x "$ROOT_DIR/.venv/bin/streamlit" ]]; then
  echo "Preparando dependencias locales con uv…"
  (cd "$ROOT_DIR" && uv sync --python 3.12)
fi

echo "Laboratorio SPA:   http://127.0.0.1:$WEB_PORT"
echo "Motor local:       http://127.0.0.1:8501"
echo "GPU por SSH:       $API_URL"
echo "Pulsa Ctrl+C para cerrar la sesión local."
QWEN_GPU_API_URL="$API_URL" "$ROOT_DIR/.venv/bin/streamlit" run "$ROOT_DIR/laboratorio.py" \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false
