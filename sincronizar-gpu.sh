#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="${LAB_SSH_HOST:-kratos}"
REMOTE_PATH="${LAB_REMOTE_DIR:-/datos/workspaces/personal/conferencias/EmbedingMultimodales}"
REMOTE_DIR="$SSH_HOST:$REMOTE_PATH/"
FILES=(gpu_server.py embed.py start-gpu.sh prueba.py requirements.txt README.md)
ACTION="${1:-check}"

case "$ACTION" in
  check)
    echo "Diferencias que se enviarían desde la copia local:"
    printf '%s\n' "${FILES[@]}" | rsync -anvc --no-times --itemize-changes --files-from=- \
      "$ROOT_DIR/gpu_remote/" "$REMOTE_DIR"
    ;;
  push)
    echo "Sincronizando código local hacia $SSH_HOST (con copia de seguridad de archivos reemplazados)…"
    printf '%s\n' "${FILES[@]}" | rsync -avc --backup \
      --suffix=".bak-$(date +%Y%m%d-%H%M%S)" --files-from=- \
      "$ROOT_DIR/gpu_remote/" "$REMOTE_DIR"
    echo "Si cambiaste código Python, reinicia el proceso en tmux embedding-gpu para aplicarlo."
    ;;
  pull)
    echo "Trayendo código desde $SSH_HOST (con copia de seguridad local de archivos reemplazados)…"
    printf '%s\n' "${FILES[@]}" | rsync -avc --backup \
      --suffix=".bak-$(date +%Y%m%d-%H%M%S)" --files-from=- \
      "$REMOTE_DIR" "$ROOT_DIR/gpu_remote/"
    ;;
  *)
    echo "Uso: $0 [check|push|pull]" >&2
    exit 2
    ;;
esac
