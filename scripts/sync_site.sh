#!/usr/bin/env bash
# Copy site/ to the folder the user deploys to Cloudflare Pages from.
# Backs the target up first; tests and dev files are not deployed.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)/site/"
DEST="${1:-$HOME/Desktop/CappingAlpha}"
[ -d "$DEST" ] && cp -R "$DEST" "${DEST}-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"
rsync -a --delete --exclude tests/ --exclude .DS_Store "$SRC" "$DEST/"
echo "synced site/ -> $DEST (backup kept next to it)"
