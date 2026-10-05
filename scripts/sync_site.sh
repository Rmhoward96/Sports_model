#!/usr/bin/env bash
# Copy site/ to the folder the user deploys to Cloudflare Pages from.
# Backs the target up first; tests and dev files are not deployed.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)/site/"
DEST="${1:-$HOME/Desktop/CappingAlpha}"
# Strip trailing slashes so the backup lands NEXT TO the target ("CappingAlpha-backup-..."), not inside it.
while [ "${#DEST}" -gt 1 ] && [ "${DEST%/}" != "$DEST" ]; do DEST="${DEST%/}"; done
[ -d "$DEST" ] && cp -R "$DEST" "${DEST}-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"
# "/tests/" is anchored to the top of site/: only site/tests is skipped, never a nested folder that happens to be named tests.
rsync -a --delete --exclude /tests/ --exclude .DS_Store "$SRC" "$DEST/"
echo "synced site/ -> $DEST (backup kept next to it)"
