#!/bin/bash
# Rsync processed .npy grids from local HDD cache to Bede /nobackup.
#
# Usage:
#   ./scripts/push_to_bede.sh [--modality supermag|tec|omni|all]
#
# Prerequisites:
#   - SSH key already added for bede.dur.ac.uk
#   - Bede username set in BEDE_USER env var (or hardcoded below)
#   - Directories on Bede created by first run (rsync --mkpath handles it)

set -euo pipefail

# ── edit these ───────────────────────────────────────────────────────────────
BEDE_USER="${BEDE_USER:-smander3}"
BEDE_HOST="bede.dur.ac.uk"
BEDE_PROJECT="bdlan12"
BEDE_DEST="/nobackup/projects/${BEDE_PROJECT}/${BEDE_USER}/iome_data"

# Local cache roots
LOCAL_ROOTS=(
    "/data5/iome_cache"
)
# ─────────────────────────────────────────────────────────────────────────────

MODALITY="${1:-all}"

# Parse optional --modality flag
while [[ $# -gt 0 ]]; do
    case $1 in
        --modality) MODALITY="$2"; shift 2 ;;
        *) shift ;;
    esac
done

case "$MODALITY" in
    all)     MODS=("superdarn" "supermag" "tec" "omni" "splits") ;;
    *)       MODS=("$MODALITY") ;;
esac

echo "[push_to_bede] target: ${BEDE_USER}@${BEDE_HOST}:${BEDE_DEST}"
echo "[push_to_bede] modalities: ${MODS[*]}"

for MOD in "${MODS[@]}"; do
    for LOCAL_ROOT in "${LOCAL_ROOTS[@]}"; do
        SRC="${LOCAL_ROOT}/${MOD}/"
        if [[ ! -d "$SRC" ]]; then
            echo "  skip ${SRC} (not found)"
            continue
        fi

        N_FILES=$(find "$SRC" -name "*.npy" | wc -l)
        if [[ "$N_FILES" -eq 0 ]]; then
            echo "  skip ${SRC} (empty)"
            continue
        fi

        echo ""
        echo "  rsync ${SRC} → ${BEDE_USER}@${BEDE_HOST}:${BEDE_DEST}/${MOD}/"
        echo "  (${N_FILES} .npy files)"

        rsync \
            --archive \
            --compress \
            --progress \
            --partial \
            --mkpath \
            --exclude "*.tmp" \
            --include "*.npy" \
            --include "*.json" \
            --exclude "*" \
            "${SRC}" \
            "${BEDE_USER}@${BEDE_HOST}:${BEDE_DEST}/${MOD}/"
    done
done

echo ""
echo "[push_to_bede] done."
echo "On Bede, pass these paths to the training scripts:"
for MOD in "${MODS[@]}"; do
    echo "  --cache_${MOD}  ${BEDE_DEST}/${MOD}"
done
