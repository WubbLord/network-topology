#!/bin/bash
#SBATCH -J update-tall-figs
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o update-tall-figs-%j.out
#SBATCH -e update-tall-figs-%j.err

set -euo pipefail

if [ -z "${PROJECT_DIR:-}" ]; then
    echo "ERROR: PROJECT_DIR is not set." >&2
    exit 1
fi
if [ -z "${RUN_ROOT:-}" ]; then
    echo "ERROR: RUN_ROOT is not set." >&2
    exit 1
fi

INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MAX_UPDATES="${MAX_UPDATES:-144}"
MIRROR_DIR="${MIRROR_DIR:-${PROJECT_DIR}/figures/current_tall_matmuls}"

module load miniforge/25.11.0-0
source "${PROJECT_DIR}/.venv/bin/activate"

cd "$PROJECT_DIR"

for ((i=1; i<=MAX_UPDATES; i++)); do
    echo "[$(date)] Updating tall matmul figures (${i}/${MAX_UPDATES})"
    python3 plot_tall_matmul_results.py "$RUN_ROOT" --mirror-dir "$MIRROR_DIR"

    if [ -f "$RUN_ROOT/map8/results.json" ] && [ -f "$RUN_ROOT/map64/results.json" ]; then
        echo "[$(date)] Both aggregates exist; final update complete."
        exit 0
    fi

    sleep "$INTERVAL_SECONDS"
done

echo "[$(date)] Reached MAX_UPDATES before both aggregates appeared."
