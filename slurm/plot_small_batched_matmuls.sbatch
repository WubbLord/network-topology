#!/bin/bash
#SBATCH -J plot-small-bmm
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o plot-small-bmm-%j.out
#SBATCH -e plot-small-bmm-%j.err

set -euo pipefail

if [ -z "${PROJECT_DIR:-}" ]; then
    echo "ERROR: PROJECT_DIR is not set." >&2
    exit 1
fi
if [ -z "${RUN_DIR:-}" ]; then
    echo "ERROR: RUN_DIR is not set." >&2
    exit 1
fi

module load miniforge/25.11.0-0
source "${PROJECT_DIR}/.venv/bin/activate"

cd "$PROJECT_DIR"
python3 plot_small_batched_matmul_results.py "$RUN_DIR" \
    --mirror-dir "${FIGURE_DIR:-${PROJECT_DIR}/figures/current_small_batched_matmuls}"
