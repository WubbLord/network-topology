#!/bin/bash
#SBATCH -J plot-tall-mm
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o plot-tall-mm-%j.out
#SBATCH -e plot-tall-mm-%j.err

set -euo pipefail

if [ -z "${PROJECT_DIR:-}" ]; then
    echo "ERROR: PROJECT_DIR is not set." >&2
    exit 1
fi
if [ -z "${RUN_ROOT:-}" ]; then
    echo "ERROR: RUN_ROOT is not set." >&2
    exit 1
fi

module load miniforge/25.11.0-0
source "${PROJECT_DIR}/.venv/bin/activate"

cd "$PROJECT_DIR"
python3 plot_tall_matmul_results.py "$RUN_ROOT" \
    --mirror-dir "${PROJECT_DIR}/figures/current_tall_matmuls"
