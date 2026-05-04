#!/bin/bash
#SBATCH -J plot-moe-af
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o plot-moe-af-%j.out
#SBATCH -e plot-moe-af-%j.err

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
python3 plot_moe_expert_ffn_results.py "$RUN_DIR" \
    --mirror-dir "${FIGURE_DIR:-${PROJECT_DIR}/figures/current_moe_expert_ffn}"
