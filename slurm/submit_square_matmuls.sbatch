#!/bin/bash
#SBATCH -J submit-square-mm
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:05:00
#SBATCH -o submit-square-mm-%j.out
#SBATCH -e submit-square-mm-%j.err

set -euo pipefail

resolve_project_dir() {
    if [ -n "${PROJECT_DIR:-}" ]; then
        cd "$PROJECT_DIR"
        pwd
        return
    fi

    local script_path="${BASH_SOURCE[0]}"
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
        local command_path
        command_path="$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)"
        if [ -n "$command_path" ] && [ "$command_path" != "(null)" ]; then
            script_path="$command_path"
        fi
    fi

    if [[ "$script_path" != /* ]] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        script_path="${SLURM_SUBMIT_DIR}/${script_path}"
    fi

    local script_dir
    script_dir="$(cd "$(dirname "$script_path")" && pwd)"
    local candidate
    candidate="$(dirname "$script_dir")"
    if [ -f "$candidate/sweep_matmuls.py" ] && [ -d "$candidate/slurm" ]; then
        cd "$candidate"
        pwd
        return
    fi

    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        cd "$SLURM_SUBMIT_DIR"
        if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
            cd "$git_root"
            pwd
            return
        fi
    fi

    echo "ERROR: unable to determine project root. Submit from inside the repo or export PROJECT_DIR." >&2
    exit 1
}

PROJECT_DIR="$(resolve_project_dir)"
LOGDIR="${PROJECT_DIR}/slurm/logs"
RUN_DIR="${PROJECT_DIR}/logs/slurm-square-matmuls-${SLURM_JOB_ID}"
FIGURE_DIR="${PROJECT_DIR}/figures/current_square_matmuls"
mkdir -p "$LOGDIR" "$RUN_DIR" "$FIGURE_DIR"

cd "$PROJECT_DIR"

# 4 N values x 2 T values x 4 topologies = 32 tasks.
ARRAY_JOB=$(sbatch --parsable \
    --partition=mit_normal \
    --time=12:00:00 \
    --cpus-per-task=8 \
    --mem=32G \
    --chdir="$LOGDIR" \
    --array="0-31%4" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_DIR}",MAP_CHIPS=64,EVAL_CHIPS=64 \
    slurm/sweep_square_matmuls.sbatch)

AGG_JOB=$(sbatch --parsable \
    --partition=mit_normal \
    --time=00:05:00 \
    --cpus-per-task=1 \
    --mem=4G \
    --chdir="$LOGDIR" \
    --dependency="afterok:${ARRAY_JOB}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_DIR}" \
    slurm/aggregate.sbatch)

PLOT_JOB=$(sbatch --parsable \
    --partition=mit_normal \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem=8G \
    --chdir="$LOGDIR" \
    --dependency="afterok:${AGG_JOB}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_DIR}",FIGURE_DIR="${FIGURE_DIR}" \
    slurm/plot_square_matmuls.sbatch)

echo "Square matmul array submitted: $ARRAY_JOB"
echo "Aggregation job submitted: $AGG_JOB"
echo "Figure job submitted: $PLOT_JOB"
echo "MAP_CHIPS=64 EVAL_CHIPS=64"
echo "Results will be in: ${RUN_DIR}/results.json"
echo "Figures will be mirrored to: ${FIGURE_DIR}"
echo "Monitor: squeue -j ${ARRAY_JOB},${AGG_JOB},${PLOT_JOB}"
echo "Logs:    ${LOGDIR}/square-mm-sweep-${ARRAY_JOB}-<task>.out"
