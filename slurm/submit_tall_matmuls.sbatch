#!/bin/bash
#SBATCH -J submit-tall-mm
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:05:00
#SBATCH -o submit-tall-mm-%j.out
#SBATCH -e submit-tall-mm-%j.err

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
RUN_ROOT="${PROJECT_DIR}/logs/slurm-tall-matmuls-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR" "$RUN_ROOT/map8" "$RUN_ROOT/map64"

cd "$PROJECT_DIR"

# 3 batch settings x 4 N values x 2 K values x 4 topologies = 96 tasks per MAP_CHIPS setting.
ARRAY_RANGE="0-95"

ARRAY_MAP8=$(sbatch --parsable \
    --partition=mit_normal \
    --time=06:00:00 \
    --cpus-per-task=8 \
    --mem=64G \
    --chdir="$LOGDIR" \
    --array="${ARRAY_RANGE}%2" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_ROOT="${RUN_ROOT}",MAP_CHIPS=8,EVAL_CHIPS=64 \
    slurm/sweep_tall_matmuls.sbatch)

ARRAY_MAP64=$(sbatch --parsable \
    --partition=mit_normal \
    --time=12:00:00 \
    --cpus-per-task=16 \
    --mem=128G \
    --chdir="$LOGDIR" \
    --array="${ARRAY_RANGE}%1" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_ROOT="${RUN_ROOT}",MAP_CHIPS=64,EVAL_CHIPS=64 \
    slurm/sweep_tall_matmuls.sbatch)

AGG_MAP8=$(sbatch --parsable \
    --partition=mit_normal \
    --time=00:05:00 \
    --cpus-per-task=1 \
    --mem=4G \
    --chdir="$LOGDIR" \
    --dependency="afterok:${ARRAY_MAP8}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_ROOT}/map8" \
    slurm/aggregate.sbatch)

AGG_MAP64=$(sbatch --parsable \
    --partition=mit_normal \
    --time=00:05:00 \
    --cpus-per-task=1 \
    --mem=4G \
    --chdir="$LOGDIR" \
    --dependency="afterok:${ARRAY_MAP64}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_ROOT}/map64" \
    slurm/aggregate.sbatch)

PLOT_JOB=$(sbatch --parsable \
    --partition=mit_normal \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem=8G \
    --chdir="$LOGDIR" \
    --dependency="afterok:${AGG_MAP8}:${AGG_MAP64}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_ROOT="${RUN_ROOT}" \
    slurm/plot_tall_matmuls.sbatch)

echo "Tall matmul MAP_CHIPS=8 array submitted:  $ARRAY_MAP8"
echo "Tall matmul MAP_CHIPS=64 array submitted: $ARRAY_MAP64"
echo "MAP8 aggregation job:  $AGG_MAP8"
echo "MAP64 aggregation job: $AGG_MAP64"
echo "Figure/summary job:    $PLOT_JOB"
echo "Results root: ${RUN_ROOT}"
echo "Expected aggregates:"
echo "  ${RUN_ROOT}/map8/results.json"
echo "  ${RUN_ROOT}/map64/results.json"
echo "Figures:"
echo "  ${RUN_ROOT}/figures/"
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOGDIR}/tall-mm-sweep-<array>-<task>.out"
