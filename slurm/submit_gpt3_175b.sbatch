#!/bin/bash
#SBATCH -J submit-gpt3-175b
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:05:00
#SBATCH -o submit-gpt3-175b-%j.out
#SBATCH -e submit-gpt3-175b-%j.err

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
RUN_DIR="${PROJECT_DIR}/logs/slurm-gpt3-175b-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR" "$RUN_DIR"

cd "$PROJECT_DIR"

ARRAY_JOB=$(sbatch --parsable \
    --chdir="$LOGDIR" \
    --array="0-3" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_DIR}" \
    slurm/sweep_gpt3_175b.sbatch)

AGG_JOB=$(sbatch --parsable \
    --chdir="$LOGDIR" \
    --dependency="afterok:${ARRAY_JOB}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_DIR}" \
    slurm/aggregate.sbatch)

echo "GPT3 175B topology array submitted: $ARRAY_JOB"
echo "Aggregation job submitted: $AGG_JOB"
echo "Results will be in: ${RUN_DIR}/results.json"
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOGDIR}/gpt3-175b-sweep-${ARRAY_JOB}-<task>.out"
