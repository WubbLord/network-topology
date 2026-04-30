#!/bin/bash
#SBATCH -J moe-sweep
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o moe-sweep-%j.out
#SBATCH -e moe-sweep-%j.err

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
    if [ -f "$candidate/sweep_moe.py" ]; then
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
cd "$PROJECT_DIR"

module load miniforge/25.11.0-0

if [ ! -x ".venv/bin/python" ]; then
    python3 -m venv .venv
    .venv/bin/python -m pip install -e . --quiet
fi

source .venv/bin/activate

RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/logs/slurm-moe-${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"

python -u sweep_moe.py --run-dir "$RUN_DIR" "$@"
