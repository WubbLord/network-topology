#!/bin/bash
#SBATCH -J submit-small-sweep
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:05:00
#SBATCH -o submit-small-%j.out
#SBATCH -e submit-small-%j.err

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
    if [ -x "$candidate/slurm/submit.sh" ]; then
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
mkdir -p "${PROJECT_DIR}/slurm/logs"

cd "$PROJECT_DIR"
bash slurm/submit.sh --small
