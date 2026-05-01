#!/bin/bash
#SBATCH -J moe-af-sweep
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err

set -euo pipefail

TOKENS_PER_EXPERT=(64 256 1024 4096)
TOPOLOGIES=("Torus 4x4x4" "Mesh 4x4x4" "Ring 64" "Circulant {1,5,17}")

NUM_TOPOS=${#TOPOLOGIES[@]}
WORKLOAD_IDX=$((SLURM_ARRAY_TASK_ID / NUM_TOPOS))
TOPO_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TOPOS))

TOKENS="${TOKENS_PER_EXPERT[$WORKLOAD_IDX]}"
WORKLOAD="MoE ExpertFFN E16 T${TOKENS}"
TOPOLOGY="${TOPOLOGIES[$TOPO_IDX]}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/logs/slurm-moe-accelforge-${SLURM_ARRAY_JOB_ID}}"

echo "========================================"
echo "Task $SLURM_ARRAY_TASK_ID: workload='$WORKLOAD' topology='$TOPOLOGY'"
echo "Run dir: $RUN_DIR"
echo "Node: $(hostname), CPUs: $SLURM_CPUS_PER_TASK, Mem: $SLURM_MEM_PER_NODE"
echo "Start: $(date)"
echo "========================================"

module load miniforge/25.11.0-0
source "${PROJECT_DIR}/.venv/bin/activate"

if [ -z "${ACCELFORGE_ROOT:-}" ]; then
    export ACCELFORGE_ROOT="$(python3 -c 'import accelforge; from pathlib import Path; print(Path(accelforge.__file__).parent.parent)')"
fi
echo "ACCELFORGE_ROOT=$ACCELFORGE_ROOT"

cd "$PROJECT_DIR"
python3 sweep_matmuls.py --topology "$TOPOLOGY" --workload "$WORKLOAD" --run-dir "$RUN_DIR"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
