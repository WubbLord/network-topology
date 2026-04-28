#!/bin/bash
#SBATCH -J batched-mm-sweep
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err

set -euo pipefail

BATCH_SIZES=(256 512 1024)
BASE_WORKLOADS=(
    "Small 2Kx8K"
    "Small 4Kx16K"
    "Medium 8Kx32K"
    "Medium 16Kx32K"
    "Wide 8Kx256K"
    "VeryWide 2Kx256K"
    "Rect 64Kx128K"
)
TOPOLOGIES=("Torus 4x4x4" "Mesh 4x4x4" "Ring 64" "Circulant {1,5,17}")

NUM_BASE_WORKLOADS=${#BASE_WORKLOADS[@]}
NUM_TOPOS=${#TOPOLOGIES[@]}
PER_BATCH=$((NUM_BASE_WORKLOADS * NUM_TOPOS))
BATCH_IDX=$((SLURM_ARRAY_TASK_ID / PER_BATCH))
REM=$((SLURM_ARRAY_TASK_ID % PER_BATCH))
WORKLOAD_IDX=$((REM / NUM_TOPOS))
TOPO_IDX=$((REM % NUM_TOPOS))

BATCH_SIZE="${BATCH_SIZES[$BATCH_IDX]}"
BASE_WORKLOAD="${BASE_WORKLOADS[$WORKLOAD_IDX]}"
WORKLOAD="Batched B${BATCH_SIZE} ${BASE_WORKLOAD}"
TOPOLOGY="${TOPOLOGIES[$TOPO_IDX]}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/logs/slurm-batched-matmuls-${SLURM_ARRAY_JOB_ID}}"

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
