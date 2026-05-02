#!/bin/bash
#SBATCH -J small-bmm-sweep
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err

set -euo pipefail

BATCH_SIZES=(1 16 256)
BASE_WORKLOADS=(
    "Small 2Kx8K"
    "Small 4Kx16K"
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
if [ "$BATCH_SIZE" = "1" ]; then
    WORKLOAD="$BASE_WORKLOAD"
else
    WORKLOAD="Batched B${BATCH_SIZE} ${BASE_WORKLOAD}"
fi
TOPOLOGY="${TOPOLOGIES[$TOPO_IDX]}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/logs/slurm-small-batched-matmuls-${SLURM_ARRAY_JOB_ID}}"
MAP_CHIPS="${MAP_CHIPS:-64}"
EVAL_CHIPS="${EVAL_CHIPS:-64}"

echo "========================================"
echo "Task $SLURM_ARRAY_TASK_ID: workload='$WORKLOAD' topology='$TOPOLOGY'"
echo "MAP_CHIPS=$MAP_CHIPS EVAL_CHIPS=$EVAL_CHIPS"
echo "Run dir: $RUN_DIR"
echo "Node: $(hostname), CPUs: $SLURM_CPUS_PER_TASK, Mem: ${SLURM_MEM_PER_NODE:-unknown}"
echo "Start: $(date)"
echo "========================================"

module load miniforge/25.11.0-0
source "${PROJECT_DIR}/.venv/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ACCELFORGE_N_JOBS="${ACCELFORGE_N_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

if [ -z "${ACCELFORGE_ROOT:-}" ]; then
    export ACCELFORGE_ROOT="$(python3 -c 'import accelforge; from pathlib import Path; print(Path(accelforge.__file__).parent.parent)')"
fi
echo "ACCELFORGE_ROOT=$ACCELFORGE_ROOT"

cd "$PROJECT_DIR"
MAP_CHIPS="$MAP_CHIPS" EVAL_CHIPS="$EVAL_CHIPS" \
    python3 sweep_matmuls.py --topology "$TOPOLOGY" --workload "$WORKLOAD" --run-dir "$RUN_DIR"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
