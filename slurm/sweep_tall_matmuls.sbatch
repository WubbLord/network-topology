#!/bin/bash
#SBATCH -J tall-mm-sweep
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err

set -euo pipefail

BATCH_SIZES=(1 16 256)
N_VALUES=(4096 16384 65536 262144)
K_VALUES=(2048 4096)
N_LABELS=(4K 16K 64K 256K)
K_LABELS=(2K 4K)
TOPOLOGIES=("Torus 4x4x4" "Mesh 4x4x4" "Ring 64" "Circulant {1,5,17}")

NUM_BATCHES=${#BATCH_SIZES[@]}
NUM_N=${#N_VALUES[@]}
NUM_K=${#K_VALUES[@]}
NUM_TOPOS=${#TOPOLOGIES[@]}
PER_BATCH=$((NUM_N * NUM_K * NUM_TOPOS))
PER_N=$((NUM_K * NUM_TOPOS))

BATCH_IDX=$((SLURM_ARRAY_TASK_ID / PER_BATCH))
REM=$((SLURM_ARRAY_TASK_ID % PER_BATCH))
N_IDX=$((REM / PER_N))
REM=$((REM % PER_N))
K_IDX=$((REM / NUM_TOPOS))
TOPO_IDX=$((REM % NUM_TOPOS))

BATCH_SIZE="${BATCH_SIZES[$BATCH_IDX]}"
N_LABEL="${N_LABELS[$N_IDX]}"
K_LABEL="${K_LABELS[$K_IDX]}"
WORKLOAD="Tall B${BATCH_SIZE} N${N_LABEL}x${K_LABEL}"
TOPOLOGY="${TOPOLOGIES[$TOPO_IDX]}"
MAP_CHIPS="${MAP_CHIPS:-8}"
EVAL_CHIPS="${EVAL_CHIPS:-64}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/logs/slurm-tall-matmuls-${SLURM_ARRAY_JOB_ID}}"
RUN_DIR="${RUN_ROOT}/map${MAP_CHIPS}"

mkdir -p "$RUN_DIR"

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
