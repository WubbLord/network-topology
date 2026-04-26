#!/bin/bash
#SBATCH -J topo-6d                       # Job name
#SBATCH -p mit_normal                    # MIT shared CPU partition (12h max)
#SBATCH -c 8                             # 8 CPUs per task (mapper is CPU-bound)
#SBATCH --mem=64G                        # 64 GB memory (6D Hypercube mapper needs more)
#SBATCH -t 06:00:00                      # 6 hours per task (192 pmapping candidates)
# PROJECT_DIR is passed via --export from submit.sh
#SBATCH -o %x-%A-%a.out                  # stdout per task (in working dir)
#SBATCH -e %x-%A-%a.err                  # stderr per task

# ---- 6D Hypercube only, smaller workloads that fit in memory ----
WORKLOADS=(
    "Small 2Kx8K"
    "Small 4Kx16K"
    "Medium 8Kx32K"
    "Medium 16Kx32K"
    "Wide 8Kx256K"
    "VeryWide 2Kx256K"
    "Rect 64Kx128K"
    "Square 128Kx128K"
    "VeryTall 256Kx8K"
)
TOPOLOGY="6D Hypercube"

WORKLOAD="${WORKLOADS[$SLURM_ARRAY_TASK_ID]}"

# Shared run directory. submit.sh exports RUN_DIR so the 6D artifacts land with
# the rest of the sweep; direct sbatch use falls back to this array's run dir.
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/logs/slurm-${SLURM_ARRAY_JOB_ID}}"

echo "========================================"
echo "Task $SLURM_ARRAY_TASK_ID: workload='$WORKLOAD' topology='$TOPOLOGY'"
echo "Run dir: $RUN_DIR"
echo "Node: $(hostname), CPUs: $SLURM_CPUS_PER_TASK, Mem: $SLURM_MEM_PER_NODE"
echo "Start: $(date)"
echo "========================================"

# ---- Environment ----
module load miniforge/25.11.0-0
source "${PROJECT_DIR}/.venv/bin/activate"

if [ -z "$ACCELFORGE_ROOT" ]; then
    export ACCELFORGE_ROOT="$(python3 -c 'import accelforge; from pathlib import Path; print(Path(accelforge.__file__).parent.parent)')"
fi
echo "ACCELFORGE_ROOT=$ACCELFORGE_ROOT"

# ---- Run ----
cd "$PROJECT_DIR"
python3 sweep_matmuls.py --topology "$TOPOLOGY" --workload "$WORKLOAD" --run-dir "$RUN_DIR"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
