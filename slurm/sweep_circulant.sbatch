#!/bin/bash
#SBATCH -J topo-circ                     # Job name
#SBATCH -p mit_normal                    # MIT shared CPU partition (12h max)
#SBATCH -c 8                             # 8 CPUs per task (mapper is CPU-bound)
#SBATCH --mem=64G                        # 64 GB memory (non-standard topology may need more)
#SBATCH -t 06:00:00                      # 6 hours per task
# PROJECT_DIR is passed via --export from submit
#SBATCH -o %x-%A-%a.out                  # stdout per task (in working dir)
#SBATCH -e %x-%A-%a.err                  # stderr per task

# ---- Circulant {1,5,17} across all workloads ----
WORKLOADS=("Square 128Kx128K" "Wide 8Kx256K" "Tall 256Kx64K" "VeryWide 2Kx256K" "Rect 64Kx128K" "LargeSquare 256Kx256K" "VeryTall 256Kx8K")
TOPOLOGY="Circulant {1,5,17}"

WORKLOAD="${WORKLOADS[$SLURM_ARRAY_TASK_ID]}"

# Shared run directory — use the main run dir so results merge automatically
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
