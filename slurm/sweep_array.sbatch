#!/bin/bash
#SBATCH -J topo-sweep                    # Job name
#SBATCH -p mit_normal                    # MIT shared CPU partition (12h max)
#SBATCH -c 8                             # 8 CPUs per task (mapper is CPU-bound)
#SBATCH --mem=40G                        # 40 GB memory (AccelForge mapper peaks ~33 GB)
#SBATCH -t 04:00:00                      # 4 hours per task
# PROJECT_DIR is passed via --export from submit.sh
#SBATCH -o %x-%A-%a.out                  # stdout per task (in working dir)
#SBATCH -e %x-%A-%a.err                  # stderr per task

# ---- Job matrix: (workload, topology) pairs ----
# Array index encodes: index = workload_idx * num_topologies + topology_idx
WORKLOADS=("Square 128Kx128K" "Wide 8Kx256K" "Tall 256Kx64K" "VeryWide 2Kx256K" "Rect 64Kx128K" "LargeSquare 256Kx256K" "VeryTall 256Kx8K")
TOPOLOGIES=("Torus 4x4x4" "Mesh 4x4x4" "Ring 64")

NUM_TOPOS=${#TOPOLOGIES[@]}
WORKLOAD_IDX=$((SLURM_ARRAY_TASK_ID / NUM_TOPOS))
TOPO_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TOPOS))

WORKLOAD="${WORKLOADS[$WORKLOAD_IDX]}"
TOPOLOGY="${TOPOLOGIES[$TOPO_IDX]}"

# Shared run directory
RUN_DIR="${PROJECT_DIR}/logs/slurm-${SLURM_ARRAY_JOB_ID}"

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
