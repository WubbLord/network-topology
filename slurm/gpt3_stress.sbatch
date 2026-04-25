#!/bin/bash
#SBATCH -J gpt3-stress                   # Job name
#SBATCH -p mit_normal                    # MIT shared CPU partition
#SBATCH -c 4                             # 4 CPUs (this is CPU-only, no AccelForge mapper)
#SBATCH --mem=8G                         # 8 GB should be plenty
#SBATCH -t 01:00:00                      # 1 hour max
#SBATCH -o logs/gpt3-stress-%j.out       # stdout
#SBATCH -e logs/gpt3-stress-%j.err       # stderr

PROJECT_DIR="/home/nvemuri/projects/network-topology"

echo "========================================"
echo "GPT-3 Network Stress Analysis"
echo "Node: $(hostname), CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "========================================"

# ---- Environment ----
module load miniforge/25.11.0-0
source "${PROJECT_DIR}/.venv/bin/activate"

# ---- Run ----
cd "$PROJECT_DIR"
python3 analyze_gpt3_stress.py

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
