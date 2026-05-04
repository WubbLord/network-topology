#!/bin/bash
#SBATCH -J topo-aggregate                # Job name
#SBATCH -p mit_normal                    # MIT shared CPU partition
#SBATCH -c 1                             # Lightweight aggregation
#SBATCH --mem=4G
#SBATCH -t 00:05:00                      # 5 minutes max
# PROJECT_DIR and RUN_DIR are passed via --export from the submit wrapper.
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# This job is submitted with --dependency=afterok:<array_job_id>
# so it only runs after all 3 topology jobs complete successfully.

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: RUN_DIR not set. Pass it via --export=RUN_DIR=<path>"
    exit 1
fi
if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: PROJECT_DIR not set. Pass it via --export=PROJECT_DIR=<path>"
    exit 1
fi

echo "Aggregating results from: $RUN_DIR"

module load miniforge/25.11.0-0
source "${PROJECT_DIR}/.venv/bin/activate"

cd "$PROJECT_DIR"
python3 aggregate_results.py "$RUN_DIR"
