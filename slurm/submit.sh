#!/bin/bash
# Submit the parallel topology sweep + aggregation job.
#
# Usage:
#   bash slurm/submit.sh           # Full sweep: main topologies + 6D Hypercube
#   bash slurm/submit.sh --small   # Smaller + selected giant workloads x main topologies
#   bash slurm/submit.sh --fast    # All workloads x main topologies, no 6D

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LOGDIR="${PROJECT_DIR}/slurm/logs"
mkdir -p "$LOGDIR"

WORKLOADS=(
    "Small 2Kx8K"
    "Small 4Kx16K"
    "Medium 8Kx32K"
    "Medium 16Kx32K"
    "Wide 8Kx256K"
    "VeryWide 2Kx256K"
    "Rect 64Kx128K"
    "Square 128Kx128K"
    "Tall 256Kx64K"
    "LargeSquare 256Kx256K"
    "VeryTall 256Kx8K"
)
TOPOLOGIES=("Torus 4x4x4" "Mesh 4x4x4" "Ring 64" "Circulant {1,5,17}")
WORKLOADS_6D=(
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

SMALL_WORKLOAD_COUNT=7
CORE_LAST=$(( ${#WORKLOADS[@]} * ${#TOPOLOGIES[@]} - 1 ))
SMALL_LAST=$(( SMALL_WORKLOAD_COUNT * ${#TOPOLOGIES[@]} - 1 ))
HYPER_LAST=$(( ${#WORKLOADS_6D[@]} - 1 ))

MODE="${1:-full}"
case "$MODE" in
    --small)
        ARRAY_RANGE="0-${SMALL_LAST}"
        echo "Submitting SMALL sweep (${SMALL_WORKLOAD_COUNT} workloads x ${#TOPOLOGIES[@]} topologies = $((SMALL_LAST + 1)) jobs)..."
        SUBMIT_6D=false
        ;;
    --fast)
        ARRAY_RANGE="0-${CORE_LAST}"
        echo "Submitting FAST sweep (${#WORKLOADS[@]} workloads x ${#TOPOLOGIES[@]} topologies = $((CORE_LAST + 1)) jobs, no 6D)..."
        SUBMIT_6D=false
        ;;
    *)
        ARRAY_RANGE="0-${CORE_LAST}"
        echo "Submitting FULL sweep ($((CORE_LAST + 1)) main jobs + $((HYPER_LAST + 1)) x 6D Hypercube)..."
        SUBMIT_6D=true
        ;;
esac

ARRAY_JOB=$(sbatch --parsable \
    --chdir="$LOGDIR" \
    --array="$ARRAY_RANGE" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    slurm/sweep_array.sbatch)
echo "  Array job submitted: $ARRAY_JOB (tasks $ARRAY_RANGE)"

RUN_DIR="${PROJECT_DIR}/logs/slurm-${ARRAY_JOB}"
DEPS="afterok:${ARRAY_JOB}"

if [ "$SUBMIT_6D" = true ]; then
    JOB_6D=$(sbatch --parsable \
        --chdir="$LOGDIR" \
        --array="0-${HYPER_LAST}" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_DIR}" \
        slurm/sweep_6d.sbatch)
    echo "  6D Hypercube job submitted: $JOB_6D ($((HYPER_LAST + 1)) tasks)"

    DEPS="${DEPS}:${JOB_6D}"
fi

AGG_JOB=$(sbatch --parsable \
    --chdir="$LOGDIR" \
    --dependency="${DEPS}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_DIR}" \
    slurm/aggregate.sbatch)
echo "  Aggregation job submitted: $AGG_JOB (runs after all jobs complete)"

echo ""
echo "Results will be in: ${RUN_DIR}/results.json"
echo ""
echo "Job matrix (main topologies):"
NUM_TOPOS=${#TOPOLOGIES[@]}
for ((w=0; w<${#WORKLOADS[@]}; w++)); do
    for ((t=0; t<NUM_TOPOS; t++)); do
        idx=$((w * NUM_TOPOS + t))
        max_idx=$(echo "$ARRAY_RANGE" | sed 's/.*-//')
        if [ "$idx" -le "$max_idx" ]; then
            printf "  Task %2d: %-28s x %s\n" "$idx" "${WORKLOADS[$w]}" "${TOPOLOGIES[$t]}"
        fi
    done
done
if [ "$SUBMIT_6D" = true ]; then
    echo ""
    echo "Job matrix (6D Hypercube, 64GB):"
    for ((w=0; w<${#WORKLOADS_6D[@]}; w++)); do
        printf "  Task %2d: %-28s x 6D Hypercube\n" "$w" "${WORKLOADS_6D[$w]}"
    done
fi
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOGDIR}/topo-sweep-${ARRAY_JOB}-<id>.out"
