#!/bin/bash
# Submit the parallel topology sweep + aggregation job.
#
# Usage:
#   bash slurm/submit.sh           # Full sweep: core topologies + circulant + 6D Hypercube
#   bash slurm/submit.sh --small   # Just Square 128Kx128K x 3 core topologies (3 jobs)
#   bash slurm/submit.sh --fast    # Only the fast topologies (21 jobs, no 6D)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LOGDIR="${PROJECT_DIR}/slurm/logs"
mkdir -p "$LOGDIR"

WORKLOADS=("Square 128Kx128K" "Wide 8Kx256K" "Tall 256Kx64K" "VeryWide 2Kx256K" "Rect 64Kx128K" "LargeSquare 256Kx256K" "VeryTall 256Kx8K")
TOPOLOGIES=("Torus 4x4x4" "Mesh 4x4x4" "Ring 64")
WORKLOADS_6D=("Wide 8Kx256K" "VeryWide 2Kx256K" "VeryTall 256Kx8K" "Square 128Kx128K" "Rect 64Kx128K")

MODE="${1:-full}"
case "$MODE" in
    --small)
        ARRAY_RANGE="0-2"
        echo "Submitting SMALL sweep (Square 128Kx128K x 3 topologies = 3 jobs)..."
        SUBMIT_6D=false
        SUBMIT_CIRCULANT=false
        ;;
    --fast)
        # 7 workloads x 3 topologies = 21 tasks (indices 0-20)
        ARRAY_RANGE="0-20"
        echo "Submitting FAST sweep (7 workloads x 3 topologies = 21 jobs, no 6D)..."
        SUBMIT_6D=false
        SUBMIT_CIRCULANT=false
        ;;
    *)
        # 7 workloads x 3 topologies = 21 tasks (indices 0-20)
        ARRAY_RANGE="0-20"
        echo "Submitting FULL sweep (21 core jobs + 7 circulant jobs + 5 x 6D Hypercube)..."
        SUBMIT_6D=true
        SUBMIT_CIRCULANT=true
        ;;
esac

ARRAY_JOB=$(sbatch --parsable \
    --chdir="$LOGDIR" \
    --array="$ARRAY_RANGE" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    slurm/sweep_array.sbatch)
echo "  Array job submitted: $ARRAY_JOB (tasks $ARRAY_RANGE)"

RUN_DIR="${PROJECT_DIR}/logs/slurm-${ARRAY_JOB}"
DEPS="afterany:${ARRAY_JOB}"

if [ "$SUBMIT_CIRCULANT" = true ]; then
    JOB_CIRC=$(sbatch --parsable \
        --chdir="$LOGDIR" \
        --array="0-6" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}",RUN_DIR="${RUN_DIR}" \
        slurm/sweep_circulant.sbatch)
    echo "  Circulant job submitted: $JOB_CIRC (7 tasks, 64GB each)"
    DEPS="${DEPS}:${JOB_CIRC}"
fi

if [ "$SUBMIT_6D" = true ]; then
    # 5 workloads x 6D Hypercube = 5 tasks (indices 0-4), 64GB, 6h
    JOB_6D=$(sbatch --parsable \
        --chdir="$LOGDIR" \
        --array="0-4" \
        --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
        slurm/sweep_6d.sbatch)
    echo "  6D Hypercube job submitted: $JOB_6D (5 tasks, 64GB each)"

    # Copy results from 6D job into the main run directory
    DEPS="${DEPS}:${JOB_6D}"

    # We need to merge 6D results into the main run dir
    MERGE_JOB=$(sbatch --parsable \
        --chdir="$LOGDIR" \
        --dependency="afterany:${JOB_6D}" \
        --wrap="cp -n ${PROJECT_DIR}/logs/slurm-${JOB_6D}/*.json ${RUN_DIR}/ 2>/dev/null || true" \
        --mem=1G -t 00:02:00 -p mit_normal -J merge-6d \
        -o merge-6d-%j.out -e merge-6d-%j.err)
    echo "  Merge job submitted: $MERGE_JOB (copies 6D results into main run dir)"
    DEPS="${DEPS}:${MERGE_JOB}"
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
echo "Job matrix (fast topologies):"
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
if [ "$SUBMIT_CIRCULANT" = true ]; then
    echo ""
    echo "Job matrix (Circulant {1,5,17}, 64GB):"
    for ((w=0; w<${#WORKLOADS[@]}; w++)); do
        printf "  Task %2d: %-28s x Circulant {1,5,17}\n" "$w" "${WORKLOADS[$w]}"
    done
fi
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOGDIR}/topo-sweep-${ARRAY_JOB}-<id>.out"
