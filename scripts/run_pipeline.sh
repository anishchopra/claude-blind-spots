#!/bin/bash
# Run the full pipeline: generate data, run inference, evaluate
#
# Usage: bash scripts/run_pipeline.sh <task> <run_name> [n_samples] [options]
#
# Options:
#   --thinking-budget N    Enable extended thinking with N token budget
#   --param key=value      Pass parameter to generator (can be repeated)
#   --regenerate           Delete existing data and regenerate from scratch
#   --rerun-inference      Delete existing predictions and rerun inference
#
# Examples:
#   bash scripts/run_pipeline.sh bar_height exp_01 10
#   bash scripts/run_pipeline.sh bar_height hard_test 50 --param height_diff=0.02
#   bash scripts/run_pipeline.sh bar_height thinking_test 10 --thinking-budget 1024
#   bash scripts/run_pipeline.sh bar_height exp_01 10 --regenerate
#   bash scripts/run_pipeline.sh bar_height exp_01 10 --rerun-inference

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash scripts/run_pipeline.sh <task> <run_name> [n_samples] [options]"
    echo ""
    echo "Options:"
    echo "  --thinking-budget N    Enable extended thinking with N token budget"
    echo "  --param key=value      Pass parameter to generator (can be repeated)"
    echo "  --regenerate           Delete existing data and regenerate from scratch"
    echo "  --rerun-inference      Delete existing predictions and rerun inference"
    echo ""
    echo "Example: bash scripts/run_pipeline.sh bar_height exp_01 10"
    exit 1
fi

TASK=$1
RUN=$2
N_SAMPLES=${3:-10}

# Shift past task, run, and n_samples
shift 3 2>/dev/null || shift $#

# Parse remaining args
THINKING_ARGS=""
GENERATE_ARGS=()
REGENERATE=false
RERUN_INFERENCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --thinking-budget)
            THINKING_ARGS="--thinking-budget $2"
            shift 2
            ;;
        --regenerate)
            REGENERATE=true
            shift
            ;;
        --rerun-inference)
            RERUN_INFERENCE=true
            shift
            ;;
        *)
            GENERATE_ARGS+=("$1")
            shift
            ;;
    esac
done

DATA_DIR="data/$RUN"

echo "=== Running pipeline for $TASK (run: $RUN, n=$N_SAMPLES) ==="

# Handle --regenerate: delete entire run directory
if [ "$REGENERATE" = true ] && [ -d "$DATA_DIR" ]; then
    echo "Regenerating: removing existing data directory $DATA_DIR"
    rm -rf "$DATA_DIR"
fi

# Handle --rerun-inference: delete prediction.json files
if [ "$RERUN_INFERENCE" = true ] && [ -d "$DATA_DIR" ]; then
    echo "Rerunning inference: removing existing prediction files"
    find "$DATA_DIR" -name "prediction.json" -delete
    # Also remove the report since it will be stale
    rm -f "$DATA_DIR/report.json"
fi

python -m scripts.generate_data $TASK --run $RUN -n $N_SAMPLES "${GENERATE_ARGS[@]}"
python -m scripts.run_inference --run $RUN $THINKING_ARGS
python -m scripts.evaluate --run $RUN
