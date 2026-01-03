#!/bin/bash
# =============================================================================
# GRPO Comparison Experiment Runner
# =============================================================================
# This script runs the GRPO comparison experiment on multi-node GPU clusters.
#
# Usage:
#   # Single node (8 GPUs)
#   ./run_experiment.sh baseline 100
#
#   # Multi-node (set MASTER_ADDR, MASTER_PORT, NODE_RANK first)
#   ./run_experiment.sh prioritized 100
#
# Arguments:
#   $1: Experiment type: "baseline" or "prioritized"
#   $2: Number of training steps (default: 100)
#   $3: Run ID for experiment tracking (default: "run1")
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
EXPERIMENT_TYPE="${1:-baseline}"
TOTAL_STEPS="${2:-100}"
RUN_ID="${3:-run1}"

# Paths (modify these for your setup)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
TRAIN_DATA="${TRAIN_DATA:-$SCRIPT_DIR/data/train.parquet}"
EVAL_DATA="${EVAL_DATA:-$SCRIPT_DIR/data/eval_online.parquet}"
CKPT_DIR="${CKPT_DIR:-$SCRIPT_DIR/checkpoints}"

# Distributed settings (set these for multi-node)
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# WandB settings
export WANDB_PROJECT="${WANDB_PROJECT:-verl_priority_sampling}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RUN_ID}"

# =============================================================================
# Validation
# =============================================================================
echo "============================================="
echo "GRPO Comparison Experiment"
echo "============================================="
echo "Experiment:    $EXPERIMENT_TYPE"
echo "Total Steps:   $TOTAL_STEPS"
echo "Run ID:        $RUN_ID"
echo "Model:         $MODEL_PATH"
echo "Train Data:    $TRAIN_DATA"
echo "Eval Data:     $EVAL_DATA"
echo "Checkpoints:   $CKPT_DIR"
echo "============================================="
echo "Distributed Settings:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NNODES:      $NNODES"
echo "  NODE_RANK:   $NODE_RANK"
echo "  GPUS/NODE:   $GPUS_PER_NODE"
echo "============================================="

# Validate experiment type
if [[ "$EXPERIMENT_TYPE" != "baseline" && "$EXPERIMENT_TYPE" != "prioritized" ]]; then
    echo "Error: EXPERIMENT_TYPE must be 'baseline' or 'prioritized'"
    exit 1
fi

# Validate paths
if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
    echo "Warning: MODEL_PATH does not exist: $MODEL_PATH"
    echo "Set MODEL_PATH environment variable to your model path"
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Error: Training data not found: $TRAIN_DATA"
    echo "Run: python prepare_data.py --create_sample"
    exit 1
fi

if [[ ! -f "$EVAL_DATA" ]]; then
    echo "Error: Eval data not found: $EVAL_DATA"
    echo "Run: python prepare_data.py --create_sample"
    exit 1
fi

# =============================================================================
# Run Training
# =============================================================================
echo ""
echo "Starting $EXPERIMENT_TYPE experiment..."
echo ""

cd "$VERL_ROOT"

# Select config
CONFIG_NAME="config_$EXPERIMENT_TYPE"

# Build command
CMD="python -m verl.trainer.main_ppo \
    --config-path $SCRIPT_DIR \
    --config-name $CONFIG_NAME \
    model_path=$MODEL_PATH \
    train_data=$TRAIN_DATA \
    eval_data=$EVAL_DATA \
    total_steps=$TOTAL_STEPS \
    run_id=$RUN_ID \
    ckpt_dir=$CKPT_DIR \
    nnodes=$NNODES \
    n_gpus_per_node=$GPUS_PER_NODE"

# For multi-node, use torchrun
if [[ "$NNODES" -gt 1 ]]; then
    echo "Running multi-node training with torchrun..."
    torchrun \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m verl.trainer.main_ppo \
        --config-path "$SCRIPT_DIR" \
        --config-name "$CONFIG_NAME" \
        model_path="$MODEL_PATH" \
        train_data="$TRAIN_DATA" \
        eval_data="$EVAL_DATA" \
        total_steps="$TOTAL_STEPS" \
        run_id="$RUN_ID" \
        ckpt_dir="$CKPT_DIR" \
        nnodes="$NNODES" \
        n_gpus_per_node="$GPUS_PER_NODE"
else
    echo "Running single-node training..."
    eval $CMD
fi

echo ""
echo "============================================="
echo "Experiment completed!"
echo "Checkpoints saved to: $CKPT_DIR"
echo "============================================="

