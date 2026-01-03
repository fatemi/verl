# Priority Sampling for GRPO Training

This example demonstrates priority-based sampling for GRPO training, comparing:

1. **Baseline**: GRPO (Dr.GRPO style) + random uniform sampling
2. **Prioritized**: GRPO (Dr.GRPO style) + priority-based sampling (ω = p(1-p))

## Prerequisites

```bash
# 1. Install verl (from project root)
cd /path/to/verl
pip install -e .

# 2. Install additional dependencies
pip install datasets  # For loading MATH-500 eval set

# 3. Install Qwen Math Eval Toolkit dependencies (for custom reward function)
pip install latex2sympy2 word2number regex sympy

# 4. Setup WandB for logging
pip install wandb
wandb login  # Or: export WANDB_API_KEY=your_key
```

## Quick Start (Single Node, 8 GPUs)

```bash
cd examples/priority_sampling

# 1. Download the guru math dataset (20.6 MB)
mkdir -p data
wget -O data/math__combined_54.4k.parquet \
  "https://huggingface.co/datasets/LLM360/guru-RL-92k/resolve/main/train/math__combined_54.4k.parquet"

# 2. Run guru.ipynb to prepare data (samples 100 problems, creates eval sets)
#    This creates: train.parquet, eval_online.parquet, eval_offline_math500.parquet

# 3. Set your model path
export MODEL_PATH=/path/to/Qwen2.5-7B  # Base model (not Instruct)

# 4. Run baseline experiment (8 GPUs, single node)
./run_experiment.sh baseline 100 run1

# 5. Run prioritized experiment
./run_experiment.sh prioritized 100 run1

# 6. Plot results
python plot_results.py --run_id run1
```

**GPU Compatibility**: Works with H100, H200, B200, or any NVIDIA GPU with sufficient memory.
B200's extra memory (192GB) provides headroom - no config changes needed.

---

## Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Configuration](#2-configuration)
3. [Running on Multi-Node Cluster](#3-running-on-multi-node-cluster)
4. [Monitoring Training](#4-monitoring-training)
5. [Incremental Training (Pause/Resume)](#5-incremental-training-pauseresume)
6. [Plotting Results](#6-plotting-results)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Data Preparation

### 1.1 Using the Notebook (Recommended)

The `guru.ipynb` notebook handles all data preparation:

1. **Download** the math subset from guru-RL-92k:
   - File: [math__combined_54.4k.parquet](https://huggingface.co/datasets/LLM360/guru-RL-92k/blob/main/train/math__combined_54.4k.parquet)
   - Save to: `examples/priority_sampling/data/math__combined_54.4k.parquet`
2. **Open** `guru.ipynb` and run all cells
3. **Outputs** are saved to `data/`:
   - `train.parquet` - 100 training problems (stratified by difficulty)
   - `train_distribution.png` - Distribution plot for paper
   - `eval_online.parquet` - Online evaluation set (MATH-500 sample)
   - `eval_offline_math500.parquet` - Full MATH-500 for offline eval

### 1.2 Data Format

Your data must be in **parquet** format with these required fields:

```python
{
    "data_source": "math",                          # Selects reward function
    "prompt": [{"role": "user", "content": "..."}], # Or just a string
    "reward_model": {
        "style": "rule",
        "ground_truth": "42"                        # Correct answer
    }
}
```

Extra columns (e.g., `qwen2.5_7b_pass_rate`) are preserved for analysis but ignored by verl.

### 1.3 Supported `data_source` Values

| Value | Use Case | Answer Format |
|-------|----------|---------------|
| `"math"` | General math problems | Flexible (boxed or plain) |
| `"aime24"` (or any `aime*`) | AIME problems | 3-digit integer (000-999) |
| `"HuggingFaceH4/MATH-500"` | MATH benchmark | `\boxed{answer}` |
| `"openai/gsm8k"` | GSM8K problems | `#### answer` |

### 1.4 Directory Structure

After downloading and running the notebook:

```
examples/priority_sampling/
├── data/
│   ├── math__combined_54.4k.parquet  # ← Download this first (20.6 MB)
│   ├── train.parquet                  # 100 sampled problems (generated)
│   ├── train_distribution.png         # Distribution plot (generated)
│   ├── eval_online.parquet            # Online eval set (generated)
│   └── eval_offline_math500.parquet   # Full MATH-500 (generated)
├── guru.ipynb                         # Data preparation notebook
├── config_baseline.yaml
├── config_prioritized.yaml
├── run_experiment.sh
├── plot_results.py
└── README.md
```

---

## 2. Configuration

### 2.1 Key Differences Between Configs

| Setting | Baseline | Prioritized |
|---------|----------|-------------|
| `priority_sampler.enabled` | `false` | `true` |
| `priority_sampler.alpha` | N/A | `0.8` (EMA smoothing) |
| `priority_sampler.success_bias` | N/A | `1e-4` (tie-breaker) |

### 2.2 Important Config Parameters

```yaml
# In config_*.yaml

data:
  train_batch_size: 16    # Problems per training step
  
actor_rollout_ref:
  rollout:
    n: 8                  # Rollouts per problem (for GRPO grouping)
    temperature: 0.7

trainer:
  test_freq: 25           # Eval every 25 steps
  save_freq: 100          # Full checkpoint every 100 steps (for resume)
  save_model_on_eval: true # Model-only checkpoint at each eval (for offline eval)
  total_training_steps: 100
```

### 2.3 Checkpoint Structure

With the above config, checkpoints are organized as:
```
checkpoints/<experiment_name>/
├── global_step_100/              # Full checkpoint (for resume)
│   ├── actor/
│   ├── critic/
│   └── data.pt
├── global_step_200/
│   └── ...
├── model_only/                   # Model-only checkpoints (for offline eval)
│   ├── step_25/
│   │   └── actor/
│   ├── step_50/
│   │   └── actor/
│   ├── step_75/
│   │   └── actor/
│   └── ...
└── latest_checkpointed_iteration.txt
```

- **Full checkpoints** (`global_step_N/`): Saved at `save_freq`, include optimizer state for resume
- **Model-only checkpoints** (`model_only/step_N/`): Saved at `test_freq`, lightweight for offline eval

### 2.4 Memory Considerations

For 3 nodes × 8 H100s:
- `train_batch_size: 16` → 16 problems × 8 rollouts = 128 generations/step
- Adjust `gpu_memory_utilization` if OOM (default: 0.85)

---

## 3. Running on Multi-Node Cluster

### 3.1 Environment Setup (All Nodes)

```bash
# On each node
cd /path/to/verl
pip install -e .

# Set environment variables
export MODEL_PATH=/shared/path/to/model
export TRAIN_DATA=/shared/path/to/train.parquet
export EVAL_DATA=/shared/path/to/eval_online.parquet
export CKPT_DIR=/shared/path/to/checkpoints

# WandB setup
export WANDB_API_KEY=your_key
export WANDB_PROJECT=verl_priority_sampling
```

### 3.2 Launch Commands

**Node 0 (Master):**
```bash
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500
export NNODES=3
export NODE_RANK=0
export GPUS_PER_NODE=8

cd examples/priority_sampling
./run_experiment.sh baseline 100 run1
```

**Node 1:**
```bash
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
export NNODES=3
export NODE_RANK=1
export GPUS_PER_NODE=8

cd examples/priority_sampling
./run_experiment.sh baseline 100 run1
```

**Node 2:**
```bash
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
export NNODES=3
export NODE_RANK=2
export GPUS_PER_NODE=8

cd examples/priority_sampling
./run_experiment.sh baseline 100 run1
```

### 3.3 Using SLURM

Create `submit_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=priority_sampling
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out

# Get master node info
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Launch on each node
srun --ntasks-per-node=1 bash -c "
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export NNODES=$SLURM_NNODES
    export NODE_RANK=\$SLURM_NODEID
    export GPUS_PER_NODE=8
    
    cd examples/priority_sampling
    ./run_experiment.sh $1 $2 $3
"
```

Submit:
```bash
sbatch submit_job.sh baseline 100 run1
sbatch submit_job.sh prioritized 100 run1
```

---

## 4. Monitoring Training

### 4.1 WandB Dashboard

Training logs to WandB automatically. Monitor:

- **`train/reward/mean`**: Average reward per step
- **`val/math/score/mean`**: MATH accuracy  
- **`val/aime24/score/mean`**: AIME accuracy
- **`priority/heap_size`**: Active problems (prioritized only)
- **`priority/solved_pool_size`**: Fully solved problems
- **`priority/unsolved_pool_size`**: Fully failed problems

### 4.2 Key Metrics to Watch

| Metric | Description | Good Sign |
|--------|-------------|-----------|
| `val/*/score/mean` | Validation accuracy | Increasing |
| `priority/heap_size` | Active pool size | Stable (not depleting) |
| `priority/num_flipped_*` | Status changes | Occasional flips |

### 4.3 Log Files

Check training logs:
```bash
# Latest output
tail -f logs/<job_id>.out

# Or in checkpoints
ls checkpoints/baseline_run1/
```

---

## 5. Incremental Training (Pause/Resume)

### 5.1 Strategy: Run in K-Step Sessions

Run training in increments of K=100 steps, then review and continue:

```bash
# Session 1: Steps 0-100
./run_experiment.sh baseline 100 run1

# Review results on WandB...

# Session 2: Steps 100-200 (auto-resumes from checkpoint)
./run_experiment.sh baseline 200 run1  # Note: total_steps=200
```

### 5.2 How Resume Works

verl auto-resumes from the latest checkpoint in:
```
checkpoints/<experiment_name>/
```

The config has:
```yaml
trainer:
  default_local_dir: ${ckpt_dir}/${trainer.experiment_name}
```

### 5.3 Manual Checkpoint Selection

To resume from a specific checkpoint:

```bash
python -m verl.trainer.main_ppo \
    --config-path examples/priority_sampling \
    --config-name config_baseline \
    trainer.resume_from_path=/path/to/checkpoint/step_100 \
    total_steps=200 \
    ...
```

### 5.4 Recommended Workflow

```
Step 0-100:   Run both baseline and prioritized
              └── Review: Are curves separating?
              
Step 100-200: Continue both
              └── Review: Is priority maintaining advantage?
              
Step 200-400: Continue if needed
              └── Final comparison
              
Step 400-600: Only if results are promising
```

---

## 6. Plotting Results

### 6.1 Quick Plot

```bash
# Using WandB data
python plot_results.py --project verl_priority_sampling --run_id run1

# Demo with mock data
python plot_results.py --demo
```

### 6.2 Output Files

```
plots/
├── accuracy_comparison.png    # Main comparison plot
├── response_length.png        # Response length over time
└── priority_distribution.png  # Solved/unsolved pool sizes
```

### 6.3 Custom Analysis

```python
import wandb
api = wandb.Api()

# Fetch runs
baseline = api.run("your_project/baseline_run1")
priority = api.run("your_project/prioritized_run1")

# Get history as DataFrame
baseline_df = baseline.history()
priority_df = priority.history()

# Compare final accuracy
print(f"Baseline final: {baseline_df['val/score/mean'].iloc[-1]:.3f}")
print(f"Priority final: {priority_df['val/score/mean'].iloc[-1]:.3f}")
```

---

## 7. Troubleshooting

### 7.1 Common Issues

**OOM (Out of Memory)**
```yaml
# Reduce in config
actor_rollout_ref:
  rollout:
    gpu_memory_utilization: 0.7  # Lower from 0.85
  actor:
    ppo_mini_batch_size: 32      # Lower from 64
```

**Data Source Not Found**
```
NotImplementedError: Reward function is not implemented for data_source='my_custom'
```
→ Use supported values: `"math"`, `"aime24"`, etc.

**WandB Connection Issues**
```bash
export WANDB_MODE=offline  # Run offline, sync later
wandb sync --sync-all      # Sync after training
```

### 7.2 Multi-Node Issues

```bash
# Test connectivity
ping <master_ip>
nc -zv <master_ip> 29500

# Check NCCL
export NCCL_DEBUG=INFO
```

---

## File Reference

| File | Purpose |
|------|---------|
| `guru.ipynb` | Data preparation notebook (uses guru-RL-92k) |
| `config_baseline.yaml` | Baseline GRPO config |
| `config_prioritized.yaml` | Priority sampling config |
| `run_experiment.sh` | Main experiment launcher |
| `plot_results.py` | Results visualization |
| `README.md` | This documentation |

---

## Expected Results

With a well-designed 100-problem training set of varying difficulty:

- **Baseline**: Uniform sampling may waste compute on too-easy/too-hard problems
- **Prioritized**: Focuses on "boundary" problems (μ_g ≈ 0.5), potentially faster learning

The priority sampler should show:
1. Growing `solved_pool_size` as easy problems are mastered
2. Growing `unsolved_pool_size` for persistently hard problems
3. Stable `heap_size` with problems at the learning frontier
