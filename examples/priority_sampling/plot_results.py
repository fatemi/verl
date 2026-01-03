#!/usr/bin/env python3
"""
Plot Comparison Results for GRPO Experiments.

This script fetches metrics from WandB and creates comparison plots
for baseline vs. prioritized GRPO training.

Usage:
    python plot_results.py --project verl_priority_sampling --run_id run1
    python plot_results.py --project verl_priority_sampling --baseline_run xxx --priority_run yyy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Optional: try to import wandb for fetching results
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Using mock data for demo.")


def fetch_wandb_history(project: str, run_name: str) -> dict:
    """Fetch training history from WandB."""
    if not HAS_WANDB:
        return None
    
    api = wandb.Api()
    
    # Find run by name
    runs = api.runs(project, filters={"display_name": run_name})
    if not runs:
        print(f"Run '{run_name}' not found in project '{project}'")
        return None
    
    run = runs[0]
    history = run.history()
    
    return {
        "steps": history["_step"].tolist(),
        "train_reward": history.get("train/reward/mean", []).tolist(),
        "val_score": history.get("val/score/mean", []).tolist(),
        "val_math_score": history.get("val/math/score/mean", []).tolist(),
        "val_aime_score": history.get("val/aime24/score/mean", []).tolist(),
        "response_length": history.get("rollout/response_length/mean", []).tolist(),
        "priority_heap_size": history.get("priority/heap_size", []).tolist(),
        "priority_solved_pool": history.get("priority/solved_pool_size", []).tolist(),
        "priority_unsolved_pool": history.get("priority/unsolved_pool_size", []).tolist(),
    }


def create_mock_data(experiment_type: str, steps: int = 100) -> dict:
    """Create mock data for demo/testing."""
    np.random.seed(42 if experiment_type == "baseline" else 43)
    
    x = np.arange(0, steps + 1, 25)  # Eval every 25 steps
    
    # Simulate learning curves
    if experiment_type == "baseline":
        # Baseline: slower improvement
        base_score = 0.2 + 0.3 * (1 - np.exp(-x / 50))
        noise = np.random.randn(len(x)) * 0.02
    else:
        # Prioritized: faster improvement
        base_score = 0.2 + 0.4 * (1 - np.exp(-x / 40))
        noise = np.random.randn(len(x)) * 0.02
    
    scores = np.clip(base_score + noise, 0, 1)
    
    return {
        "steps": x.tolist(),
        "val_score": scores.tolist(),
        "val_math_score": (scores * 0.9).tolist(),
        "val_aime_score": (scores * 0.3).tolist(),  # AIME is harder
        "response_length": (500 + 100 * np.random.randn(len(x))).tolist(),
        "priority_heap_size": (80 - x * 0.3).tolist() if experiment_type == "prioritized" else [],
        "priority_solved_pool": (x * 0.15).tolist() if experiment_type == "prioritized" else [],
        "priority_unsolved_pool": (x * 0.05).tolist() if experiment_type == "prioritized" else [],
    }


def plot_comparison(baseline_data: dict, priority_data: dict, output_dir: str = "plots"):
    """Create comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig_size = (12, 5)
    
    # ==========================================================================
    # Plot 1: Accuracy Comparison
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    # Overall score
    ax = axes[0]
    if baseline_data.get("val_score"):
        ax.plot(baseline_data["steps"], baseline_data["val_score"], 
                "b-o", label="GRPO (Baseline)", markersize=6)
    if priority_data.get("val_score"):
        ax.plot(priority_data["steps"], priority_data["val_score"], 
                "r-s", label="GRPO + Priority", markersize=6)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Validation Score")
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Per-dataset scores
    ax = axes[1]
    if baseline_data.get("val_math_score"):
        ax.plot(baseline_data["steps"], baseline_data["val_math_score"], 
                "b-o", label="Baseline (MATH)", markersize=6, alpha=0.7)
    if baseline_data.get("val_aime_score"):
        ax.plot(baseline_data["steps"], baseline_data["val_aime_score"], 
                "b--^", label="Baseline (AIME)", markersize=6, alpha=0.7)
    if priority_data.get("val_math_score"):
        ax.plot(priority_data["steps"], priority_data["val_math_score"], 
                "r-s", label="Priority (MATH)", markersize=6, alpha=0.7)
    if priority_data.get("val_aime_score"):
        ax.plot(priority_data["steps"], priority_data["val_aime_score"], 
                "r--d", label="Priority (AIME)", markersize=6, alpha=0.7)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Dataset Scores")
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")
    plt.close()
    
    # ==========================================================================
    # Plot 2: Response Length
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if baseline_data.get("response_length"):
        ax.plot(baseline_data["steps"], baseline_data["response_length"], 
                "b-o", label="GRPO (Baseline)", markersize=6)
    if priority_data.get("response_length"):
        ax.plot(priority_data["steps"], priority_data["response_length"], 
                "r-s", label="GRPO + Priority", markersize=6)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Response Length (tokens)")
    ax.set_title("Average Response Length")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "response_length.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'response_length.png'}")
    plt.close()
    
    # ==========================================================================
    # Plot 3: Priority Sampler Statistics (if available)
    # ==========================================================================
    if priority_data.get("priority_heap_size"):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        steps = priority_data["steps"]
        ax.stackplot(
            steps,
            priority_data.get("priority_heap_size", [0] * len(steps)),
            priority_data.get("priority_solved_pool", [0] * len(steps)),
            priority_data.get("priority_unsolved_pool", [0] * len(steps)),
            labels=["Active (Heap)", "Solved Pool", "Unsolved Pool"],
            colors=["#2ecc71", "#3498db", "#e74c3c"],
            alpha=0.7,
        )
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Number of Problems")
        ax.set_title("Priority Sampler: Problem Distribution")
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        plt.savefig(output_dir / "priority_distribution.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'priority_distribution.png'}")
        plt.close()
    
    print(f"\nAll plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO comparison results")
    parser.add_argument("--project", type=str, default="verl_priority_sampling", help="WandB project name")
    parser.add_argument("--run_id", type=str, help="Run ID to find both experiments")
    parser.add_argument("--baseline_run", type=str, help="Baseline run name (overrides run_id)")
    parser.add_argument("--priority_run", type=str, help="Priority run name (overrides run_id)")
    parser.add_argument("--output_dir", type=str, default="plots", help="Output directory for plots")
    parser.add_argument("--demo", action="store_true", help="Use mock data for demo")
    
    args = parser.parse_args()
    
    if args.demo or not HAS_WANDB:
        print("Using mock data for demonstration...")
        baseline_data = create_mock_data("baseline", steps=100)
        priority_data = create_mock_data("prioritized", steps=100)
    else:
        # Determine run names
        baseline_name = args.baseline_run or f"baseline_{args.run_id}"
        priority_name = args.priority_run or f"prioritized_{args.run_id}"
        
        print(f"Fetching data from WandB project: {args.project}")
        print(f"  Baseline run: {baseline_name}")
        print(f"  Priority run: {priority_name}")
        
        baseline_data = fetch_wandb_history(args.project, baseline_name) or {}
        priority_data = fetch_wandb_history(args.project, priority_name) or {}
    
    plot_comparison(baseline_data, priority_data, args.output_dir)


if __name__ == "__main__":
    main()

