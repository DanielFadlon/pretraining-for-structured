"""
Plot TensorBoard Results

Extracts metrics from TensorBoard event files and saves plots as images.

Usage:
    python scripts/plot_tensorboard_results.py <log_dir> [output_dir]
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_logs(log_dir: str) -> dict[str, list[tuple[int, float]]]:
    """
    Load all scalar metrics from TensorBoard event files.

    Args:
        log_dir: Path to the TensorBoard log directory

    Returns:
        Dictionary mapping metric names to list of (step, value) tuples
    """
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0}  # Load all scalars
    )
    ea.Reload()

    metrics = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]

    return metrics


def plot_metrics(metrics: dict[str, list[tuple[int, float]]], output_dir: str) -> None:
    """
    Plot and save all metrics.

    Args:
        metrics: Dictionary of metrics from load_tensorboard_logs
        output_dir: Directory to save plot images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group metrics by category
    train_metrics = {k: v for k, v in metrics.items() if 'train' in k.lower()}
    eval_metrics = {k: v for k, v in metrics.items() if 'eval' in k.lower()}
    other_metrics = {k: v for k, v in metrics.items()
                     if 'train' not in k.lower() and 'eval' not in k.lower()}

    # Plot training metrics
    if train_metrics:
        _plot_metric_group(train_metrics, "Training Metrics",
                          os.path.join(output_dir, "training_metrics.png"))

    # Plot evaluation metrics
    if eval_metrics:
        _plot_metric_group(eval_metrics, "Evaluation Metrics",
                          os.path.join(output_dir, "evaluation_metrics.png"))

    # Plot other metrics
    if other_metrics:
        _plot_metric_group(other_metrics, "Other Metrics",
                          os.path.join(output_dir, "other_metrics.png"))

    # Plot individual metrics
    for name, values in metrics.items():
        safe_name = name.replace('/', '_').replace(' ', '_')
        _plot_single_metric(name, values,
                           os.path.join(output_dir, f"{safe_name}.png"))

    print(f"Plots saved to: {output_dir}")


def _plot_metric_group(metrics: dict, title: str, output_path: str) -> None:
    """Plot a group of metrics in a single figure."""
    n_metrics = len(metrics)
    if n_metrics == 0:
        return

    cols = min(2, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        steps, vals = zip(*values) if values else ([], [])
        ax.plot(steps, vals, marker='o', markersize=3, linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_single_metric(name: str, values: list[tuple[int, float]], output_path: str) -> None:
    """Plot a single metric."""
    if not values:
        return

    steps, vals = zip(*values)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, vals, marker='o', markersize=4, linewidth=2, color='#2E86AB')
    plt.title(name, fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add min/max annotations
    if len(vals) > 1:
        min_idx = vals.index(min(vals))
        max_idx = vals.index(max(vals))
        plt.annotate(f'Min: {vals[min_idx]:.4f}',
                    xy=(steps[min_idx], vals[min_idx]),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=9, color='red')
        plt.annotate(f'Max: {vals[max_idx]:.4f}',
                    xy=(steps[max_idx], vals[max_idx]),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(metrics: dict[str, list[tuple[int, float]]]) -> None:
    """Print a summary of all metrics."""
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)

    for name, values in sorted(metrics.items()):
        if not values:
            continue
        steps, vals = zip(*values)
        print(f"\n{name}:")
        print(f"  Steps: {min(steps)} -> {max(steps)}")
        print(f"  Min: {min(vals):.6f} at step {steps[vals.index(min(vals))]}")
        print(f"  Max: {max(vals):.6f} at step {steps[vals.index(max(vals))]}")
        print(f"  Final: {vals[-1]:.6f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(log_dir, "plots")

    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    print(f"Loading TensorBoard logs from: {log_dir}")
    metrics = load_tensorboard_logs(log_dir)

    if not metrics:
        print("No metrics found in the log directory.")
        sys.exit(1)

    print(f"Found {len(metrics)} metrics: {list(metrics.keys())}")

    print_summary(metrics)
    plot_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
