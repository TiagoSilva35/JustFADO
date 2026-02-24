import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_metric_over_iterations(metric_values, metric_name, fairness_type):
    """
    Plots the given metric values over iterations for a specific model and fairness constraint.

    Args:
        metric_values (list): List of metric values recorded at each iteration.
        metric_name (str): Name of the metric being plotted (e.g., 'Accuracy', 'DP').
        model_type (str): Type of model used (e.g., 'mlp', 'forest').
        fairness_type (str): Type of fairness constraint applied (e.g., 'dp', 'eo').
        constraint_type (str): Type of constraint applied (e.g., 'node', 'global').
    """
    print(metric_values)
    metric_values = list(map(float, metric_values))
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=list(range(1, len(metric_values) + 1)), y=metric_values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.xticks(range(1, len(metric_values) + 1))
    plt.grid()
    plt.show()


def plot_metrics_over_timesteps(timestep_results, save_path='files/metrics_over_timesteps.png'):
    """Plot accuracy, DP, and EO over timesteps with drift phase boundaries.

    Args:
        timestep_results: dict with keys 'accuracy', 'dp', 'eo', 'n_samples'.
        save_path: Path to save the figure.
    """
    n = timestep_results['n_samples']
    timesteps = np.arange(1, n + 1)

    # Drift phase boundaries (from create_drifted_ds.py)
    splits = [0.15, 0.5, 0.6, 0.75, 1.0]
    boundaries = [int(s * n) for s in splits[:-1]]
    phase_labels = ['Warmup', 'Abrupt Drift', 'Recovery 1', 'Slow Drift', 'Recovery 2']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    metrics = [
        ('accuracy', 'Accuracy', 'tab:blue'),
        ('dp', 'Demographic Parity', 'tab:orange'),
        ('eo', 'Equalized Odds', 'tab:red'),
    ]

    for ax, (key, label, color) in zip(axes, metrics):
        values = timestep_results[key]
        ax.plot(timesteps, values, color=color, linewidth=1.0, alpha=0.85)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Draw drift phase boundaries
        for b in boundaries:
            ax.axvline(x=b, color='grey', linestyle='--', alpha=0.6)

        # Shade and label phases
        prev = 0
        colors_bg = ['#d4edda', '#f8d7da', '#d1ecf1', '#fff3cd', '#d4edda']
        for i, b in enumerate(boundaries + [n]):
            ax.axvspan(prev, b, alpha=0.08, color=colors_bg[i])
            if ax == axes[0]:
                mid = (prev + b) / 2
                ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1.0,
                        phase_labels[i], ha='center', va='bottom',
                        fontsize=9, fontstyle='italic', alpha=0.7)
            prev = b

    axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    axes[0].set_title('Metrics Over Timesteps (Drifted Test Set)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Timestep metrics plot saved to: {save_path}")
    plt.show()
