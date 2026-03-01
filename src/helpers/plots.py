import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


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

    drift_pts = timestep_results.get('drift_points', [])
    
    for ax, (key, label, color) in zip(axes, metrics):
        values = timestep_results[key]
        ax.plot(timesteps, values, color=color, linewidth=1.0, alpha=0.85)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        for pts in drift_pts:
            ax.axvline(x=pts, color='purple', linestyle='--', alpha=0.7, label='Drift Detected' if pts == drift_pts[0] else "")
        if drift_pts and ax == axes[0]:
            ax.legend(fontsize=9)
        
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


# ---------------------------------------------------------------------------
# Comparison plots across drift scenarios
# ---------------------------------------------------------------------------

def plot_scenario_comparison_timesteps(all_results, output_dir='files/experiments'):
    """Overlay timestep metrics for every scenario.

    Args:
        all_results: list of dicts, each with keys
            'scenario', 'timestep_results'.
        output_dir: directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ('accuracy', 'Accuracy'),
        ('dp', 'Demographic Parity'),
        ('eo', 'Equalized Odds'),
    ]

    # --- One figure per metric, all scenarios overlaid ---
    for key, label in metrics:
        fig, ax = plt.subplots(figsize=(16, 6))
        for res in all_results:
            ts = res.get('timestep_results')
            if ts is None:
                continue
            n = ts['n_samples']
            timesteps = np.arange(1, n + 1)
            ax.plot(timesteps, ts[key], linewidth=1.0, alpha=0.8, label=res['scenario'])

        # Phase boundaries (use first available n_samples)
        for res in all_results:
            ts = res.get('timestep_results')
            if ts is not None:
                n = ts['n_samples']
                splits = [0.15, 0.50, 0.60, 0.75]
                for s in splits:
                    ax.axvline(x=int(s * n), color='grey', linestyle='--', alpha=0.4)
                break

        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Over Timesteps – All Scenarios', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f'comparison_{key}_timesteps.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()


def plot_scenario_comparison_bar(all_results, output_dir='files/experiments'):
    """Bar chart comparing final test metrics across scenarios.

    Args:
        all_results: list of dicts with 'scenario', 'test_metrics'.
        output_dir: directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = []
    acc_vals, dp_vals, eo_vals, f1_vals = [], [], [], []
    for res in all_results:
        tm = res.get('test_metrics')
        if tm is None:
            continue
        labels.append(res['scenario'])
        acc_vals.append(tm.get('accuracy', 0))
        dp_vals.append(tm.get('dp', 0))
        eo_vals.append(tm.get('eo', 0))
        f1_vals.append(tm.get('f1', 0))

    if not labels:
        print("No test metrics to plot.")
        return

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 7))
    ax.bar(x - 1.5 * width, acc_vals, width, label='Accuracy', color='tab:blue')
    ax.bar(x - 0.5 * width, dp_vals, width, label='DP', color='tab:orange')
    ax.bar(x + 0.5 * width, eo_vals, width, label='EO', color='tab:red')
    ax.bar(x + 1.5 * width, f1_vals, width, label='F1', color='tab:green')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=25, ha='right')
    ax.set_ylabel('Metric Value')
    ax.set_title('Test Metrics – All Drift Scenarios (Aranyani)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_bar_metrics.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_summary_heatmap(all_results, output_dir='files/experiments'):
    """Single-row heatmap of final metrics for each scenario.

    Args:
        all_results: list of dicts with 'scenario', 'test_metrics'.
        output_dir: directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    filtered = [r for r in all_results if r.get('test_metrics')]
    if not filtered:
        print("No test metrics for heatmap.")
        return

    scenarios = [r['scenario'] for r in filtered]
    metric_keys = [('accuracy', 'Accuracy'), ('dp', 'DP'), ('eo', 'EO'), ('f1', 'F1')]

    data = np.full((len(scenarios), len(metric_keys)), np.nan)
    for i, r in enumerate(filtered):
        for j, (mk, _) in enumerate(metric_keys):
            data[i, j] = r['test_metrics'].get(mk, np.nan)

    fig, ax = plt.subplots(figsize=(8, max(3, len(scenarios) * 0.7)))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=[ml for _, ml in metric_keys],
                yticklabels=scenarios, ax=ax, linewidths=0.5)
    ax.set_title('Aranyani – Metrics by Drift Scenario', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'heatmap_scenarios.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
