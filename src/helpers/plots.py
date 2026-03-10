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


def _smooth(values, window=50):
    """Apply a rolling average to reduce noise while preserving trends."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # 'valid' mode shrinks the array; pad edges with edge values to keep length
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def plot_metrics_over_timesteps(timestep_results, save_path='files/metrics_over_timesteps.png',
                                skip_samples=0, smooth_window=100):
    """Plot accuracy, DP, and EO over timesteps with drift phase boundaries.

    All metrics are rolling-window values (computed in utils.py with WINDOW_SIZE=500).
    smooth_window applies a light visual smoothing on top to remove remaining noise.

    Args:
        timestep_results: dict returned by evaluate_over_timesteps.
        save_path: Path to save the figure.
        skip_samples: Number of initial samples to skip in the plot.
        smooth_window: Rolling average window for visual smoothing only.
    """
    n = timestep_results['n_samples']
    plot_start = skip_samples
    timesteps = np.arange(1, n + 1)[plot_start:]

    # Ground-truth drift phase boundaries (from create_drifted_ds.py)
    splits = [0.15, 0.5, 0.6, 0.75, 1.0]
    boundaries = [int(s * n) for s in splits[:-1]]
    phase_labels = ['Warmup', 'Abrupt Drift', 'Recovery 1', 'Slow Drift', 'Recovery 2']
    colors_bg    = ['#d4edda', '#f8d7da',     '#d1ecf1',    '#fff3cd',    '#d4edda']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Metrics Over Timesteps (Drifted Test Set)',
                 fontsize=14, fontweight='bold')

    metrics = [
        ('accuracy', 'Accuracy',           'tab:blue',   [0.5, 1.0]),
        ('dp',       'Demographic Parity',  'tab:orange', [0.0, 0.5]),
        ('eo',       'Equalized Odds',      'tab:red',    [0.0, 0.5]),
    ]

    metric_drift_pts = {
        'accuracy': timestep_results.get('drifted_points', []),
        'dp':       timestep_results.get('drifted_points_dp', []),
        'eo':       timestep_results.get('drifted_points_eo', []),
    }

    for ax, (key, label, color, ylim) in zip(axes, metrics):
        values = _smooth(np.array(timestep_results[key][plot_start:], dtype=float),
                         window=smooth_window)
        ax.plot(timesteps, values, color=color, linewidth=1.8)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

        # Detected drift markers
        pts = metric_drift_pts[key]
        first = True
        for p in pts:
            ax.axvline(x=p, color='purple', linestyle='--', linewidth=1.2, alpha=0.8,
                       label='Drift detected' if first else '_nolegend_')
            first = False
        if pts:
            ax.legend(fontsize=9, loc='lower left')

        # Phase shading + boundary lines + labels (top subplot only)
        prev = 0
        for i, b in enumerate(boundaries + [n]):
            ax.axvspan(prev, b, alpha=0.07, color=colors_bg[i], zorder=0)
            if prev > 0:
                ax.axvline(x=prev, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
            if ax is axes[0]:
                ax.text((prev + b) / 2, ylim[1] * 0.97,
                        phase_labels[i], ha='center', va='top',
                        fontsize=8, fontstyle='italic', color='dimgrey')
            prev = b

    axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Timestep metrics plot saved to: {save_path}")
    plt.show()


def plot_per_tree_accuracy(timestep_results, save_path='files/per_tree_accuracy.png',
                           skip_samples=0, smooth_window=100):
    """Plot the rolling-window accuracy of each individual tree over timesteps.

    Args:
        timestep_results: dict returned by evaluate_over_timesteps (must contain
            'per_tree_accuracy').
        save_path: Path to save the figure.
        skip_samples: Number of initial samples to skip in the plot.
        smooth_window: Rolling average window for visual smoothing.
    """
    per_tree = timestep_results.get('per_tree_accuracy')
    if not per_tree:
        print("No per-tree accuracy data found in timestep_results.")
        return

    n = timestep_results['n_samples']
    plot_start = skip_samples
    timesteps = np.arange(1, n + 1)[plot_start:]

    # Phase boundaries
    splits = [0.15, 0.5, 0.6, 0.75, 1.0]
    boundaries = [int(s * n) for s in splits[:-1]]
    phase_labels = ['Warmup', 'Abrupt Drift', 'Recovery 1', 'Slow Drift', 'Recovery 2']
    colors_bg    = ['#d4edda', '#f8d7da',     '#d1ecf1',    '#fff3cd',    '#d4edda']

    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle('Per-Tree Rolling Accuracy Over Timesteps', fontsize=14, fontweight='bold')

    for tree_id, tree_accs in enumerate(per_tree):
        values = _smooth(np.array(tree_accs[plot_start:], dtype=float), window=smooth_window)
        ax.plot(timesteps, values, linewidth=1.6, color=palette[tree_id % len(palette)],
                label=f'Tree {tree_id}')

    # Ensemble accuracy for reference
    ensemble = _smooth(np.array(timestep_results['accuracy'][plot_start:], dtype=float),
                       window=smooth_window)
    ax.plot(timesteps, ensemble, linewidth=2.2, color='black', linestyle='--',
            label='Ensemble')

    # Drift markers
    for p in timestep_results.get('drifted_points', []):
        ax.axvline(x=p, color='purple', linestyle='--', linewidth=1.0, alpha=0.7)

    # Tree-reset markers (one colour per tree, labelled once)
    reset_events = timestep_results.get('tree_reset_events', [])
    reset_labeled = set()
    for ev in reset_events:
        tid = ev['tree_id']
        col = palette[tid % len(palette)]
        label = f'Tree {tid} reset' if tid not in reset_labeled else '_nolegend_'
        ax.axvline(x=ev['timestep'], color=col, linestyle=':', linewidth=1.4,
                   alpha=0.9, label=label)
        reset_labeled.add(tid)

    # Phase shading
    prev = 0
    for i, b in enumerate(boundaries + [n]):
        ax.axvspan(prev, b, alpha=0.07, color=colors_bg[i], zorder=0)
        if prev > 0:
            ax.axvline(x=prev, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.text((prev + b) / 2, 0.99, phase_labels[i], ha='center', va='top',
                fontsize=8, fontstyle='italic', color='dimgrey',
                transform=ax.get_xaxis_transform())
        prev = b

    ax.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim([0.4, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-tree accuracy plot saved to: {save_path}")
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
            values = _smooth(np.array(ts[key], dtype=float), window=50)
            ax.plot(timesteps, values, linewidth=1.0, alpha=0.8, label=res['scenario'])

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


# ---------------------------------------------------------------------------
# Aranyani vs ARF comparison plots
# ---------------------------------------------------------------------------

def plot_aranyani_vs_arf(aranyani_results, arf_results, save_path,
                         scenario_name='', smooth_window=100, skip_samples=0):
    """Single figure comparing Aranyani and ARF across accuracy, DP, and EO.

    Args:
        aranyani_results: dict from evaluate_over_timesteps.
        arf_results: dict from evaluate_arf_over_timesteps.
        save_path: Path to save the figure.
        scenario_name: Used in the figure title.
        smooth_window: Rolling average window for visual smoothing.
        skip_samples: Number of initial samples to skip.
    """
    n = aranyani_results['n_samples']
    plot_start = skip_samples
    timesteps = np.arange(1, n + 1)[plot_start:]

    splits = [0.15, 0.5, 0.6, 0.75, 1.0]
    boundaries = [int(s * n) for s in splits[:-1]]
    phase_labels = ['Warmup', 'Abrupt Drift', 'Recovery 1', 'Slow Drift', 'Recovery 2']
    colors_bg    = ['#d4edda', '#f8d7da', '#d1ecf1', '#fff3cd', '#d4edda']

    metrics = [
        ('accuracy', 'Accuracy',          [0.5, 1.0]),
        ('dp',       'Demographic Parity', [0.0, 0.5]),
        ('eo',       'Equalized Odds',     [0.0, 0.5]),
    ]

    title = f'Aranyani vs ARF – {scenario_name}' if scenario_name else 'Aranyani vs ARF'
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for ax, (key, label, ylim) in zip(axes, metrics):
        aran_vals = _smooth(np.array(aranyani_results[key][plot_start:], dtype=float),
                            window=smooth_window)
        arf_vals  = _smooth(np.array(arf_results[key][plot_start:], dtype=float),
                            window=smooth_window)

        ax.plot(timesteps, aran_vals, color='tab:blue',   linewidth=1.8, label='Aranyani')
        ax.plot(timesteps, arf_vals,  color='tab:orange', linewidth=1.8,
                linestyle='--', label='ARF')

        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

        # Aranyani drift detection markers
        first = True
        for p in aranyani_results.get('drifted_points', []):
            ax.axvline(x=p, color='purple', linestyle='--', linewidth=1.2, alpha=0.8,
                       label='Drift detected (Aranyani)' if first else '_nolegend_')
            first = False

        ax.legend(fontsize=9, loc='lower left')

        # Phase shading
        prev = 0
        for i, b in enumerate(boundaries + [n]):
            ax.axvspan(prev, b, alpha=0.07, color=colors_bg[i], zorder=0)
            if prev > 0:
                ax.axvline(x=prev, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
            if ax is axes[0]:
                ax.text((prev + b) / 2, ylim[1] * 0.97,
                        phase_labels[i], ha='center', va='top',
                        fontsize=8, fontstyle='italic', color='dimgrey')
            prev = b

    axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Aranyani vs ARF plot saved to: {save_path}")
    plt.close()


