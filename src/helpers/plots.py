import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_metric_over_iterations(metric_values, metric_name, fairness_type):
    metric_values = list(map(float, metric_values))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(metric_values) + 1), metric_values, marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _smooth(values, window=50):
    """Apply a rolling average to reduce noise while preserving trends."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # 'valid' mode shrinks the array; pad edges with edge values to keep length
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def _dynamic_ylim(values, pad=0.05):
    """Return (ymin, ymax) with a small padding around the data range."""
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    margin = max((hi - lo) * pad, pad)
    return lo - margin, hi + margin


def plot_metrics_over_timesteps(timestep_results, save_path='files/metrics_over_timesteps.png',
                                skip_samples=0, smooth_window=100):
    n = timestep_results['n_samples']
    plot_start = skip_samples
    timesteps = np.arange(1, n + 1)[plot_start:]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Metrics Over Timesteps', fontsize=14, fontweight='bold')

    metrics = [
        ('accuracy', 'Accuracy',          'tab:blue'),
        ('dp',       'Demographic Parity', 'tab:orange'),
        ('eo',       'Equalized Odds',     'tab:red'),
    ]

    drift_pts = timestep_results.get('drifted_points', [])

    for ax, (key, label, color) in zip(axes, metrics):
        values = _smooth(np.array(timestep_results[key][plot_start:], dtype=float),
                         window=smooth_window)
        ax.plot(timesteps, values, color=color, linewidth=1.8)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_ylim(_dynamic_ylim(values))
        ax.grid(True, alpha=0.3)

        first = True
        for p in drift_pts:
            ax.axvline(x=p, color='purple', linestyle='--', linewidth=1.2, alpha=0.8,
                       label='Drift detected' if first else '_nolegend_')
            first = False
        if drift_pts:
            ax.legend(fontsize=9, loc='lower left')

    axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Timestep metrics plot saved to: {save_path}")
    plt.show()


def plot_aranyani_vs_arf(aranyani_results, arf_results, save_path,
                        scenario_name='', smooth_window=100, skip_samples=0,
                        fair_arf_results=None):
    n = aranyani_results['n_samples']
    plot_start = skip_samples
    timesteps = np.arange(1, n + 1)[plot_start:]

    metrics = [
        ('accuracy', 'Accuracy'),
        ('dp',       'Demographic Parity'),
        ('eo',       'Equalized Odds'),
    ]

    title = (
        f'Aranyani vs ARF vs Fair-ARF - {scenario_name}'
        if fair_arf_results is not None and scenario_name
        else (
            'Aranyani vs ARF vs Fair-ARF'
            if fair_arf_results is not None
            else (f'Aranyani vs ARF - {scenario_name}' if scenario_name else 'Aranyani vs ARF')
        )
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    drift_pts = aranyani_results.get('drifted_points', [])

    for ax, (key, label) in zip(axes, metrics):
        aran_vals = _smooth(np.array(aranyani_results[key][plot_start:], dtype=float),
                            window=smooth_window)
        arf_vals  = _smooth(np.array(arf_results[key][plot_start:], dtype=float),
                            window=smooth_window)

        ax.plot(timesteps, aran_vals, color='tab:blue',   linewidth=1.8, label='Aranyani')
        ax.plot(timesteps, arf_vals,  color='tab:orange', linewidth=1.8,
                linestyle='--', label='ARF')
        fair_arf_results = None
        if fair_arf_results is not None:
            fair_vals = _smooth(
                np.array(fair_arf_results[key][plot_start:], dtype=float),
                window=smooth_window,
            )
            ax.plot(
                timesteps,
                fair_vals,
                color='tab:green',
                linewidth=1.8,
                linestyle='-.',
                label='Fair-ARF (DP-targeted)',
            )
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        combined = np.concatenate([aran_vals, arf_vals])
        if fair_arf_results is not None:
            combined = np.concatenate([combined, fair_vals])
        ax.set_ylim(_dynamic_ylim(combined))
        ax.grid(True, alpha=0.3)

        first = True
        for p in drift_pts:
            ax.axvline(x=p, color='purple', linestyle='--', linewidth=1.2, alpha=0.8,
                       label='Drift detected' if first else '_nolegend_')
            first = False
        ax.legend(fontsize=9, loc='lower left')

    axes[-1].set_xlabel('Timestep', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()
