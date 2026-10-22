"""Onset-aligned prequential curves on Adult, one column per injected drift:
rolling DP, rolling accuracy, and FADO's lambda_t, mean over seeds.

    python fig_adult_onset_grid.py <out.pdf> [scenario ...]

Only runs of the paper's sweeps are used (PAPER_SWEEPS), so other Adult runs in
files/experiments/wandb/ (e.g. the matched-lambda Aranyani sweeps) never mix
in. Every seed is kept: each column shows the window that every run of that
scenario has (BEFORE samples before the onset, up to AFTER after it; the skip
drifts drop rows, so their streams are shorter after the onset). Scenarios
without local result folders are skipped and reported.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'adult_onset_grid.pdf'
SCENARIOS = sys.argv[2:] or ['sim_x_permutations', 'sim_y_swaps',
                             'sim_y_prior_skip', 'sim_x_exceed_skip']
PAPER_SWEEPS = {'uwhcwsdg', 'bi6frxi3', 'l8c4i60o'}  # Table III, Adult
BEFORE, AFTER, BIN = 6000, 4800, 50
EPSILON, LAMBDA_BASE = 0.05, 0.1

ARMS = [  # fixed categorical order (validated palette), plus a dash for print/CVD
    ('fado', 'FADO', '#2a78d6', '-'),
    ('aranyani', 'Aranyani', '#eb6834', '--'),
    ('arf', 'ARF', '#1baf7a', ':'),
    ('rfr', 'RFR', '#eda100', '-.'),
]
TITLES = {'sim_x_permutations': 'x_permutations', 'sim_y_swaps': 'y_swaps',
          'sim_y_prior_skip': 'y_prior_skip', 'sim_x_exceed_skip': 'x_exceed_skip'}


def load(scenario):
    """{model: [(onset, {metric: array})]} for the paper runs of `scenario`."""
    runs = {m: [] for m, *_ in ARMS}
    for path in glob.glob(f'{REPO}/files/experiments/wandb/*/dataset_adult/seed_pipeline_results.json'):
        d = json.load(open(path))
        cfg = d.get('config', {})
        if cfg.get('wandb_sweep_id') not in PAPER_SWEEPS or cfg.get('drift_scenario') != scenario:
            continue
        for run in d['seed_runs']:
            if run['model'] not in runs:
                continue
            ts = run['results'][0]['timestep_results']
            runs[run['model']].append((ts['drift_info']['onset'], {
                k: np.asarray(ts[k], dtype=float) for k in ('dp', 'accuracy', 'lambda') if ts.get(k)}))
    return runs


def window(runs):
    """Largest BIN-aligned (before, after) window that every run contains."""
    pairs = [(o, len(s['dp']) - o) for rs in runs.values() for o, s in rs]
    before = min(BEFORE, min(p[0] for p in pairs)) // BIN * BIN
    after = min(AFTER, min(p[1] for p in pairs)) // BIN * BIN
    return before, after


data = {}
for sc in SCENARIOS:
    runs = load(sc)
    if not runs['fado']:
        print(f'[skip] {sc}: no local result folders from the paper sweeps')
        continue
    data[sc] = (runs, *window(runs))

if not data:
    raise SystemExit('No scenario has local data.')

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                     'pdf.fonttype': 42})
ink, muted, grid = '#2b2b2b', '#6b6b6b', '#e4e4e2'
ncol = len(data)
fig, axes = plt.subplots(3, ncol, figsize=(7.16, 4.1), sharex='col', sharey='row', squeeze=False,
                         gridspec_kw={'height_ratios': [1.2, 1.0, 0.8], 'hspace': 0.14,
                                      'wspace': 0.08})
for c, (sc, (runs, before, after)) in enumerate(data.items()):
    x = (np.arange((before + after) // BIN) * BIN - before + BIN / 2) / 1000.0

    def binned(series, onset):
        return series[onset - before:onset + after].reshape(-1, BIN).mean(axis=1)

    for r, key in ((0, 'dp'), (1, 'accuracy')):
        ax = axes[r, c]
        for model, name, color, dash in ARMS:
            segs = [binned(s[key], o) for o, s in runs[model]]
            if segs:
                ax.plot(x, np.mean(segs, axis=0), color=color, ls=dash, label=name,
                        lw=1.4 if model == 'fado' else 1.1, zorder=3 if model == 'fado' else 2)
    axes[0, c].axhline(EPSILON, color=muted, lw=0.8, ls=(0, (2, 2)))
    lam = np.asarray([binned(s['lambda'], o) for o, s in runs['fado']])
    q25, q75 = np.percentile(lam, [25, 75], axis=0)
    axes[2, c].fill_between(x, q25, q75, color='#2a78d6', alpha=0.18, lw=0)
    axes[2, c].plot(x, lam.mean(axis=0), color='#2a78d6', lw=1.4)
    axes[2, c].axhline(LAMBDA_BASE, color=muted, lw=0.8, ls='--')
    n_seeds = len(runs['fado'])
    axes[0, c].set_title(f'{TITLES.get(sc, sc)} ($n={n_seeds}$)', color=ink)
    axes[2, c].set_xlabel('Samples from onset (thousands)', color=ink)
    print(f'{sc}: window -{before}/+{after}, seeds ' +
          ' '.join(f'{m}={len(runs[m])}' for m, *_ in ARMS))

axes[0, 0].set_ylabel('Demographic parity\n(rolling)', color=ink)
axes[1, 0].set_ylabel('Accuracy\n(rolling)', color=ink)
axes[2, 0].set_ylabel('FADO $\\lambda_t$\n(mean, IQR)', color=ink)
axes[0, 0].text(axes[0, 0].get_xlim()[0], EPSILON, ' $\\varepsilon$', color=muted,
                va='bottom', fontsize=7)
axes[2, 0].text(axes[2, 0].get_xlim()[0], LAMBDA_BASE, ' Aranyani $\\lambda$', color=muted,
                va='bottom', fontsize=7)
for ax in axes.flat:
    ax.axvline(0, color=ink, lw=0.8, alpha=0.6)
    ax.grid(axis='y', color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.subplots_adjust(top=0.9)
fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.935),
           frameon=False, handlelength=2.2, columnspacing=1.2)
fig.savefig(OUT, bbox_inches='tight')
fig.savefig(OUT.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
print(f'-> {OUT}')
