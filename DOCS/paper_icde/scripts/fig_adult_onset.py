"""Onset-aligned prequential curves on Adult (one injected drift), mean over
seeds: rolling DP, rolling accuracy, and FADO's lambda_t.

    python fig_adult_onset.py <scenario> <out.pdf>
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
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else 'sim_x_permutations'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'adult_onset.pdf'
BEFORE, AFTER, BIN = 6000, 4800, 50

ARMS = [  # fixed categorical order (validated palette), plus a dash for print/CVD
    ('fado', 'FADO', '#2a78d6', '-'),
    ('aranyani', 'Aranyani', '#eb6834', '--'),
    ('arf', 'ARF', '#1baf7a', ':'),
    ('rfr', 'RFR', '#eda100', '-.'),
]

curves = {k: {'dp': [], 'accuracy': [], 'lambda': []} for k, *_ in ARMS}
seeds = set()
for path in glob.glob(f'{REPO}/files/experiments/wandb/*/dataset_adult/seed_pipeline_results.json'):
    d = json.load(open(path))
    if d.get('config', {}).get('drift_scenario') != SCENARIO:
        continue
    for run in d['seed_runs']:
        for row in run['results']:
            ts = row.get('timestep_results') or {}
            info = ts.get('drift_info') or {}
            onset = info.get('onset')
            if row.get('scenario') != SCENARIO or onset is None or run['model'] not in curves:
                continue
            n = len(ts['dp'])
            if onset - BEFORE < 0 or onset + AFTER > n:
                continue
            seeds.add(run['seed'])
            for key in ('dp', 'accuracy', 'lambda'):
                series = ts.get(key)
                if series is None:
                    continue
                seg = np.asarray(series[onset - BEFORE:onset + AFTER], dtype=float)
                curves[run['model']][key].append(seg.reshape(-1, BIN).mean(axis=1))

x = (np.arange((BEFORE + AFTER) // BIN) * BIN - BEFORE + BIN / 2) / 1000.0
# IEEE template: 8 pt Times New Roman for figure labels.
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                     'pdf.fonttype': 42})
fig, axes = plt.subplots(3, 1, figsize=(3.45, 4.3), sharex=True,
                         gridspec_kw={'height_ratios': [1.2, 1.0, 0.8], 'hspace': 0.12})
ink, muted, grid = '#2b2b2b', '#6b6b6b', '#e4e4e2'
for ax, key, label in ((axes[0], 'dp', 'Demographic parity\n(rolling)'),
                       (axes[1], 'accuracy', 'Accuracy\n(rolling)')):
    for model, name, color, dash in ARMS:
        segs = curves[model][key]
        if not segs:
            continue
        m = np.mean(segs, axis=0)
        ax.plot(x, m, color=color, lw=1.4 if model == 'fado' else 1.1, ls=dash, label=name,
                zorder=3 if model == 'fado' else 2)
    ax.set_ylabel(label, color=ink)
lam = np.asarray(curves['fado']['lambda'])
q25, q50, q75 = np.percentile(lam, [25, 50, 75], axis=0)
axes[2].fill_between(x, q25, q75, color='#2a78d6', alpha=0.18, lw=0)
axes[2].plot(x, lam.mean(axis=0), color='#2a78d6', lw=1.4, label='FADO $\\lambda_t$ (mean, IQR)')
axes[2].axhline(0.1, color=muted, lw=0.8, ls='--')
axes[2].text(x[2], 0.1, ' fixed weight of Aranyani', color=muted, va='bottom', fontsize=8)
axes[2].set_ylabel('Penalty weight\n$\\lambda_t$', color=ink)
axes[2].set_xlabel('Samples relative to drift onset (thousands)', color=ink)
for ax in axes:
    ax.axvline(0, color=ink, lw=0.8, ls='-', alpha=0.6)
    ax.grid(axis='y', color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)
axes[0].text(0.05, axes[0].get_ylim()[1], 'drift onset', color=ink, fontsize=8, va='top')
axes[0].legend(ncol=4, loc='lower center', bbox_to_anchor=(0.45, 1.0), frameon=False,
               handlelength=2.0, columnspacing=0.8)
fig.savefig(OUT, bbox_inches='tight')
fig.savefig(OUT.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
print(f'{SCENARIO}: seeds={len(seeds)} ' +
      ' '.join(f'{m}={len(curves[m]["dp"])}' for m, *_ in ARMS) + f' -> {OUT}')
