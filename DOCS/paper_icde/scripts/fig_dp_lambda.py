"""Rolling demographic parity and FADO's fairness weight over the stream, mean over seeds.

    cd DOCS/paper_icde/figures && python ../scripts/fig_dp_lambda.py dp_lambda.pdf [--allow-partial]

One column per block of Table III: Folktables over its whole stream, and each Adult drift aligned
on its onset (the window every run of that scenario contains). Top row: rolling DP of FADO, Aranyani
and Aranyani with leaked lambda, with the target epsilon. Bottom row: FADO's lambda_t (mean and
interquartile band) against the two constant weights. Runs come from the sweeps of
make_tables.SWEEPS whose run folders are on this machine. A method is drawn only when all 30 seeds
of that block are present, unless --allow-partial (for previews; the panel title then gives n).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402
from runs import local_runs  # noqa: E402

BIN = 100
EPSILON = 0.05
FOLK = 'folktables'
COLUMNS = [(FOLK, 'Folktables'), ('sim_x_permutations', 'x_permutations'), ('sim_y_swaps', 'y_swaps'),
           ('sim_y_prior_skip', 'y_prior_skip'), ('sim_x_exceed_skip', 'x_exceed_skip')]
# Fixed categorical order of the validated default palette, plus a dash for print and CVD.
ARMS = [('fado', 'FADO', '#2a78d6', '-'), ('aranyani', 'Aranyani', '#eb6834', '--'),
        ('tuned', 'Aranyani, leaked $\\lambda$', '#1baf7a', ':')]
INK, MUTED, GRID = '#2b2b2b', '#6b6b6b', '#e4e4e2'


def load():
    """{scenario: {arm: [run, ...]}} from the local folders of the main and leaked-lambda sweeps."""
    main = mt.SWEEPS['main_adult'] + mt.SWEEPS['main_folktables']
    tuned = mt.SWEEPS['tuned_adult'] + mt.SWEEPS['tuned_folktables']
    data = {}
    for run in local_runs(set(main), models=['fado', 'aranyani']) + local_runs(set(tuned), models=['aranyani']):
        arm = 'tuned' if run['sweep'] in tuned else run['model']
        data.setdefault(run['scenario'], {}).setdefault(arm, []).append(run)
    return data


def aligned(runs, scenario):
    """x (thousands of samples) and the per-run binned series of `runs`, over a common window."""
    if scenario == FOLK:
        n = min(len(r['dp']) for r in runs) // BIN * BIN
        x = (np.arange(n // BIN) * BIN + BIN / 2) / 1000
        return x, lambda r, k: np.asarray(r[k][:n], float).reshape(-1, BIN).mean(1)
    before = min(r['drift_info']['onset'] for r in runs) // BIN * BIN
    after = min(len(r['dp']) - r['drift_info']['onset'] for r in runs) // BIN * BIN
    x = (np.arange((before + after) // BIN) * BIN - before + BIN / 2) / 1000

    def series(r, k):
        o = r['drift_info']['onset']
        return np.asarray(r[k][o - before:o + after], float).reshape(-1, BIN).mean(1)
    return x, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='dp_lambda.pdf')
    ap.add_argument('--allow-partial', action='store_true')
    args = ap.parse_args()
    data, leaked = load(), mt.leaked_lambdas()
    cols = [(sc, t) for sc, t in COLUMNS if sc in data]
    for sc, t in COLUMNS:
        if sc not in data:
            print(f'[skip] {t}: no local run folders of the paper sweeps')
    if not cols:
        raise SystemExit('No block has local data.')

    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'font.family': 'Times New Roman',
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, len(cols), figsize=(7.16, 3.2), sharex='col', sharey='row', squeeze=False,
                             gridspec_kw={'height_ratios': [1.15, 1.0], 'hspace': 0.12, 'wspace': 0.08})
    for c, (sc, title) in enumerate(cols):
        runs = data[sc]
        every = [r for arm in runs.values() for r in arm]
        x, series = aligned(every, sc)
        counts = []
        for arm, name, color, dash in ARMS:
            rs = runs.get(arm, [])
            counts.append(f'{name.split(",")[0] if arm != "tuned" else "leaked"} {len(rs)}')
            if not rs or (len(rs) < mt.N_SEEDS and not args.allow_partial):
                continue
            dp = np.mean([series(r, 'dp') for r in rs], axis=0)
            axes[0, c].plot(x, dp, color=color, ls=dash, lw=1.4 if arm == 'fado' else 1.1, label=name,
                            zorder=3 if arm == 'fado' else 2)
            if arm == 'fado':
                lam = np.array([series(r, 'lambda') for r in rs])
                q25, q75 = np.percentile(lam, [25, 75], axis=0)
                axes[1, c].fill_between(x, q25, q75, color=color, alpha=0.18, lw=0)
                axes[1, c].plot(x, lam.mean(0), color=color, ls=dash, lw=1.4)
            else:
                const = mt.ARANYANI_LAMBDA if arm == 'aranyani' else leaked['folktables' if sc == FOLK else 'adult']
                axes[1, c].plot(x, np.full_like(x, const), color=color, ls=dash, lw=1.1)
        axes[0, c].axhline(EPSILON, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        axes[0, c].set_title(title if not args.allow_partial else f'{title}\n{", ".join(counts)}', color=INK,
                             fontsize=8 if not args.allow_partial else 6)
        if sc == FOLK:
            axes[1, c].set_xlabel('Samples (thousands)', color=INK)
        if sc != FOLK:
            for r in (0, 1):
                axes[r, c].axvline(0, color=INK, lw=0.8, alpha=0.6)
        print(f'{title}: window {x[0]:.1f}..{x[-1]:.1f}k samples; seeds: {", ".join(counts)}')
    axes[0, 0].set_ylabel('Demographic parity\n(rolling)', color=INK)
    axes[1, 0].set_ylabel('Fairness weight $\\lambda$', color=INK)
    axes[0, 0].text(axes[0, 0].get_xlim()[0], EPSILON, ' $\\varepsilon$', color=MUTED, va='bottom', fontsize=7)
    for ax in axes.flat:
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    adult = [c for c, (sc, _) in enumerate(cols) if sc != FOLK]
    if adult:  # one label across the onset-aligned columns, so the long labels do not collide
        fig.canvas.draw()
        left, right = axes[1, adult[0]].get_position().x0, axes[1, adult[-1]].get_position().x1
        bottom = axes[1, adult[0]].get_position().y0
        fig.text((left + right) / 2, bottom - 0.085, 'Samples from the drift onset (thousands)', ha='center',
                 va='top', color=INK, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.84 if not args.allow_partial else 0.78)
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.93), frameon=False,
               handlelength=2.2, columnspacing=1.4)
    fig.savefig(args.out, bbox_inches='tight')
    fig.savefig(args.out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
