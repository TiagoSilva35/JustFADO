"""Build the paper's result tables from W&B and the local run folders.

    python DOCS/paper_icde/scripts/make_tables.py            # cached W&B data where present
    python DOCS/paper_icde/scripts/make_tables.py --refresh  # download everything again

Writes (paper.tex \\input's the two .tex files, so the text is never touched):
    DOCS/paper_icde/tables/results.tex          Table III: F1 / DP / EO, whole stream and after onset
    DOCS/paper_icde/tables/detection.tex        Table IV: drift detection by FADO on Adult
    DOCS/paper_icde/supplementary/detection_split.{md,csv}
                                                FADO - Aranyani after onset, detected vs undetected
    DOCS/paper_icde/supplementary/significance.md
                                                FADO vs each baseline: paired Wilcoxon and Welch
                                                t-test, Holm-corrected (the paper's "Welch results
                                                in the repository")

Sources:
    * W&B per-arm tables (`arm_metrics`) and run summaries of the sweeps in SWEEPS,
      cached in files/experiments/wandb_arm_metrics/<sweeps>/ (raw.csv, detection.csv).
      Only finished runs are used; re-run with --refresh while sweeps are still running.
    * F1 needs the per-sample predictions, which only exist in the result folders of the
      machine that ran each sweep (files/experiments/wandb/<run id>/). Copy them with rsync.
      For runs whose folder is missing, the whole-stream F1 falls back to
      files/experiments/extra_metrics/per_seed.csv (written by stream_f1.py).

Rules (same as the tables already in the paper):
    * a cell is filled only when every seed (30) of that method has the value; otherwise
      it is left empty;
    * bold = best mean in the block; FADO's superscripts = paired Wilcoxon, Holm-corrected
      across Aranyani, ARF and RFR, significant in FADO's favour against all three;
    * the tuned-lambda Aranyani gets a dagger when it differs from FADO (paired Wilcoxon,
      p < 0.05).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PAPER = os.path.join(REPO, 'DOCS', 'paper_icde')
CACHE = os.path.join(REPO, 'files', 'experiments', 'wandb_arm_metrics')
F1_FALLBACK = os.path.join(REPO, 'files', 'experiments', 'extra_metrics', 'per_seed.csv')
ENTITY, PROJECT = 'tjcsilva04-universidade-de-coimbra', 'fado-ablations'
N_SEEDS = 30

# Which W&B sweeps feed the tables. Edit here when a sweep is re-run.
SWEEPS = {
    'main_folktables': ['jo5dqqhy'],                          # FADO, Aranyani, ARF, RFR
    'main_adult': ['uwhcwsdg', 'bi6frxi3', 'l8c4i60o'],       # same, 4 injected drifts
    'tuned_folktables': ['9ulal3qn'],                         # Aranyani, lambda 0.5, pre-train 0.1
    'tuned_adult': ['rm2glvmh', 'wqab0alm', 'pzbwmd4e'],      # Aranyani, lambda 1.5, pre-train 0.1
}
BLOCKS = [  # (scenario key, block title in the table)
    ('folktables', r'\folk{}, natural cross-year shift'),
    ('sim_x_permutations', r'\adult{}, x\_permutations'),
    ('sim_y_swaps', r'\adult{}, y\_swaps'),
    ('sim_y_prior_skip', r'\adult{}, y\_prior\_skip'),
    ('sim_x_exceed_skip', r'\adult{}, x\_exceed\_skip'),
]
ADULT = [sc for sc, _ in BLOCKS if sc != 'folktables']
ROWS = [('fado', r'\fado{}'), ('aranyani', r'\ara{}'), ('tuned', r'\ara{}, tuned $\lambda^{\mathrm{d}}$'),
        ('arf', r'\arf{}'), ('rfr', r'\rfr{}')]
BASELINES = ['aranyani', 'arf', 'rfr']
# (column, higher is better)
COLS = [('f1', True), ('dp', False), ('eo', False),
        ('f1_post', True), ('post_drift_dp', False), ('post_drift_eo', False)]
POST = {'f1_post', 'post_drift_dp', 'post_drift_eo'}


# ----------------------------------------------------------------------------- data
def _cache_dir(sweeps):
    return os.path.join(CACHE, '+'.join(sweeps))


def arm_metrics(sweeps, refresh):
    """W&B `arm_metrics` rows of `sweeps` (finished runs), cached as raw.csv."""
    path = os.path.join(_cache_dir(sweeps), 'raw.csv')
    if refresh or not os.path.isfile(path):
        sys.path.insert(0, REPO)
        from src.fetch_wandb_arm_metrics import backfill_from_summary
        from src.fetch_wandb_trajectories import fetch
        os.makedirs(_cache_dir(sweeps), exist_ok=True)
        raw = fetch(ENTITY, PROJECT, ','.join(sweeps),
                    cache_dir=os.path.join(_cache_dir(sweeps), 'artifacts'), table_name='arm_metrics')
        raw = backfill_from_summary(raw, ENTITY, PROJECT)
        raw.to_csv(path, index=False)
        print(f'[fetch] {"+".join(sweeps)}: {len(raw)} rows')
    return pd.read_csv(path)


def detection(sweeps, refresh):
    """Per-run detection results of FADO (true positives, delays) from W&B run summaries."""
    path = os.path.join(_cache_dir(sweeps), 'detection.csv')
    if refresh or not os.path.isfile(path):
        import wandb
        api = wandb.Api(timeout=120)
        rows = []
        for sid in sweeps:
            for run in api.sweep(f'{ENTITY}/{PROJECT}/{sid}').runs:
                if run.state != 'finished':
                    continue
                s = run.summary

                def num(key):
                    v = s.get(key)
                    return float(v) if isinstance(v, (int, float)) else np.nan
                rows.append(dict(scenario=run.config.get('drift_scenario'), seed=int(run.config.get('seeds')),
                                 acc_tp=num('fado/accuracy_true_positives'), acc_mtd=num('fado/accuracy_mtd'),
                                 fair_tp=num('fado/fairness_true_positives'), fair_mtd=num('fado/fairness_mtd'),
                                 acc_fa=num('fado/accuracy_false_alarms'), fair_fa=num('fado/fairness_false_alarms')))
        os.makedirs(_cache_dir(sweeps), exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f'[fetch] detection {"+".join(sweeps)}: {len(rows)} runs')
    return pd.read_csv(path)


def _f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan


def local_f1(raw, dataset, model_name=None):
    """Whole-stream and after-onset F1 from the local result folders of the runs in `raw`."""
    ids = set(raw.run_id)
    rows = []
    for path in glob.glob(f'{REPO}/files/experiments/wandb/*/dataset_{dataset}/seed_pipeline_results.json'):
        if path.split(os.sep)[-3] not in ids:
            continue
        for run in json.load(open(path))['seed_runs']:
            ts = run['results'][0]['timestep_results']
            y = np.asarray(ts['y_true_all'], dtype=int)
            p = np.asarray(ts['y_preds_all'], dtype=int)
            info = ts.get('drift_info')
            onset = info['onset'] if info else None
            rows.append(dict(scenario=run['results'][0]['scenario'], model=model_name or run['model'],
                             seed=int(run['seed']), f1=_f1(y, p),
                             f1_post=_f1(y[onset:], p[onset:]) if onset is not None else np.nan))
    return pd.DataFrame(rows, columns=['scenario', 'model', 'seed', 'f1', 'f1_post'])


def load(refresh):
    keep = ['scenario', 'model', 'seed', 'accuracy', 'dp', 'eo',
            'post_drift_accuracy', 'post_drift_dp', 'post_drift_eo']
    parts, f1s = [], []
    for key, dataset in (('main_folktables', 'folktables'), ('main_adult', 'adult')):
        raw = arm_metrics(SWEEPS[key], refresh)
        raw = raw[raw.model.isin(['fado'] + BASELINES)]
        parts.append(raw[keep])
        f1s.append(local_f1(raw, dataset))
    for key, dataset in (('tuned_folktables', 'folktables'), ('tuned_adult', 'adult')):
        raw = arm_metrics(SWEEPS[key], refresh)
        raw = raw[raw.model == 'aranyani'].assign(model='tuned')
        parts.append(raw[keep])
        f1s.append(local_f1(raw, dataset, model_name='tuned'))
    data = pd.concat(parts, ignore_index=True)
    f1 = pd.concat(f1s, ignore_index=True).drop_duplicates(['scenario', 'model', 'seed'])
    # whole-stream F1 of the main runs whose folders are on another machine
    if os.path.isfile(F1_FALLBACK):
        fb = pd.read_csv(F1_FALLBACK)[['scenario', 'model', 'seed', 'f1']]
        fb = fb[fb.model.isin(['fado'] + BASELINES)]
        missing = fb.merge(f1[['scenario', 'model', 'seed']], how='left', indicator=True)
        missing = missing[missing._merge == 'left_only'].drop(columns='_merge')
        f1 = pd.concat([f1, missing.assign(f1_post=np.nan)], ignore_index=True)
    data = data.merge(f1, on=['scenario', 'model', 'seed'], how='left')
    data.loc[data.scenario == 'folktables', list(POST) + ['post_drift_accuracy']] = np.nan
    return data


# ----------------------------------------------------------------------------- tables
def _holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[k]))
        adj[k] = running
    return adj


def _stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def results_table(data):
    models = [m for m, _ in ROWS]
    cells, filled = {}, {}
    for sc, _ in BLOCKS:
        d = data[data.scenario == sc]
        by = {m: d[d.model == m].set_index('seed') for m in models}
        for col, higher in COLS:
            if sc == 'folktables' and col in POST:
                for m in models:
                    cells[(sc, m, col)] = 'n/a'
                continue
            have = [m for m in models if len(by[m]) and by[m][col].notna().sum() == N_SEEDS]
            if 'fado' not in have or not set(BASELINES) <= set(have):
                for m in models:
                    cells[(sc, m, col)] = ''
                continue
            means = {m: by[m][col].mean() for m in have}
            best = (max if higher else min)(means, key=means.get)
            pvals, favour = [], []
            for b in BASELINES:
                diff = (by['fado'][col] - by[b][col].reindex(by['fado'].index)).dropna()
                pvals.append(wilcoxon(diff).pvalue)
                favour.append((diff.mean() > 0) == higher)
            star = _stars(max(_holm(np.array(pvals)))) if all(favour) else ''
            for m in models:
                if m not in have:
                    cells[(sc, m, col)] = ''
                    continue
                v = f"{means[m]:.3f}\\pm {by[m][col].std(ddof=1):.3f}"
                if m == best:
                    v = f'\\mathbf{{{v}}}'
                if m == 'fado' and star:
                    v += f'\\sig{{{star}}}'
                if m == 'tuned':
                    diff = (by['fado'][col] - by['tuned'][col].reindex(by['fado'].index)).dropna()
                    if wilcoxon(diff).pvalue < 0.05:
                        v += r'^{\dagger}'
                cells[(sc, m, col)] = f'${v}$'
                filled[(sc, m)] = filled.get((sc, m), 0) + 1
    body = []
    for k, (sc, title) in enumerate(BLOCKS):
        if k:
            body.append(r'\midrule')
        body.append(r'\multicolumn{7}{@{}l}{{' + title + r'}} \\')
        for m, label in ROWS:
            body.append(' & '.join([label] + [cells[(sc, m, c)] for c, _ in COLS]) + r' \\')
    tex = '\n'.join([
        r'\begin{table*}[t]', r'\centering',
        r'\caption{F1, Demographic Parity, and Equalized Odds Over 30 Seeds, Whole Stream and After the Drift Onset$^{\mathrm{a}}$}',
        r'\label{tab:results}', r'\setlength{\tabcolsep}{4pt}', r'\footnotesize',
        r'\begin{tabular}{@{}l ccc ccc@{}}', r'\toprule',
        r'& \multicolumn{3}{c}{Whole stream} & \multicolumn{3}{c}{After onset$^{\mathrm{c}}$} \\',
        r'\cmidrule(lr){2-4} \cmidrule(l){5-7}',
        r'Method & F1$^{\mathrm{b}}$\,$\uparrow$ & DP\,$\downarrow$ & EO\,$\downarrow$ & F1$^{\mathrm{b}}$\,$\uparrow$ & DP\,$\downarrow$ & EO\,$\downarrow$ \\',
        r'\midrule'] + body + [
        r'\bottomrule', r'\end{tabular}',
        r'\tnote{$^{\mathrm{a}}$Mean $\pm$ standard deviation over 30 seeds. Bold marks the best mean per',
        r"metric within each block. Superscripts on \fado{} mark a paired Wilcoxon test, Holm-corrected, significant in \fado{}'s favor against each of \ara{}, \arf{}, and \rfr{}",
        r'($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$); columns with no marker have mixed per-baseline directions and are dissected in the text.',
        r'$^{\mathrm{b}}$F1 of the positive class (income $>$ \$50k), computed from the per-sample predictions (one confusion matrix per run and part of the stream). Empty cells: not yet computed.',
        r'$^{\mathrm{c}}$From the injected drift onset to the end of the test stream (the drifted part of the stream); DP and EO are the rolling-window values averaged over that part. \folk{} has no injected onset (n/a).',
        r"$^{\mathrm{d}}$\ara{} with a constant $\lambda$ chosen after the fact from \fado{}'s own runs, at least \fado{}'s mean $\lambda_t$ on that dataset ($0.5$ on \folk{}, $1.5$ on \adult{}). This gives the baseline an advantage it would not have in practice, the value of $\lambda$ leaked from \fado{}, so it is the strongest fixed-$\lambda$ \ara{} we can construct. It is pre-trained with $\lambda_{\mathrm{base}}=0.1$, like \fado{}, so both start from the same forest. $^{\dagger}$Significantly different from \fado{} (paired two-sided Wilcoxon test, $p<0.05$); bold shows which is better. Empty rows: runs not finished.}",
        r'\end{table*}'])
    return tex + '\n', filled


def detection_table(det, main_adult):
    fado = main_adult[main_adult.model == 'fado']
    det = det.copy()
    det['first'] = det[['acc_mtd', 'fair_mtd']].min(axis=1)
    det['either'] = (det.acc_tp > 0) | (det.fair_tp > 0)
    names = {'sim_x_permutations': r'x\_permutations', 'sim_y_swaps': r'y\_swaps',
             'sim_y_prior_skip': r'y\_prior\_skip', 'sim_x_exceed_skip': r'x\_exceed\_skip'}

    def rate(k, n):
        return f'{k}/{n} ({100 * k / n:.0f}\\%)'

    def delay(x):
        x = x.dropna()
        return f'${x.mean():.0f}$' if len(x) else '--'

    def row(label, d, f):
        n = len(d)
        return ' & '.join([label, rate(int(d.either.sum()), n), delay(d['first']),
                           rate(int((d.acc_tp > 0).sum()), n), delay(d.acc_mtd),
                           rate(int((d.fair_tp > 0).sum()), n), delay(d.fair_mtd),
                           f'${f.accuracy_false_alarms.mean():.2f}$',
                           f'${f.fairness_false_alarms.mean():.2f}$']) + r' \\'
    rows = []
    for sc in ADULT:
        d, f = det[det.scenario == sc], fado[fado.scenario == sc]
        assert len(d) == N_SEEDS and len(f) == N_SEEDS, (sc, len(d), len(f))
        rows.append(row(names[sc], d, f))
    rows += [r'\midrule', row('All', det, fado)]
    tex = '\n'.join([
        r'\begin{table*}[t]', r'\centering',
        r'\caption{Detection of the Injected Drift by \fado{} on \adult{}, 30 Seeds per Scenario$^{\mathrm{a}}$}',
        r'\label{tab:detection}', r'\setlength{\tabcolsep}{5pt}', r'\footnotesize',
        r'\begin{tabular}{@{}l cc cc cc cc@{}}', r'\toprule',
        r'& \multicolumn{2}{c}{Either detector} & \multicolumn{2}{c}{Accuracy detector} & \multicolumn{2}{c}{Fairness monitor} & \multicolumn{2}{c}{False alarms per run$^{\mathrm{c}}$} \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(l){8-9}',
        r'Scenario & Detected & Delay$^{\mathrm{b}}$ & Detected & Delay$^{\mathrm{b}}$ & Detected & Delay$^{\mathrm{b}}$ & Accuracy & Fairness \\',
        r'\midrule'] + rows + [
        r'\bottomrule', r'\end{tabular}',
        r'\tnote{$^{\mathrm{a}}$An onset counts as detected when the detector raises an alarm within $1{,}000$ samples after it, the length of one rolling window: a window-based detector cannot register a change before its window has refilled, so a later alarm is no longer attributed to the injected drift. \emph{Either detector} counts a run once if the accuracy detector or the fairness monitor detects the onset; \folk{} has no injected onset and is not scored.',
        r'$^{\mathrm{b}}$Mean number of samples from the onset to the first alarm within that window, over the detected runs only; by construction it is at most $1{,}000$, so it must be read together with the detection rate.',
        r"$^{\mathrm{c}}$Alarms not matched to the onset, averaged over the 30 runs; every alarm of the fairness monitor, matched or not, resets the penalty's running statistics.}",
        r'\end{table*}'])
    return tex + '\n'


def detection_split(det, main_adult):
    """FADO - Aranyani after the onset, split by whether FADO detected the drift."""
    f = main_adult[main_adult.model == 'fado'].set_index(['scenario', 'seed'])
    a = main_adult[main_adult.model == 'aranyani'].set_index(['scenario', 'seed'])
    d = det.set_index(['scenario', 'seed']).copy()
    d['detected'] = (d.acc_tp > 0) | (d.fair_tp > 0)
    for m in ('post_drift_dp', 'post_drift_eo', 'post_drift_accuracy'):
        d['d_' + m] = f[m] - a[m]
    d = d.reset_index()
    mets = ['d_post_drift_dp', 'd_post_drift_eo', 'd_post_drift_accuracy']

    def cell(x, m):
        if not len(x):
            return '–'
        v = f'{x[m].mean():+.3f}'
        if len(x) >= 6:
            v += _stars(wilcoxon(x[m]).pvalue).replace('*', '\\*')
        return v

    def between(y, n):
        if len(y) < 3 or len(n) < 3:
            return '–'
        p = mannwhitneyu(y.d_post_drift_dp, n.d_post_drift_dp).pvalue
        return f'{p:.2f}' if p >= 0.01 else '< 0.01'
    names = {'sim_x_permutations': 'x_permutations', 'sim_y_swaps': 'y_swaps',
             'sim_y_prior_skip': 'y_prior_skip', 'sim_x_exceed_skip': 'x_exceed_skip'}
    lines = ['| Scenario | Runs | ΔDP | ΔEO | ΔAcc. | Runs | ΔDP | ΔEO | ΔAcc. | Groups differ, p (ΔDP) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for key, sub in [(sc, d[d.scenario == sc]) for sc in ADULT] + [('All', d)]:
        y, n = sub[sub.detected], sub[~sub.detected]
        label = '**All**' if key == 'All' else names[key]
        lines.append('| ' + ' | '.join([label, str(len(y))] + [cell(y, m) for m in mets] + [str(len(n))]
                                       + [cell(n, m) for m in mets] + [between(y, n)]) + ' |')
    md = '\n'.join([
        '# FADO minus Aranyani after the drift onset, by whether FADO detected the drift', '',
        'Supplementary result for the ICDE submission (Adult, 4 injected drifts, 30 seeds per',
        'scenario, the runs of Table III). Generated by `DOCS/paper_icde/scripts/make_tables.py`;',
        'kept out of the paper because ICDE does not allow appendices.', '',
        'The first group of columns covers runs where the onset was detected; the second, runs',
        'where it was not.', ''] + lines + ['',
        '**How to read it**', '',
        '- **Δ** is FADO minus Aranyani (λ = 0.1) over the part of the stream after the injected',
        '  onset, paired by seed and averaged over the runs of each group. Negative ΔDP and ΔEO,',
        '  and positive ΔAcc., favour FADO.',
        '- **Detected** means either FADO detector (the accuracy detector or the fairness monitor)',
        '  raised an alarm within 1,000 samples after the onset, as in Table IV of the paper.',
        '- **Stars** come from a two-sided Wilcoxon signed-rank test of Δ against zero',
        '  (\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001). They are not corrected for multiple',
        '  comparisons. Groups with fewer than six runs are not tested.',
        '- **Last column:** a two-sided Mann–Whitney test of ΔDP between detected and undetected',
        '  runs. It is not computed when a group has fewer than three runs.',
        '- **Caveat:** runs are not assigned to the groups at random. A drift that is easier to',
        '  detect may also be stronger, so this shows an association, not the effect of detection.', '',
        '**Data.** Per-run values are in `adult_detected_vs_undetected.csv`, next to this file.', ''])
    return md, d


SIG_METRICS = [('accuracy', 'Accuracy', True), ('f1', 'F1', True), ('dp', 'DP', False), ('eo', 'EO', False),
               ('post_drift_accuracy', 'Accuracy after onset', True), ('f1_post', 'F1 after onset', True),
               ('post_drift_dp', 'DP after onset', False), ('post_drift_eo', 'EO after onset', False)]


def significance_report(data):
    """FADO against each baseline, per block and metric: paired Wilcoxon and Welch t-test,
    each Holm-corrected across the baselines of that comparison set."""
    sets = [('Against the published baselines', BASELINES), ('Against the tuned-lambda Aranyani', ['tuned'])]
    out = ['# Significance of FADO against each baseline', '',
           'Generated by `DOCS/paper_icde/scripts/make_tables.py`. For every block of Table III and every',
           'metric: the mean difference FADO minus baseline over the paired seeds, the number of seeds on',
           'which FADO is better, and the p-values of the two-sided paired Wilcoxon signed-rank test and the',
           "two-sided Welch t-test. Both are Holm-corrected across the baselines of the same set; the",
           'tuned-lambda Aranyani is tested on its own. Only metrics with all 30 seeds for both methods are',
           'tested.', '']
    for title, baselines in sets:
        out += [f'## {title}', '']
        for sc, block in BLOCKS:
            d = data[data.scenario == sc]
            fado = d[d.model == 'fado'].set_index('seed')
            lines = []
            for col, label, higher in SIG_METRICS:
                tests = []
                for b in baselines:
                    other = d[d.model == b].set_index('seed')
                    if col not in fado or fado[col].notna().sum() != N_SEEDS or \
                            len(other) == 0 or other[col].notna().sum() != N_SEEDS:
                        continue
                    diff = (fado[col] - other[col].reindex(fado.index)).dropna()
                    better = int(((diff > 0) if higher else (diff < 0)).sum())
                    tests.append((b, diff.mean(), better, len(diff), wilcoxon(diff).pvalue,
                                  ttest_ind(fado[col], other[col], equal_var=False).pvalue))
                if not tests:
                    continue
                pw = _holm(np.array([t[4] for t in tests]))
                pt = _holm(np.array([t[5] for t in tests]))
                for (b, mean, better, n, _, _), w, t in zip(tests, pw, pt):
                    lines.append(f'| {label} | {b} | {mean:+.4f} | {better}/{n} | {w:.1e} | {t:.1e} |')
            if lines:
                name = block.replace('\\folk{}', 'Folktables').replace('\\adult{}', 'Adult').replace('\\_', '_')
                out += [f'### {name}', '',
                        '| Metric | Baseline | FADO − baseline | FADO better | Wilcoxon p (Holm) | Welch p (Holm) |',
                        '|---|---|---|---|---|---|'] + lines + ['']
    return '\n'.join(out)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--refresh', action='store_true', help='download everything from W&B again')
    args = parser.parse_args()

    data = load(args.refresh)
    main_adult = arm_metrics(SWEEPS['main_adult'], False)
    det = detection(SWEEPS['main_adult'], args.refresh)

    os.makedirs(os.path.join(PAPER, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(PAPER, 'supplementary'), exist_ok=True)
    results, filled = results_table(data)
    open(os.path.join(PAPER, 'tables', 'results.tex'), 'w').write(results)
    open(os.path.join(PAPER, 'tables', 'detection.tex'), 'w').write(detection_table(det, main_adult))
    open(os.path.join(PAPER, 'supplementary', 'significance.md'), 'w').write(significance_report(data))
    md, split = detection_split(det, main_adult)
    open(os.path.join(PAPER, 'supplementary', 'detection_split.md'), 'w').write(md)
    split.to_csv(os.path.join(PAPER, 'supplementary', 'adult_detected_vs_undetected.csv'), index=False)

    print('Table III filled cells per row (of 3 for Folktables, 6 for Adult):')
    for sc, _ in BLOCKS:
        print(f'  {sc:20s} ' + '  '.join(f'{m}={filled.get((sc, m), 0)}' for m, _ in ROWS))
    print(f'written: {os.path.relpath(os.path.join(PAPER, "tables"), REPO)}/results.tex, detection.tex; '
          f'{os.path.relpath(os.path.join(PAPER, "supplementary"), REPO)}/detection_split.md')


if __name__ == '__main__':
    main()
