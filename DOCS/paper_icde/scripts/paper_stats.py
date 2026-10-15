"""Paper numbers (Table III): whole-stream mean ± std over seeds, and FADO vs
each baseline under the paired Wilcoxon and the Welch t-test, both
Holm-corrected within each metric across the baselines.

    python paper_stats.py adult                # local files/experiments/wandb/*/dataset_adult
    python paper_stats.py <raw.csv> [...]      # one or more fetch_wandb_arm_metrics raw.csv

With more than one scenario, a "mean of N" block follows: each seed is first
averaged over the scenarios, keeping only seeds where every model finished
every scenario.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
REF = 'fado'
BASELINES = ['aranyani', 'arf', 'rfr']
METRICS = [('accuracy', True), ('dp', False), ('eo', False)]
EXTRA = ['stream_final_lambda', 'post_drift_lambda', 'fairness_resets',
         'fairness_detections', 'fairness_true_positives', 'fairness_false_alarms',
         'accuracy_detections', 'accuracy_true_positives', 'accuracy_false_alarms',
         'ms_per_sample', 'pretrain_wall_seconds']


def load_adult():
    rows = []
    for run in glob.glob(f'{REPO}/files/experiments/wandb/*/dataset_adult'):
        try:
            cfg = json.load(open(f'{run}/seed_pipeline_results.json'))['config']
        except (OSError, KeyError, ValueError):
            continue  # unfinished run
        for path in glob.glob(f'{run}/model_*/seed_*/results.json'):
            seed = int(os.path.basename(os.path.dirname(path)).split('_')[1])
            for r in json.load(open(path)):
                if r.get('error'):
                    continue
                rows.append({'seed': seed, 'model': r['model'], 'scenario': r['scenario'],
                             'sweep': cfg.get('wandb_sweep_id'),
                             **{k: r.get(k) for k, _ in METRICS},
                             **{k: r.get(k) for k in EXTRA}})
    return pd.DataFrame(rows)


def holm(pvals):
    order = np.argsort(pvals)
    out, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        out[i] = running
    return out


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else ''


def report(df, label):
    df = df.copy()
    for c in [m for m, _ in METRICS] + EXTRA:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    models = [m for m in [REF] + BASELINES if m in set(df.model)]
    # keep seeds where every model is present
    have = df.groupby('seed')['model'].agg(set)
    seeds = sorted(s for s, m in have.items() if set(models) <= m)
    df = df[df.seed.isin(seeds) & df.model.isin(models)]
    print(f'\n##### {label}: {len(seeds)} complete seeds, models {models}')
    summary = df.groupby('model')[[m for m, _ in METRICS] + [e for e in EXTRA if e in df]].agg(['mean', 'std'])
    for model in models:
        cells = []
        for m, _ in METRICS:
            cells.append(f"{m}={summary.loc[model, (m, 'mean')]:.4f}±{summary.loc[model, (m, 'std')]:.4f}")
        print(f'  {model:9s} ' + ' '.join(cells))
    print('  extras (mean):')
    for model in models:
        ex = [f"{e}={summary.loc[model, (e, 'mean')]:.3g}" for e in EXTRA
              if e in df and not np.isnan(summary.loc[model, (e, 'mean')])]
        print(f'    {model:9s} ' + ' '.join(ex))
    print('  FADO vs baseline, paired Wilcoxon | Welch (each Holm across baselines):')
    for m, higher in METRICS:
        wide = df.pivot_table(index='seed', columns='model', values=m)
        ps, ws, deltas, wins = [], [], [], []
        for b in BASELINES:
            d = (wide[REF] - wide[b]).dropna()
            ps.append(stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0)
            ws.append(stats.ttest_ind(wide[REF].dropna(), wide[b].dropna(), equal_var=False).pvalue)
            deltas.append(d.mean())
            better = (d > 0) if higher else (d < 0)
            wins.append(f'{int(better.sum())}/{len(d)}')
        adj, adj_w = holm(np.array(ps)), holm(np.array(ws))
        print(f'    {m:9s} ' + '  '.join(
            f'{b}: Δ{dl:+.4f} p={p:.2g}{stars(p)} | {pw:.2g}{stars(pw)} win {w}'
            for b, dl, p, pw, w in zip(BASELINES, deltas, adj, adj_w, wins)))


def mean_over_scenarios(df):
    """Average each (seed, model) over the scenarios ("mean of 4" in Table III)."""
    cols = [c for c in [m for m, _ in METRICS] + EXTRA if c in df]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    n = df.scenario.nunique()
    cells = df.groupby(['seed', 'model']).scenario.nunique()
    keep = cells[cells == n].reset_index()[['seed', 'model']]
    df = df.merge(keep, on=['seed', 'model'])
    out = df.groupby(['seed', 'model'], as_index=False)[cols].mean()
    out['scenario'] = f'mean of {n}'
    return out


def report_all(df, label):
    for scenario, block in df.groupby('scenario'):
        report(block, f'{label} / {scenario}')
    if df.scenario.nunique() > 1:
        avg = mean_over_scenarios(df.copy())
        report(avg, f'{label} / {avg.scenario.iloc[0]}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')  # ± and Δ on a Windows console
    args = sys.argv[1:] or ['adult']
    if args == ['adult']:
        df = load_adult()
        print(df.groupby(['sweep', 'scenario', 'model']).seed.nunique().to_string())
        report_all(df, 'adult')
    else:
        df = pd.concat([pd.read_csv(a) for a in args], ignore_index=True)
        # a seed re-run in a second sweep counts once
        df = df.drop_duplicates(['seed', 'scenario', 'model'], keep='last')
        report_all(df, ', '.join(os.path.basename(os.path.dirname(a)) for a in args))
