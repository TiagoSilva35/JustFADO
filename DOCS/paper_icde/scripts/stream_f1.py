"""Whole-stream classification metrics from the per-sample predictions.

Reads every local W&B run folder (files/experiments/wandb/*/dataset_<name>),
builds one confusion matrix per (seed, model, scenario) from
timestep_results.y_preds_all / y_true_all, and appends the rows to
files/experiments/extra_metrics/per_seed.csv (same columns as the rows already
there). Then prints mean ± std per model and the paired Wilcoxon of FADO
against each baseline, Holm-corrected across the baselines.

    python DOCS/paper_icde/scripts/stream_f1.py adult sim_x_permutations:uwhcwsdg sim_y_swaps:bi6frxi3

Each scenario is given as <scenario>:<sweep id>, so stray runs of the same
scenario from other sweeps are ignored.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(REPO, 'files', 'experiments', 'extra_metrics', 'per_seed.csv')
REF, BASELINES = 'fado', ['aranyani', 'arf', 'rfr']
COLUMNS = ['dataset', 'scenario', 'model', 'seed', 'table_acc', 'acc_stream', 'bal_acc',
           'kappa_m', 'precision', 'recall', 'f1', 'pos_rate', 'maj_acc']


def stream_row(dataset, scenario, model, seed, result):
    steps = result['timestep_results']
    pred = np.asarray(steps['y_preds_all'], dtype=int)
    true = np.asarray(steps['y_true_all'], dtype=int)
    n = min(len(pred), len(true))
    pred, true = pred[:n], true[:n]
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    acc = (tp + tn) / n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pos_rate = (tp + fn) / n
    maj_acc = max(pos_rate, 1 - pos_rate)
    return {'dataset': dataset, 'scenario': scenario, 'model': model, 'seed': int(seed),
            'table_acc': result.get('accuracy'), 'acc_stream': acc,
            'bal_acc': (recall + specificity) / 2,
            'kappa_m': (acc - maj_acc) / (1 - maj_acc) if maj_acc < 1 else 0.0,
            'precision': precision, 'recall': recall, 'f1': f1,
            'pos_rate': pos_rate, 'maj_acc': maj_acc}


def collect(dataset, wanted):
    rows = []
    for run in sorted(glob.glob(f'{REPO}/files/experiments/wandb/*/dataset_{dataset}')):
        try:
            payload = json.load(open(f'{run}/seed_pipeline_results.json'))
        except (OSError, ValueError):
            continue  # unfinished run
        sweep = payload['config'].get('wandb_sweep_id')
        for seed_run in payload['seed_runs']:
            for result in seed_run['results']:
                if result.get('error') or wanted.get(result['scenario']) != sweep:
                    continue
                rows.append(stream_row(dataset, result['scenario'], seed_run['model'],
                                       seed_run['seed'], result))
    return pd.DataFrame(rows, columns=COLUMNS)


def holm(pvals):
    order = np.argsort(pvals)
    out, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        out[i] = running
    return out


def report(df):
    for scenario, block in df.groupby('scenario'):
        wide = block.pivot_table(index='seed', columns='model', values='f1')
        wide = wide.dropna()
        print(f'\n{scenario}: {len(wide)} complete seeds')
        for model in [REF] + BASELINES:
            print(f'  {model:9s} f1={wide[model].mean():.4f}±{wide[model].std():.4f}')
        ps = [stats.wilcoxon(wide[REF] - wide[b]).pvalue for b in BASELINES]
        for b, p, d in zip(BASELINES, holm(np.array(ps)), [(wide[REF] - wide[b]).mean() for b in BASELINES]):
            print(f'  fado vs {b:9s} Δ{d:+.4f} Holm p={p:.2g}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    dataset = sys.argv[1]
    wanted = dict(arg.split(':') for arg in sys.argv[2:])
    new = collect(dataset, wanted)
    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=COLUMNS)
    key = ['dataset', 'scenario', 'model', 'seed']
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates(key, keep='last')
    merged.to_csv(OUT, index=False)
    print(f'{len(new)} rows from local runs; {len(merged)} rows in {OUT}')
    report(new)
