"""Why FADO's demographic parity sits above its target epsilon: the dual-ascent bound, run by run.

    PYTHONHASHSEED=0 python DOCS/paper_icde/scripts/dual_bound.py

Projected dual ascent, mu <- clip(mu + eta (v_t - eps), 0, lambda_max), telescopes into

    (1/T) sum_t (v_t - eps)  <=  mu_T / (eta T)          (equality while the clip never binds)

for every one-sided constraint: on a finite stream the average violation is bounded by the final
price over (dual step x length), not by zero. v_t is unbiased for the disparity D_a (Prop. 1), so
the left side estimates the stream's average disparity minus eps.

For every FADO-family run (any arm with the adaptive constraint) with local per-sample results,
this script regenerates the stream's groups (runs.stream_groups), replays the controller from
the stored predictions, checks the replay against the stored lambda_t, and reports per period
(before the onset on Adult; the whole stream on Folktables):

    rolling DP     mean of the W = 1000 rolling DP that the paper's tables average
    stream DP      DP of all the period's predictions at once, (max_a |p_a - mean p|)
    avg violation  (1/T) sum_t (v_t - eps) of the most violated constraint, plus eps
    bound          eps + mu_T / (eta T) of that constraint
    lambda_T       lambda_t at the end of the period

Writes DOCS/paper_icde/supplementary/dual_bound.md. Covers the sweeps of make_tables.SWEEPS whose
run folders are on this machine; the report lists which.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402
from runs import local_runs, stream_groups  # noqa: E402

sys.path.insert(0, mt.REPO)
from src.models.forest.lambda_controller import LambdaController  # noqa: E402

OUT = os.path.join(mt.PAPER, 'supplementary', 'dual_bound.md')
KEYS = ['main_adult', 'main_folktables', 'dualstep_adult', 'ablation_adult', 'ablation_folktables',
        'intersectional_folktables']
NAMES = {'sim_x_permutations': 'x_permutations', 'sim_y_swaps': 'y_swaps', 'sim_y_prior_skip': 'y_prior_skip',
         'sim_x_exceed_skip': 'x_exceed_skip', 'folktables': 'Folktables'}


def stream_dp(pred, a, k):
    rates = np.array([pred[a == g].mean() for g in range(k)])
    return np.max(np.abs(rates - rates.mean()))


def replay(run, a):
    """Replay the controller: per-step lambda, and per constraint the cumulative violation and mu."""
    c = run['lambda_controller']
    ctrl = LambdaController(c['lambda_base'], epsilon=c['epsilon'], eta=c['eta'],
                            lambda_max=c['lambda_max'], num_groups=c['num_groups'])
    pred = np.asarray(run['y_preds_all'], dtype=int)
    k = c['num_groups']
    lam, cum, mus = np.empty(len(pred)), np.zeros((len(pred), k, 2)), np.empty((len(pred), k, 2))
    total = np.zeros((k, 2))
    for t, (p, g) in enumerate(zip(pred, a)):
        observed = ctrl.signal.observe(int(p), int(g))
        if observed is not None:  # the same update as LambdaController.update
            v = np.array([observed[f'group_{i}'] for i in range(k)])
            violation = np.stack([v, -v], axis=1) - ctrl.epsilon
            ctrl.mu = np.clip(ctrl.mu + ctrl.eta * violation, 0.0, ctrl.lambda_max)
            total += violation
        lam[t], cum[t], mus[t] = ctrl.value, total, ctrl.mu
    return lam, cum, mus


def analyse():
    rows, covered = [], {}
    for key in KEYS:
        sweeps = mt.SWEEPS.get(key) or []
        runs = [r for r in local_runs(set(sweeps)) if r.get('lambda_controller') and r.get('lambda')]
        covered[key] = sorted({r['sweep'] for r in runs})
        for run in runs:
            model, seed = run['model'], run['seed']
            a = stream_groups(run)
            lam, cum, mus = replay(run, a)
            c = run['lambda_controller']
            err = float(np.max(np.abs(lam - np.asarray(run['lambda'], dtype=float))))
            pred = np.asarray(run['y_preds_all'], dtype=int)
            onset = (run.get('drift_info') or {}).get('onset')
            periods = [('before onset', onset)] if onset else []
            periods.append(('whole stream', len(pred)))
            for period, end in periods:
                worst = np.unravel_index(np.argmax(cum[end - 1]), cum[end - 1].shape)
                rows.append(dict(key=key, eta=c['eta'], scenario=NAMES.get(run['scenario'], run['scenario']),
                                 model=model, seed=seed, period=period, T=end, replay_error=err,
                                 rolling_dp=float(np.mean(run['dp'][:end])),
                                 stream_dp=float(stream_dp(pred[:end], a[:end], c['num_groups'])),
                                 avg_violation=c['epsilon'] + cum[end - 1][worst] / end,
                                 bound=c['epsilon'] + mus[end - 1][worst] / (c['eta'] * end),
                                 lambda_T=float(lam[end - 1]),
                                 frac_below_eps=float(np.mean(np.asarray(run['dp'][:end]) <= c['epsilon']))))
    return pd.DataFrame(rows), covered


def main():
    df, covered = analyse()
    cols = ['rolling_dp', 'stream_dp', 'avg_violation', 'bound', 'lambda_T', 'frac_below_eps', 'T']
    lines = ['# Demographic parity against the target epsilon: the dual-ascent bound', '',
             'Generated by `DOCS/paper_icde/scripts/dual_bound.py` (see its docstring for the derivation).',
             'Means over the seeds of each row; ε = 0.05. "avg violation" and "bound" are for the most violated',
             'one-sided constraint; the bound holds with equality while the projection at zero never binds.', '',
             '**Coverage (local run folders):** ' + '; '.join(f'{k}: {", ".join(v) or "none"}' for k, v in covered.items()),
             '']
    if not len(df):
        lines.append('No local runs with the adaptive constraint.')
    else:
        lines.append(f'**Replay check:** largest |replayed λ_t − stored λ_t| over all runs and steps = '
                     f'{df.replay_error.max():.1e} ({df.seed.nunique()} seeds, {len(df)} run-periods).')
        lines.append('')
        g = df.groupby(['key', 'eta', 'scenario', 'model', 'period'])
        table = g[cols].mean().join(g.size().rename('seeds')).reset_index()
        lines += ['| Sweep | η | Scenario | Arm | Period | Seeds | Rolling DP | Stream DP | Avg violation + ε | Bound | λ at end | Time rolling DP ≤ ε | T |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for _, r in table.iterrows():
            lines.append(f'| {r.key} | {r.eta:g} | {r.scenario} | {r.model} | {r.period} | {r.seeds} | {r.rolling_dp:.4f} | '
                         f'{r.stream_dp:.4f} | {r.avg_violation:.4f} | {r.bound:.4f} | {r.lambda_T:.2f} | '
                         f'{100 * r.frac_below_eps:.0f}% | {r["T"]:.0f} |')
    lines.append('')
    open(OUT, 'w').write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'-> {os.path.relpath(OUT, mt.REPO)}')


if __name__ == '__main__':
    main()
