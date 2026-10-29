"""Build the paper's result tables from W&B and the local run folders.

Two steps, so every machine can contribute what only it has:

    python DOCS/paper_icde/scripts/make_tables.py crawl   # any machine, no W&B needed
    python DOCS/paper_icde/scripts/make_tables.py build   # any machine (default command)
    python DOCS/paper_icde/scripts/make_tables.py all     # crawl, then build
    ... build --refresh                                    # download W&B data again

crawl   scans this machine's run folders (files/experiments/wandb/<run id>/), keeps the runs of
        the sweeps in SWEEPS, computes F1 (whole stream and after the onset) from the
        per-sample predictions, and upserts one row per (sweep, run, model, seed) into the
        tracked store DOCS/paper_icde/data/f1_store.csv. Rows from other machines are kept.
        Commit and push the store; on another machine, pull and build. build uses only the
        store rows of runs that W&B lists as finished runs of those sweeps.
build   merges the W&B metrics with the F1 store and regenerates every output below. The
        result depends only on the store and W&B, so it is the same on every machine.

Writes (paper.tex \\input's the two .tex files, so the text is never touched):
    DOCS/paper_icde/tables/results.tex          Table III: F1 / DP / EO, whole stream and after onset
    DOCS/paper_icde/tables/detection.tex        Table IV: drift detection by FADO on Adult
    DOCS/paper_icde/supplementary/detection_split.{md,csv}
                                                FADO - Aranyani after onset, detected vs undetected
    DOCS/paper_icde/supplementary/significance.md
                                                FADO vs each method: every test behind the marks of
                                                Table III, plus the Welch t-test (the paper's "Welch
                                                results in the repository")

Sources:
    * W&B per-arm tables (`arm_metrics`) and run summaries of the sweeps in SWEEPS,
      cached in files/experiments/wandb_arm_metrics/<sweeps>/ (raw.csv, detection.csv).
      Only finished runs are used; re-run with --refresh while sweeps are still running.
    * F1 comes from DOCS/paper_icde/data/f1_store.csv, filled by `crawl` on each machine.
      For main runs not in the store yet, the whole-stream F1 falls back to
      files/experiments/extra_metrics/per_seed.csv (written earlier by stream_f1.py).

Rules (same as the tables already in the paper):
    * a cell is filled only when every seed (30) of that method has the value; otherwise
      it is left empty;
    * bold = best mean of the column in the block (only when every method has the column);
    * FADO is the reference: every other method's cell is marked with a bullet (FADO
      significantly better) or a circle (FADO significantly worse) by a two-sided paired
      Wilcoxon test, Holm-corrected across the four methods compared with FADO in that
      column of that block (alpha 0.05);
    * the overall block on top counts, per method, FADO's significant wins / no significant
      differences / significant losses over every cell of the table.
"""
import argparse
import glob
import json
import os
import socket
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PAPER = os.path.join(REPO, 'DOCS', 'paper_icde')
CACHE = os.path.join(REPO, 'files', 'experiments', 'wandb_arm_metrics')
F1_FALLBACK = os.path.join(REPO, 'files', 'experiments', 'extra_metrics', 'per_seed.csv')
F1_STORE = os.path.join(PAPER, 'data', 'f1_store.csv')
STORE_COLS = ['dataset', 'scenario', 'model', 'seed', 'sweep', 'run_id', 'f1', 'f1_post', 'host', 'updated']
ENTITY, PROJECT = 'tjcsilva04-universidade-de-coimbra', 'fado-ablations'
N_SEEDS = 30

# Which W&B sweeps feed the tables. Edit here when a sweep is re-run.
SWEEPS = {
    'main_folktables': ['jo5dqqhy'],                          # FADO, Aranyani, ARF, RFR
    'main_adult': ['uwhcwsdg', 'bi6frxi3', 'l8c4i60o'],       # same, 4 injected drifts
    'tuned_folktables': ['9ulal3qn'],                         # Aranyani, lambda 0.5, pre-train 0.1
    'tuned_adult': ['rm2glvmh', 'wqab0alm', 'pzbwmd4e'],      # Aranyani, lambda 1.5, pre-train 0.1
    # Experiments added after review (src/configs/sweeps/sweep_<name>.yaml). Put each sweep id
    # here once launch.sh has registered it; until then its report says it is pending.
    'ablation_adult': [],             # fado_dual, fado_lambda_only, fado_no_reset; 4 drifts
    'ablation_folktables': [],        # the same three arms
    'dualstep_adult': [],             # FADO with lambda_dual_lr 0.03 and 0.1
    'intersectional_folktables': [],  # FADO, Aranyani, ARF on sex x race (4 groups)
}
BLOCKS = [  # (scenario key, block title in the table); the four drifts are injected into Adult
    ('folktables', r'\folk{}'),
    ('sim_x_permutations', r'x\_permutations'),
    ('sim_y_swaps', r'y\_swaps'),
    ('sim_y_prior_skip', r'y\_prior\_skip'),
    ('sim_x_exceed_skip', r'x\_exceed\_skip'),
]
ADULT = [sc for sc, _ in BLOCKS if sc != 'folktables']
ROWS = [('fado', r'\fado{}'), ('aranyani', r'\ara{}'), ('tuned', r'\ara{}, leaked $\lambda$'),
        ('arf', r'\arf{}'), ('rfr', r'\rfr{}')]
BASELINES = ['aranyani', 'arf', 'rfr']
COMPARATORS = ['aranyani', 'tuned', 'arf', 'rfr']  # every method FADO is tested against
NAMES = {'aranyani': 'Aranyani', 'tuned': 'Aranyani, leaked λ', 'arf': 'ARF', 'rfr': 'RFR'}
ALPHA = 0.05
# (column, higher is better)
COLS = [('f1', True), ('dp', False), ('eo', False),
        ('f1_post', True), ('post_drift_dp', False), ('post_drift_eo', False)]
POST = {'f1_post', 'post_drift_dp', 'post_drift_eo', 'post_drift_accuracy'}


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


def _sweep_roles():
    """{sweep id: (dataset, role)}; role 'main' keeps the model names, 'tuned' maps aranyani."""
    roles = {}
    for key, sweeps in SWEEPS.items():
        role, dataset = key.split('_', 1)
        for sid in sweeps:
            roles[sid] = (dataset, role)
    return roles


def crawl():
    """Compute F1 from this machine's run folders and upsert it into the shared store."""
    roles, host = _sweep_roles(), socket.gethostname()
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
    rows = []
    for path in glob.glob(os.path.join(REPO, 'files', 'experiments', 'wandb', '*', 'dataset_*',
                                       'seed_pipeline_results.json')):
        data = json.load(open(path))
        cfg = data.get('config', {})
        sweep = cfg.get('wandb_sweep_id')
        if sweep not in roles:
            continue
        dataset, role = roles[sweep]
        run_id = cfg.get('wandb_run_id') or path.split(os.sep)[-3]
        for run in data['seed_runs']:
            model = run['model']
            if role == 'tuned':
                if model != 'aranyani':
                    continue
                model = 'tuned'
            ts = run['results'][0]['timestep_results']
            y = np.asarray(ts['y_true_all'], dtype=int)
            p = np.asarray(ts['y_preds_all'], dtype=int)
            info = ts.get('drift_info')
            onset = info['onset'] if info else None
            rows.append(dict(dataset=dataset, scenario=run['results'][0]['scenario'], model=model,
                             seed=int(run['seed']), sweep=sweep, run_id=run_id, f1=_f1(y, p),
                             f1_post=_f1(y[onset:], p[onset:]) if onset is not None else np.nan,
                             host=host, updated=now))
    found = pd.DataFrame(rows, columns=STORE_COLS)
    key = ['sweep', 'run_id', 'model', 'seed']
    store = pd.read_csv(F1_STORE) if os.path.isfile(F1_STORE) else pd.DataFrame(columns=STORE_COLS)
    merged = found.merge(store[key + ['f1', 'f1_post']], on=key, how='left', suffixes=('', '_old'))
    merged['f1_old'] = pd.to_numeric(merged.f1_old, errors='coerce')
    new = merged.f1_old.isna().sum()
    changed = ((merged.f1_old.notna()) & ~np.isclose(merged.f1, merged.f1_old, equal_nan=True)).sum()
    kept = store.merge(found[key], on=key, how='left', indicator=True)
    kept = kept[kept._merge == 'left_only'].drop(columns='_merge')
    out = pd.concat([kept, found], ignore_index=True)[STORE_COLS]
    out = out.sort_values(['dataset', 'scenario', 'model', 'seed', 'sweep', 'run_id']).reset_index(drop=True)
    os.makedirs(os.path.dirname(F1_STORE), exist_ok=True)
    out.to_csv(F1_STORE, index=False)
    print(f'[crawl] {host}: {len(found)} rows from local folders ({new} new, {changed} changed); '
          f'store now {len(out)} rows -> {os.path.relpath(F1_STORE, REPO)}')
    cover = out.groupby(['scenario', 'model']).seed.nunique().unstack(fill_value=0)
    print(cover.to_string())


def store_f1(valid_runs):
    """F1 rows (scenario, model, seed, f1, f1_post) from the shared store, restricted to the
    runs W&B lists as finished runs of the configured sweeps. Store rows of other runs (for
    example a run deleted from W&B whose folder survived on a machine) are ignored."""
    if not os.path.isfile(F1_STORE):
        return pd.DataFrame(columns=['scenario', 'model', 'seed', 'f1', 'f1_post'])
    store = pd.read_csv(F1_STORE)
    ignored = store[~store.run_id.isin(valid_runs)]
    if len(ignored):
        print(f'[f1] ignoring {len(ignored)} store rows of {ignored.run_id.nunique()} run(s) that are not '
              f'finished runs of the configured sweeps in W&B: {", ".join(sorted(ignored.run_id.unique()))}')
    store = store[store.run_id.isin(valid_runs)]
    dup = store.duplicated(['scenario', 'model', 'seed'], keep=False)
    if dup.any():
        print(f'[warn] {dup.sum()} store rows share (scenario, model, seed) across runs; using the latest')
        store = store.sort_values('updated').drop_duplicates(['scenario', 'model', 'seed'], keep='last')
    return store[['scenario', 'model', 'seed', 'f1', 'f1_post']]


def load(refresh):
    keep = ['scenario', 'model', 'seed', 'accuracy', 'dp', 'eo',
            'post_drift_accuracy', 'post_drift_dp', 'post_drift_eo', 'stream_final_lambda', 'post_drift_lambda']
    parts, valid_runs = [], set()
    for key in ('main_folktables', 'main_adult'):
        raw = arm_metrics(SWEEPS[key], refresh)
        parts.append(raw[raw.model.isin(['fado'] + BASELINES)][keep])
        valid_runs |= set(raw.run_id)
    for key in ('tuned_folktables', 'tuned_adult'):
        raw = arm_metrics(SWEEPS[key], refresh)
        parts.append(raw[raw.model == 'aranyani'].assign(model='tuned')[keep])
        valid_runs |= set(raw.run_id)
    data = pd.concat(parts, ignore_index=True)
    f1 = store_f1(valid_runs)
    # whole-stream F1 of the main runs not crawled yet (computed earlier by stream_f1.py)
    if os.path.isfile(F1_FALLBACK):
        fb = pd.read_csv(F1_FALLBACK)[['scenario', 'model', 'seed', 'f1']]
        fb = fb[fb.model.isin(['fado'] + BASELINES)]
        missing = fb.merge(f1[['scenario', 'model', 'seed']], how='left', indicator=True)
        missing = missing[missing._merge == 'left_only'].drop(columns='_merge')
        if len(missing):
            print(f'[f1] {len(missing)} whole-stream F1 values from the legacy per_seed.csv '
                  f'(crawl their sweeps to replace them)')
        f1 = pd.concat([f1, missing.assign(f1_post=np.nan)], ignore_index=True)
    data = data.merge(f1, on=['scenario', 'model', 'seed'], how='left')
    data.loc[data.scenario == 'folktables', list(POST)] = np.nan
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


def paired_tests(data, reference, comparators, metrics=None, blocks=None, table_cols=()):
    """`reference` against each comparator, per block and metric: two-sided paired Wilcoxon test.

    `gain` is the reference's mean advantage over the paired seeds, signed so that a positive
    value favours the reference for every metric (reference - method for F1 and accuracy,
    method - reference for DP and EO). p_col is Holm-corrected across the comparators of one
    block and metric, the family behind the marks; p_all across every comparison in
    `table_cols`, the stricter check reported alongside. Only pairs with all 30 seeds are tested."""
    metrics = SIG_METRICS if metrics is None else metrics
    table_cols = set(table_cols)
    rows = []
    for sc, _ in (BLOCKS if blocks is None else blocks):
        d = data[data.scenario == sc]
        ref = d[d.model == reference].set_index('seed')
        for col, _, higher in metrics:
            if col not in d or ref[col].notna().sum() != N_SEEDS:
                continue
            for b in comparators:
                other = d[d.model == b].set_index('seed')
                if other[col].notna().sum() != N_SEEDS:
                    continue
                gain = (ref[col] - other[col].reindex(ref.index)).dropna()
                gain = gain if higher else -gain
                rows.append(dict(scenario=sc, col=col, model=b, table=col in table_cols,
                                 gain=gain.mean(), better=int((gain > 0).sum()), n=len(gain),
                                 p=wilcoxon(gain).pvalue,
                                 p_welch=ttest_ind(ref[col], other[col], equal_var=False).pvalue))
    t = pd.DataFrame(rows, columns=['scenario', 'col', 'model', 'table', 'gain', 'better', 'n', 'p', 'p_welch'])
    if not len(t):
        return t.assign(p_col=[], p_welch_col=[], p_all=[], result=[])
    by_cell = t.groupby(['scenario', 'col'])
    t['p_col'] = by_cell.p.transform(lambda s: _holm(s.values))
    t['p_welch_col'] = by_cell.p_welch.transform(lambda s: _holm(s.values))
    t['p_all'] = np.nan
    if t.table.any():
        t.loc[t.table, 'p_all'] = _holm(t.loc[t.table, 'p'].values)
    t['result'] = np.where(t.p_col >= ALPHA, 'n.s.', np.where(t.gain > 0, 'win', 'loss'))
    return t


def fado_tests(data):
    """The tests behind the marks of Table III: FADO against each method."""
    return paired_tests(data, 'fado', COMPARATORS, table_cols=[c for c, _ in COLS])


ARANYANI_LAMBDA = 0.1  # Aranyani's published fixed weight, lambda_base in Table II
LEAKED_SWEEPS = {d: os.path.join(REPO, 'src', 'configs', 'sweeps', f'sweep_aranyani_stream_lambda_{d}.yaml')
                 for d in ('adult', 'folktables')}


def leaked_lambdas():
    """{dataset: constant lambda} of the leaked-lambda Aranyani, read from its sweep configs."""
    import yaml
    return {d: float(yaml.safe_load(open(p))['parameters']['lambda_const']['value']) for d, p in LEAKED_SWEEPS.items()}


WIN, LOSS = r'^{\bullet}', r'^{\circ}'
SLOT = r'\phantom{^{\bullet}}'  # unmarked cells keep the marker's width, so the columns stay aligned
FADO_SHADE = r'\rowcolor{black!6}'  # colortbl
OVERALL_ROWS = [('aranyani', r'vs.\ \ara{}'), ('tuned', r'vs.\ ' + dict(ROWS)['tuned']),
                ('arf', r'vs.\ \arf{}'), ('rfr', r'vs.\ \rfr{}')]
OUTCOMES = [('win', 'Significant wins'), ('n.s.', 'No significant difference'), ('loss', 'Significant losses')]


def overall_block(tests):
    """Top of Table III: FADO's significant wins / no significant difference / losses against
    each method, summed over every cell of the table (blocks x periods x metrics)."""
    t = tests[tests.table]
    per_method = t.groupby('model').size()
    assert per_method.nunique() == 1, per_method  # same cells for every method, so the counts compare

    widths = (2, 3, 2)  # columns spanned by wins / no significant difference / losses

    def row(cells):
        return ' & '.join(r'\multicolumn{' + str(w) + '}{c}{' + str(c) + '}' for w, c in zip(widths, cells))
    lines = [r'\multicolumn{8}{@{}l}{\textbf{Overall:} \fado{} against each method over its '
             + str(per_method.iloc[0]) + r' cells below} \\',
             '& ' + row([h for _, h in OUTCOMES]) + r' \\',
             r'\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(l){7-8}']
    for m, label in OVERALL_ROWS:
        r = t[t.model == m].result
        lines.append(label + ' & ' + row([int((r == k).sum()) for k, _ in OUTCOMES]) + r' \\')
    lines += [r'\addlinespace[2pt]',
              f'Total ({len(t)} comparisons) & ' + row([int((t.result == k).sum()) for k, _ in OUTCOMES]) + r' \\']

    return lines


def text_numbers(tests):
    """The figures the Results text quotes about the marks of Table III."""
    t = tests[tests.table]
    loss = t[t.result == 'loss']
    rest = loss[~loss.col.isin(['f1', 'f1_post'])]
    marked = t[t.result != 'n.s.']
    return (f'{len(t)} comparisons: {int((t.result == "win").sum())} wins, {int((t.result == "n.s.").sum())} n.s., '
            f'{len(loss)} losses ({len(loss) - len(rest)} on F1; the {len(rest)} on DP/EO all against the leaked '
            f'variant: {bool((rest.model == "tuned").all())}); {int((marked.p_all < ALPHA).sum())} of {len(marked)} '
            f'marks survive Holm across the whole table')


def metric_cells(data, tests, models, blocks, cols, sd=True):
    """LaTeX cells {(block, model, col): '$mean±sd$'}, bold for the best mean of the column in the
    block and the mark of the model's test against the reference. A cell needs all 30 seeds;
    otherwise it is left empty. Returns (cells, filled cells per row, whether any is empty)."""
    result = tests.set_index(['scenario', 'col', 'model']).result if len(tests) else pd.Series(dtype=object)
    cells, filled, empty = {}, {}, False
    for sc, _ in blocks:
        d = data[data.scenario == sc]
        by = {m: d[d.model == m].set_index('seed') for m in models}
        for col, higher in cols:
            if sc == 'folktables' and col in POST:
                continue
            have = [m for m in models if len(by[m]) and col in by[m] and by[m][col].notna().sum() == N_SEEDS]
            means = {m: by[m][col].mean() for m in have}
            # bold only when every method has the column, so a missing row never makes another one best
            best = (max if higher else min)(means, key=means.get) if len(have) == len(models) else None
            for m in models:
                if m not in have:
                    cells[(sc, m, col)], empty = '', True
                    continue
                v = f"{means[m]:.3f}{{\\pm}}{by[m][col].std(ddof=1):.3f}" if sd else f'{means[m]:.3f}'
                if m == best:
                    v = f'\\mathbf{{{v}}}'
                r = result.get((sc, col, m))
                v += WIN if r == 'win' else LOSS if r == 'loss' else SLOT
                cells[(sc, m, col)] = f'${v}$'
                filled[(sc, m)] = filled.get((sc, m), 0) + 1
    return cells, filled, empty


def results_table(data, tests):
    cells, filled, empty = metric_cells(data, tests, [m for m, _ in ROWS], BLOCKS, COLS)
    leaked = leaked_lambdas()

    def lam(sc, m, col):
        """The fairness weight: FADO's mean lambda_t over the stream, the Aranyani constants, none for ARF/RFR."""
        if m == 'fado':
            x = data[(data.scenario == sc) & (data.model == 'fado')][col]
            return f'${x.mean():.2f}$' if x.notna().sum() == N_SEEDS else ''  # mean only: it describes the run, it is not compared
        if m == 'aranyani':
            return f'${ARANYANI_LAMBDA:g}$'
        if m == 'tuned':
            return f"${leaked['folktables' if sc == 'folktables' else 'adult']:g}$"
        return '--'
    no_onset = r'\multirow{' + str(len(ROWS)) + r'}{*}{\emph{not applicable: no injected onset}}'
    body = []
    for k, (sc, title) in enumerate(BLOCKS):
        if k:
            body.append(r'\midrule')
        body.append(r'\multicolumn{8}{@{}l}{' + title + r'} \\')
        for i, (m, label) in enumerate(ROWS):
            whole = [cells[(sc, m, c)] for c, _ in COLS[:3]] + [lam(sc, m, 'stream_final_lambda')]
            if sc == 'folktables':
                after = [r'\multicolumn{3}{c}{' + (no_onset if i == 0 else '') + '}']
            else:
                after = [cells[(sc, m, c)] for c, _ in COLS[3:]]
            body.append((FADO_SHADE if m == 'fado' else '') + ' & '.join([label] + whole + after) + r' \\')
    overall = overall_block(tests)

    tex = '\n'.join([
        # [!t]: the table is taller than \dbltopfraction allows and would otherwise float to the end
        r'\begin{table*}[!t]', r'\centering',
        r'\caption{F1, Demographic Parity, and Equalized Odds Over 30 Seeds}',
        r'\label{tab:results}', r'\setlength{\tabcolsep}{5pt}', r'\footnotesize',
        r'\begin{tabular}{@{}l cccc ccc@{}}', r'\toprule'] + overall + [
        r'\midrule[\heavyrulewidth]',
        r'& \multicolumn{4}{c}{Whole stream} & \multicolumn{3}{c}{After the onset} \\',
        r'\cmidrule(lr){2-5} \cmidrule(l){6-8}',
        r'Method & F1\,$\uparrow$ & DP\,$\downarrow$ & EO\,$\downarrow$ & $\lambda$ & F1\,$\uparrow$ & DP\,$\downarrow$ & EO\,$\downarrow$ \\',
        r'\midrule'] + body + [
        r'\bottomrule', r'\end{tabular}',
        r'\tnote{Mean $\pm$ standard deviation over 30 seeds. Bold: best mean in the block. $^{\bullet}$/$^{\circ}$: \fado{} significantly better/worse'
        r' than that method (paired Wilcoxon test, Holm-corrected, $\alpha=0.05$). $\lambda$: fairness weight, the mean of $\lambda_t$ over the'
        r' stream for \fado{} and a constant for the \ara{} variants.' + (' Empty cell: fewer than 30 seeds.' if empty else '') + '}',
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
        r'\caption{Detection of the Injected Drift by \fado{} on \adult{}, 30 Seeds per Scenario}',
        r'\label{tab:detection}', r'\setlength{\tabcolsep}{5pt}', r'\footnotesize',
        r'\begin{tabular}{@{}l cc cc cc cc@{}}', r'\toprule',
        r'& \multicolumn{2}{c}{Either detector} & \multicolumn{2}{c}{Accuracy detector} & \multicolumn{2}{c}{Fairness monitor} & \multicolumn{2}{c}{False alarms per run} \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(l){8-9}',
        r'Scenario & Detected & Delay & Detected & Delay & Detected & Delay & Accuracy & Fairness \\',
        r'\midrule'] + rows + [
        r'\bottomrule', r'\end{tabular}',
        r'\tnote{Detected: an alarm within $1{,}000$ samples after the onset. Delay: mean number of samples from the onset to that alarm,'
        r' over the detected runs.}',
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


def significance_report(tests):
    """Every test behind the marks of Table III, per block, plus the Welch t-test."""
    marks = {'win': '•', 'loss': '◦', 'n.s.': '–'}
    marked = tests[tests.table & (tests.result != 'n.s.')]
    out = ['# Significance of FADO against each method', '',
           'Generated by `DOCS/paper_icde/scripts/make_tables.py`; every mark of Table III can be checked here.',
           'FADO is the reference. For every block, metric and method (all 30 seeds, paired by seed):', '',
           '- **FADO advantage**: mean paired difference in metric units (an absolute difference, not a',
           '  percentage), signed so that a **positive value favours FADO** for every metric: FADO − method for',
           '  F1 and accuracy, method − FADO for DP and EO.',
           '- **FADO better**: seeds on which FADO is better than the method.',
           '- **Wilcoxon p**: two-sided paired Wilcoxon signed-rank test, raw and Holm-corrected across the four',
           f'  methods compared with FADO in the same block and metric. This is the family of Table III (α = {ALPHA}).',
           '- **Holm, whole table**: the stricter check, Holm-corrected across all '
           f'{int(tests.table.sum())} comparisons shown in Table III;',
           f'  {int((marked.p_all < ALPHA).sum())} of its {len(marked)} marked differences stay below {ALPHA}. '
           'Accuracy is not in the table, so it has none.',
           '- **Mark**: • FADO significantly better, ◦ FADO significantly worse, – no significant difference',
           '  detected (not evidence of equivalence).',
           '- **Welch p**: two-sided Welch t-test, Holm-corrected like the Wilcoxon test; it ignores the pairing',
           '  and is reported for reference only.', '']
    for sc, block in BLOCKS:
        t = tests[tests.scenario == sc]
        if not len(t):
            continue
        name = block.replace('\\folk{}', 'Folktables').replace('\\adult{}', 'Adult').replace('\\_', '_')
        out += [f'## {name}', '',
                '| Metric | Method | FADO advantage | FADO better | Wilcoxon p | Wilcoxon p (Holm) | Holm, whole table | Mark | Welch p (Holm) |',
                '|---|---|---|---|---|---|---|---|---|']
        for col, label, _ in SIG_METRICS:
            for _, r in t[t.col == col].iterrows():
                whole = f'{r.p_all:.1e}' if r.table else '–'
                mark = marks[r.result] if r.table else 'not in table'
                out.append(f'| {label} | {NAMES[r.model]} | {r.gain:+.4f} | {r.better}/{r.n} | {r.p:.1e} | '
                           f'{r.p_col:.1e} | {whole} | {mark} | {r.p_welch_col:.1e} |')
        out.append('')
    return '\n'.join(out)


# ----------------------------------------------------------------------------- review experiments
# Experiments added after review (src/configs/sweeps/sweep_{ablation,dual_step,intersectional}_*.yaml).
# Each report combines its new runs with the arms the main sweeps already ran on the same seeds and
# streams, so no arm is run twice. While a sweep is not in SWEEPS its rows read "pending".
COL_NAMES = {'f1': 'F1', 'dp': 'DP', 'eo': 'EO', 'f1_post': 'F1 after onset', 'post_drift_dp': 'DP after onset',
             'post_drift_eo': 'EO after onset', 'phase_pre_drift_dp': 'DP before onset', 'stream_final_lambda': 'λ',
             'post_drift_lambda': 'λ after onset', 'dp_sex': 'DP, sex', 'dp_race': 'DP, race', 'eo_sex': 'EO, sex',
             'eo_race': 'EO, race'}
LOWER = {'dp', 'eo', 'post_drift_dp', 'post_drift_eo', 'phase_pre_drift_dp', 'dp_sex', 'dp_race', 'eo_sex', 'eo_race'}
ABLATION_ROWS = [('aranyani', 'none (Aranyani)'), ('fado_no_lambda', 'controller only'), ('fado_dual', 'dual only'),
                 ('fado_lambda_only', 'dual + reset'), ('fado_no_reset', 'dual + controller'), ('fado', 'all (FADO)')]
ABLATION_NEW = ['fado_dual', 'fado_lambda_only', 'fado_no_reset']


def run_meta(sweeps, refresh=False):
    """Config and machine of every finished run of `sweeps` (W&B run metadata), cached as meta.csv."""
    path = os.path.join(_cache_dir(sweeps), 'meta.csv')
    if refresh or not os.path.isfile(path):
        import wandb
        api = wandb.Api(timeout=120)
        rows = []
        for sid in sweeps:
            for run in api.sweep(f'{ENTITY}/{PROJECT}/{sid}').runs:
                if run.state != 'finished':
                    continue
                md, cfg = run.metadata or {}, run.config
                rows.append(dict(run_id=run.id, sweep=sid,
                                 scenario=cfg.get('drift_scenario') or cfg.get('pipeline_dataset'),
                                 lambda_dual_lr=cfg.get('lambda_dual_lr'), intersectional=cfg.get('intersectional'),
                                 host=md.get('host'), os=md.get('os'), python=md.get('python'),
                                 cpu_count=md.get('cpu_count')))
        os.makedirs(_cache_dir(sweeps), exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path)


def sweep_metrics(key, models=None, refresh=False):
    """Per-seed metrics of the runs of SWEEPS[key] with their F1 (by run), or None if not registered."""
    sweeps = SWEEPS.get(key) or []
    if not sweeps:
        return None
    raw = arm_metrics(sweeps, refresh)
    if models is not None:
        raw = raw[raw.model.isin(models)]
    store = pd.read_csv(F1_STORE) if os.path.isfile(F1_STORE) else pd.DataFrame(columns=STORE_COLS)
    f1 = store[store.run_id.isin(set(raw.run_id))][['run_id', 'model', 'seed', 'f1', 'f1_post']]
    data = raw.merge(f1, on=['run_id', 'model', 'seed'], how='left')
    data.loc[data.scenario == 'folktables', [c for c in POST if c in data]] = np.nan
    return data


def arm_report_md(title, intro, data, tests, rows, blocks, cols, expected):
    """Markdown: mean ± sd per block and arm, with the mark of the arm's test against the reference.
    `expected` lists, per block, arms whose sweep is not registered yet (shown as pending)."""
    marks = {'win': ' •', 'loss': ' ◦'}
    result = tests.set_index(['scenario', 'col', 'model']).result if len(tests) else {}
    out = [f'# {title}', ''] + intro + ['']
    for sc, block in blocks:
        d = data[data.scenario == sc] if data is not None else pd.DataFrame(columns=['model'])
        cs = [c for c in cols if not (sc == 'folktables' and c in POST)]
        shown = [(m, label) for m, label in rows if (d.model == m).any() or m in expected.get(sc, ())]
        if not shown:
            continue
        name = block.replace('\\folk{}', 'Folktables').replace('\\_', '_')
        out += [f'## {name}', '', '| Arm | ' + ' | '.join(COL_NAMES[c] for c in cs) + ' |',
                '|---' * (len(cs) + 1) + '|']
        for m, label in shown:
            x = d[d.model == m]
            cells = []
            for c in cs:
                v = x[c].dropna() if c in x else pd.Series(dtype=float)
                if len(v) == N_SEEDS:
                    cells.append(f'{v.mean():.3f} ± {v.std(ddof=1):.3f}' + marks.get(result.get((sc, c, m)), ''))
                elif not len(x):
                    cells.append('pending')        # the arm's sweep has not run yet
                elif not len(v) and 'lambda' in c:
                    cells.append('–')              # no adaptive weight
                else:
                    cells.append(f'{len(v)}/30 seeds')
            out.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        out.append('')
    return '\n'.join(out)


def _metrics(cols):
    return [(c, COL_NAMES[c], c not in LOWER) for c in cols]


def ablation_report(refresh):
    """Component ablation: each arm against full FADO, both datasets, in one report."""
    parts = [sweep_metrics('main_adult', ['fado', 'aranyani'], refresh),
             sweep_metrics('main_folktables', ['fado', 'aranyani', 'fado_no_lambda'], refresh),
             sweep_metrics('ablation_adult', ABLATION_NEW, refresh),
             sweep_metrics('ablation_folktables', ABLATION_NEW, refresh)]
    data = pd.concat([p for p in parts if p is not None], ignore_index=True)
    cols = ['f1', 'dp', 'eo', 'stream_final_lambda', 'f1_post', 'post_drift_dp', 'post_drift_eo']
    tests = paired_tests(data, 'fado', [m for m, _ in ABLATION_ROWS if m != 'fado'],
                         metrics=_metrics([c for c in cols if c != 'stream_final_lambda']))
    expected = {sc: ABLATION_NEW for sc, _ in BLOCKS
                if not SWEEPS['ablation_folktables' if sc == 'folktables' else 'ablation_adult']}
    intro = ['Generated by `DOCS/paper_icde/scripts/make_tables.py` from sweeps `ablation_adult` and',
             '`ablation_folktables` (src/configs/sweeps/sweep_ablation_*.yaml). "none" and "all" are the',
             'Aranyani and FADO runs of Table III, and "controller only" is the Folktables ablation of the',
             'paper (`fado_no_lambda`); every arm runs on the same seeds and streams. Components: dual = the',
             'adaptive constraint (dual ascent on λ_t); reset = drift detectors plus fairness monitor resetting',
             'the fairness statistics; controller = learning-rate and temperature reaction to confirmed drift.',
             '', 'Mean ± sd over 30 seeds. • full FADO significantly better than that arm, ◦ significantly worse',
             '(paired two-sided Wilcoxon, Holm across the arms of the column, α = 0.05).']
    md = arm_report_md('Component ablation', intro, data, tests, ABLATION_ROWS, BLOCKS, cols, expected)
    return md, data, tests


def dual_step_report(refresh):
    """FADO with larger dual steps against the paper's eta = 0.01, on the scenarios of the sweep."""
    title = 'Dual step size and the demographic-parity target'
    intro = ['Generated by `DOCS/paper_icde/scripts/make_tables.py` from sweep `dualstep_adult`',
             '(src/configs/sweeps/sweep_dual_step_adult.yaml). η = 0.01 is FADO in Table III. The bound',
             'average violation ≤ μ_T / (η T) predicts that a larger η brings DP closer to ε = 0.05;',
             'supplementary/dual_bound.md checks the bound on every run with local per-sample results.',
             '', 'Mean ± sd over 30 seeds. • η = 0.01 significantly better than that η, ◦ significantly worse',
             '(paired two-sided Wilcoxon, Holm across the step sizes of the column, α = 0.05).']
    if not SWEEPS['dualstep_adult']:
        return '\n'.join([f'# {title}', ''] + intro + ['', '**Pending:** sweep `dualstep_adult` is not registered.', ''])
    new = sweep_metrics('dualstep_adult', ['fado'], refresh)
    new = new.merge(run_meta(SWEEPS['dualstep_adult'], refresh)[['run_id', 'lambda_dual_lr']], on='run_id')
    new['model'] = 'fado_eta' + new.lambda_dual_lr.map(lambda e: f'{e:g}')
    scenarios = sorted(new.scenario.unique(), key=ADULT.index)
    base = sweep_metrics('main_adult', ['fado'], refresh)
    data = pd.concat([base[base.scenario.isin(scenarios)], new], ignore_index=True)
    arms = sorted(new.model.unique(), key=lambda m: float(m[8:]))
    rows = [('fado', 'η = 0.01 (paper)')] + [(m, f'η = {m[8:]}') for m in arms]
    cols = ['phase_pre_drift_dp', 'dp', 'eo', 'f1', 'post_drift_dp', 'post_drift_eo', 'f1_post',
            'stream_final_lambda', 'post_drift_lambda']
    tests = paired_tests(data, 'fado', arms, metrics=_metrics([c for c in cols if 'lambda' not in c]),
                         blocks=[b for b in BLOCKS if b[0] in scenarios])
    return arm_report_md(title, intro, data, tests, rows, [b for b in BLOCKS if b[0] in scenarios], cols, {})


def intersectional_report(refresh):
    """FADO, Aranyani and ARF with four groups (sex x race) on Folktables."""
    title = 'Intersectional fairness on Folktables (sex × race, four groups)'
    intro = ['Generated by `DOCS/paper_icde/scripts/make_tables.py` from sweep `intersectional_folktables`',
             '(src/configs/sweeps/sweep_intersectional_folktables.yaml). DP and EO are over the four groups',
             '(rolling window, as in Table III); "DP, sex" and "DP, race" treat each attribute on its own, over',
             'the whole prediction stream. RFR is not run: its penalty is defined for two groups.',
             '', 'Mean ± sd over 30 seeds. • FADO significantly better than that method, ◦ significantly worse',
             '(paired two-sided Wilcoxon, Holm across the methods of the column, α = 0.05).']
    rows = [('fado', 'FADO'), ('aranyani', 'Aranyani'), ('arf', 'ARF')]
    cols = ['f1', 'dp', 'eo', 'dp_sex', 'dp_race', 'eo_sex', 'eo_race', 'stream_final_lambda']
    data = sweep_metrics('intersectional_folktables', [m for m, _ in rows], refresh)
    tests = (paired_tests(data, 'fado', ['aranyani', 'arf'], metrics=_metrics(cols[:-1]),
                          blocks=[BLOCKS[0]]) if data is not None else pd.DataFrame())
    expected = {} if data is not None else {'folktables': [m for m, _ in rows]}
    return arm_report_md(title, intro, data, tests, rows, [BLOCKS[0]], cols, expected)


def review_reports(refresh):
    out = os.path.join(PAPER, 'supplementary')
    md, _, _ = ablation_report(refresh)
    open(os.path.join(out, 'ablation.md'), 'w').write(md)
    open(os.path.join(out, 'dual_step.md'), 'w').write(dual_step_report(refresh))
    open(os.path.join(out, 'intersectional.md'), 'w').write(intersectional_report(refresh))
    pending = [k for k in ('ablation_adult', 'ablation_folktables', 'dualstep_adult', 'intersectional_folktables')
               if not SWEEPS[k]]
    print('[review] supplementary/ablation.md, dual_step.md, intersectional.md'
          + (f'; not registered yet: {", ".join(pending)}' if pending else ''))


# ----------------------------------------------------------------------------- main
def missing_report(data):
    """Which table cells are still empty, and which sweeps would fill them."""
    roles = _sweep_roles()
    need = []
    for sc, _ in BLOCKS:
        for m, _ in ROWS:
            d = data[(data.scenario == sc) & (data.model == m)]
            for col, _ in COLS:
                if sc == 'folktables' and col in POST:
                    continue
                n = int(d[col].notna().sum()) if len(d) else 0
                if n < N_SEEDS:
                    need.append((sc, m, col, n))
    if not need:
        print('[status] every cell has 30 seeds')
        return
    print('[status] cells with fewer than 30 seeds (F1 columns need a crawl on the machine with '
          'those run folders; the others need the sweep to finish, then build --refresh):')
    for sc, m, col, n in need:
        group = ('tuned_' if m == 'tuned' else 'main_') + ('folktables' if sc == 'folktables' else 'adult')
        print(f'    {sc:20s} {m:9s} {col:14s} {n:2d}/30   sweeps: {", ".join(SWEEPS[group])}')


def build(refresh):
    data = load(refresh)
    main_adult = arm_metrics(SWEEPS['main_adult'], False)
    det = detection(SWEEPS['main_adult'], refresh)

    os.makedirs(os.path.join(PAPER, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(PAPER, 'supplementary'), exist_ok=True)
    tests = fado_tests(data)
    results, filled = results_table(data, tests)
    open(os.path.join(PAPER, 'tables', 'results.tex'), 'w').write(results)
    open(os.path.join(PAPER, 'tables', 'detection.tex'), 'w').write(detection_table(det, main_adult))
    open(os.path.join(PAPER, 'supplementary', 'significance.md'), 'w').write(significance_report(tests))
    md, split = detection_split(det, main_adult)
    open(os.path.join(PAPER, 'supplementary', 'detection_split.md'), 'w').write(md)
    split.to_csv(os.path.join(PAPER, 'supplementary', 'adult_detected_vs_undetected.csv'), index=False)

    print('Table III filled cells per row (of 3 for Folktables, 6 for Adult):')
    for sc, _ in BLOCKS:
        print(f'  {sc:20s} ' + '  '.join(f'{m}={filled.get((sc, m), 0)}' for m, _ in ROWS))
    missing_report(data)
    print('[text] numbers quoted in the Results text: ' + text_numbers(tests))
    review_reports(refresh)
    print(f'written: {os.path.relpath(os.path.join(PAPER, "tables"), REPO)}/results.tex, detection.tex; '
          f'{os.path.relpath(os.path.join(PAPER, "supplementary"), REPO)}/')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('command', nargs='?', default='build', choices=['crawl', 'build', 'all'])
    parser.add_argument('--refresh', action='store_true', help='download W&B data again (build)')
    args = parser.parse_args()
    if args.command in ('crawl', 'all'):
        crawl()
    if args.command in ('build', 'all'):
        build(args.refresh)


if __name__ == '__main__':
    main()
