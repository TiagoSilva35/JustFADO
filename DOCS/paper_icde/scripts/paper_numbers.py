"""Recompute every number the paper's text quotes that is not printed in Table III or IV.

    python DOCS/paper_icde/scripts/paper_numbers.py

Writes DOCS/paper_icde/supplementary/paper_numbers.md: one row per claim, with the value,
the computation, and the data it comes from, so each number in the text can be checked.

Sources (the same as make_tables.py, plus the per-sample run folders where a claim needs them):
    * W&B per-arm metrics of the paper's sweeps (make_tables.load / arm_metrics, cached);
    * DOCS/paper_icde/data/f1_store.csv (F1 from the per-sample predictions);
    * DOCS/paper_icde/supplementary/adult_detected_vs_undetected.csv (written by make_tables);
    * local run folders files/experiments/wandb/<run>/ for the per-sample curves. Only the
      sweeps present on this machine are used; each row says which runs it covers.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402
import runs as run_store  # noqa: E402

OUT = os.path.join(mt.PAPER, 'supplementary', 'paper_numbers.md')
NAMES = {'sim_x_permutations': 'x_permutations', 'sim_y_swaps': 'y_swaps',
         'sim_y_prior_skip': 'y_prior_skip', 'sim_x_exceed_skip': 'x_exceed_skip',
         'folktables': 'Folktables'}
rows = []


def claim(where, text, value, how, source):
    rows.append((where, text, value, how, source))


def local_runs(sweeps, scenario=None):
    """{(model, seed): per-sample results} of the local run folders of `sweeps` (runs.local_runs).
    Callers pass one scenario, or a sweep that has only one, so (model, seed) is unique."""
    return {(r["model"], r["seed"]): r for r in run_store.local_runs(sweeps, scenario)}


def fmt(x, nd=3):
    return f'{x:.{nd}f}'


# ----------------------------------------------------------------------------- data
data = mt.load(False)
tests = mt.fado_tests(data)
adult = mt.arm_metrics(mt.SWEEPS['main_adult'], False)
folk_raw = mt.arm_metrics(mt.SWEEPS['main_folktables'], False)
mean = data.groupby(['scenario', 'model']).mean(numeric_only=True)
ADULT = mt.ADULT

# ----------------------------------------------------------------------------- Results intro
folk_local = local_runs({'jo5dqqhy'})
share = [np.mean(ts['y_true_all']) for (m, s), ts in folk_local.items() if m == 'fado']
adult_share = {}
for sc, sweeps in [('sim_x_exceed_skip', {'l8c4i60o'}), ('sim_y_prior_skip', {'l8c4i60o'}),
                   ('sim_x_permutations', {'rm2glvmh'})]:
    runs = local_runs(sweeps, sc)
    adult_share[sc] = [np.mean(ts['y_true_all']) for (m, s), ts in runs.items() if m in ('fado', 'aranyani')]
# y_swaps: no run folder of the paper's sweeps on this machine, so its label streams are regenerated
# with the pipeline's own loader (data only, no model). The regeneration is checked against the
# stored labels of the x_exceed_skip runs.
sys.path.insert(0, mt.REPO)
os.environ.setdefault('PYTHONHASHSEED', '0')
from absl import flags  # noqa: E402
import src.main as pipeline  # noqa: E402
flags.FLAGS(['paper_numbers'])
seeds = sorted(adult.seed.unique())
regen = {sc: {int(s): np.asarray(pipeline._load_dataset_splits('adult', sc, int(s))[3]) for s in seeds}
         for sc in ('sim_y_swaps', 'sim_x_exceed_skip')}
stored = {s: ts['y_true_all'] for (m, s), ts in local_runs({'l8c4i60o'}, 'sim_x_exceed_skip').items()
          if m == 'aranyani'}
same = [np.array_equal(regen['sim_x_exceed_skip'][s], y) for s, y in stored.items()]
adult_share['sim_y_swaps'] = [y.mean() for y in regen['sim_y_swaps'].values()]
per_sc = {sc: np.mean(v) for sc, v in adult_share.items()}
claim('V (intro)', 'positive class is 40% of the Folktables stream and 24% of the Adult test split',
      f'Folktables {100 * np.mean(share):.1f}% (n={len(share)} streams); Adult test split (x_permutations, labels '
      f'unchanged) {100 * per_sc["sim_x_permutations"]:.1f}%; with the drift: '
      + ', '.join(f'{NAMES[sc]} {100 * v:.1f}%' for sc, v in per_sc.items() if sc != 'sim_x_permutations')
      + f' (y_swaps per seed {100 * min(adult_share["sim_y_swaps"]):.1f}–{100 * max(adult_share["sim_y_swaps"]):.1f}%)',
      'mean of y_true over each test stream, averaged over seeds',
      'local run folders: jo5dqqhy (Folktables), l8c4i60o, rm2glvmh; y_swaps regenerated with the pipeline loader '
      f'(the regenerated x_exceed_skip labels equal the stored ones on {sum(same)}/{len(same)} seeds)')

claim('V (intro)', '108 comparisons: 79 wins, 15 losses; 11 losses on F1, the 4 on DP/EO all against the leaked '
      'variant; 81 of the 94 marks survive Holm across the whole table',
      mt.text_numbers(tests), 'make_tables.fado_tests (the tests behind the marks of Table III)',
      'W&B arm metrics + F1 store')

# ----------------------------------------------------------------------------- RQ1
pre_post = adult.groupby(['scenario', 'model'])[['phase_pre_drift_dp', 'post_drift_dp',
                                                 'phase_pre_drift_eo', 'post_drift_eo']].mean()
a = pre_post.xs('aranyani', level='model')
txt = '; '.join(f"{NAMES[sc]}: DP {fmt(a.loc[sc, 'phase_pre_drift_dp'])}→{fmt(a.loc[sc, 'post_drift_dp'])}, "
                f"EO {fmt(a.loc[sc, 'phase_pre_drift_eo'])}→{fmt(a.loc[sc, 'post_drift_eo'])}" for sc in ADULT)
ax = adult[(adult.model == 'aranyani') & (adult.scenario == 'sim_x_permutations')]
claim('V-A', 'Aranyani before→after the onset: EO 0.047→0.115 (x_perm), 0.047→0.076 and DP 0.082→0.097 (y_prior), '
      'DP 0.082→0.054 (y_swaps), 0.082→0.061 (x_exceed)', txt,
      'mean over 30 seeds of phase_pre_drift_* (onset excluded) and post_drift_* (onset to end)',
      'W&B arm metrics, sweeps ' + '+'.join(mt.SWEEPS['main_adult']))
claim('V-A', 'x_permutations: Aranyani EO increases on all 30 seeds',
      f"{int((ax.post_drift_eo > ax.phase_pre_drift_eo).sum())}/{len(ax)} seeds",
      'count of seeds with post_drift_eo > phase_pre_drift_eo', 'W&B arm metrics')
lines = []
ok_fall, ok_above = True, True
for m in ('arf', 'rfr'):
    for sc in ADULT:
        pre, post = pre_post.loc[(sc, m), 'phase_pre_drift_dp'], pre_post.loc[(sc, m), 'post_drift_dp']
        f_pre, f_post = pre_post.loc[(sc, 'fado'), 'phase_pre_drift_dp'], pre_post.loc[(sc, 'fado'), 'post_drift_dp']
        if sc in ('sim_y_swaps', 'sim_x_exceed_skip'):
            ok_fall &= post < pre
        ok_above &= pre > f_pre and post > f_post
        lines.append(f'{m} {NAMES[sc]} {fmt(pre)}→{fmt(post)} (FADO {fmt(f_pre)}→{fmt(f_post)})')
claim('V-A', 'ARF and RFR: DP falls after the onset of y_swaps and x_exceed_skip, and stays above FADO\'s before and '
      'after every onset', f'falls: {ok_fall}; above FADO: {ok_above}. ' + '; '.join(lines),
      'means over 30 seeds of phase_pre_drift_dp and post_drift_dp', 'W&B arm metrics')
halves = []
for m in ['fado', 'aranyani', 'tuned', 'arf', 'rfr']:
    src = local_runs({'9ulal3qn'}) if m == 'tuned' else folk_local
    key = 'aranyani' if m == 'tuned' else m
    h = [(np.mean(ts['dp'][:len(ts['dp']) // 2]), np.mean(ts['dp'][len(ts['dp']) // 2:]))
         for (mm, s), ts in src.items() if mm == key]
    if h:
        h = np.array(h)
        halves.append((m, h[:, 0].mean(), h[:, 1].mean(), len(h)))
claim('V-A', 'Folktables: every method has a slightly lower DP on the 2018 half than on the 2017 half',
      '; '.join(f'{m} {fmt(x)}→{fmt(y)} (n={n})' for m, x, y, n in halves)
      + f'; all lower: {all(y < x for _, x, y, _ in halves)}',
      'mean of the rolling DP over the first and second half of each stream (the test stream is the two years '
      'in equal shares, then subsampled uniformly to 10%), averaged over seeds',
      'local run folders: jo5dqqhy, 9ulal3qn (leaked variant)')

f = adult[adult.model == 'fado']
fp = {sc: f[f.scenario == sc] for sc in ADULT}
yp = fp['sim_y_prior_skip']
claim('V-B', 'FADO EO before→after: 0.047→0.064 (x_perm); the increase under y_prior_skip is not significant',
      f"x_perm {fmt(fp['sim_x_permutations'].phase_pre_drift_eo.mean())}→{fmt(fp['sim_x_permutations'].post_drift_eo.mean())}; "
      f"y_prior {fmt(yp.phase_pre_drift_eo.mean())}→{fmt(yp.post_drift_eo.mean())}, Wilcoxon p = "
      f"{wilcoxon(yp.post_drift_eo - yp.phase_pre_drift_eo).pvalue:.2f}",
      'means over 30 seeds; two-sided paired Wilcoxon of post_drift_eo against phase_pre_drift_eo', 'W&B arm metrics')

# ----------------------------------------------------------------------------- RQ2
red = {sc: 1 - mean.loc[(sc, 'fado'), 'dp'] / mean.loc[(sc, 'aranyani'), 'dp'] for sc, _ in mt.BLOCKS}
claim('Abstract, V-B, VI', 'DP reduction relative to Aranyani ranges from 25% (x_exceed_skip) to 33% (y_swaps)',
      '; '.join(f'{NAMES[sc]} {100 * v:.1f}%' for sc, v in red.items()),
      '1 − mean DP(FADO) / mean DP(Aranyani), whole stream, 30 seeds', 'Table III data')
red_post = {sc: 1 - mean.loc[(sc, 'fado'), 'post_drift_dp'] / mean.loc[(sc, 'aranyani'), 'post_drift_dp'] for sc in ADULT}
claim('Abstract, VI', 'DP reduction relative to Aranyani after the onset',
      '; '.join(f'{NAMES[sc]} {100 * v:.1f}%' for sc, v in red_post.items()),
      '1 − mean post-onset DP(FADO) / mean post-onset DP(Aranyani), 30 seeds', 'Table III data')
dp_pub = tests[tests.col.isin(['dp', 'post_drift_dp']) & tests.model.isin(mt.BASELINES)]
claim('V-B', 'DP significantly lower than each of Aranyani, ARF, RFR in every block and period (p < 0.001)',
      f"{int((dp_pub.result == 'win').sum())}/{len(dp_pub)} wins; largest Holm-corrected p = {dp_pub.p_col.max():.1e}",
      'make_tables.fado_tests', 'Table III data')
f1_gap = {sc: mean.loc[(sc, 'aranyani'), 'f1'] - mean.loc[(sc, 'fado'), 'f1'] for sc, _ in mt.BLOCKS}
claim('Abstract, V-B, VI', 'F1 cost relative to Aranyani over the whole stream: at most 0.019',
      '; '.join(f'{NAMES[sc]} {v:+.4f}' for sc, v in f1_gap.items()),
      'mean F1(Aranyani) − mean F1(FADO), whole stream (positive = FADO lower)', 'Table III data')
f1_arf = {sc: mean.loc[(sc, 'fado'), c] - mean.loc[(sc, 'arf'), c]
          for sc, _ in mt.BLOCKS for c in (['f1'] if sc == 'folktables' else ['f1'])}
claim('V-B', 'FADO remains well above ARF in F1 in every block (whole stream)',
      '; '.join(f'{NAMES[sc]} +{v:.3f}' for sc, v in f1_arf.items()), 'mean F1(FADO) − mean F1(ARF)', 'Table III data')

lam_pre = f.phase_pre_drift_lambda.mean()
lam_post = f.groupby('scenario').post_drift_lambda.mean()
claim('V-B (leaked weight)', 'FADO\'s weight averages 0.97 before the onset and 2.30 after the onset of x_permutations, '
      'against the leaked constant 1.5',
      f'λ before the onset {lam_pre:.2f} (all Adult runs; identical across scenarios); after the onset of x_permutations '
      f'{lam_post["sim_x_permutations"]:.2f}; leaked constant {mt.leaked_lambdas()["adult"]:g}',
      'means over 30 seeds of phase_pre_drift_lambda and post_drift_lambda; the constant from the leaked sweep config',
      'W&B arm metrics; src/configs/sweeps/sweep_aranyani_stream_lambda_adult.yaml')

claim('V-B (utility)', 'y_prior_skip is the drift after which FADO raises its weight the most, to 3.01 on average',
      'mean λ_t after the onset: ' + ', '.join(f'{NAMES[sc]} {lam_post[sc]:.2f}' for sc in ADULT)
      + f'; largest: {NAMES[lam_post[ADULT].idxmax()]}', 'mean over 30 seeds of post_drift_lambda', 'W&B arm metrics')

split = pd.read_csv(os.path.join(mt.PAPER, 'supplementary', 'adult_detected_vs_undetected.csv'))
und, det = split[~split.detected], split[split.detected]
claim('V-B', 'undetected runs: 62; FADO DP after onset lower than Aranyani in 61; mean ΔDP −0.013 (undetected) '
      'vs −0.035 (detected); the EO advantage largely requires a detection',
      f'{len(und)} undetected, {int((und.d_post_drift_dp < 0).sum())} with ΔDP < 0; ΔDP {und.d_post_drift_dp.mean():+.3f} '
      f'vs {det.d_post_drift_dp.mean():+.3f}; ΔEO {und.d_post_drift_eo.mean():+.3f} (undetected, Wilcoxon p = '
      f'{wilcoxon(und.d_post_drift_eo).pvalue:.2f}) vs {det.d_post_drift_eo.mean():+.3f} (detected, p = '
      f'{wilcoxon(det.d_post_drift_eo).pvalue:.1e})',
      'Δ = FADO − Aranyani after the onset, paired by seed; detected = either detector within 1,000 samples',
      'supplementary/adult_detected_vs_undetected.csv (make_tables.py)')

# ----------------------------------------------------------------------------- ablation
ab = folk_raw.set_index(['model', 'seed'])
f1_local = {}
for (m, s), ts in folk_local.items():
    if m in ('fado', 'fado_no_lambda'):
        f1_local[(m, s)] = mt._f1(np.asarray(ts['y_true_all'], dtype=int), np.asarray(ts['y_preds_all'], dtype=int))
ab_lines = []
for col, higher in [('f1', True), ('accuracy', True), ('dp', False), ('eo', False)]:
    if col == 'f1':
        x = pd.Series({s: v for (m, s), v in f1_local.items() if m == 'fado'})
        y = pd.Series({s: v for (m, s), v in f1_local.items() if m == 'fado_no_lambda'})
    else:
        x, y = ab.loc['fado'][col], ab.loc['fado_no_lambda'][col]
    gain = (x - y.reindex(x.index)).dropna()
    gain = gain if higher else -gain
    ab_lines.append(f'{col} {fmt(y.mean())}→{fmt(x.mean())} (no-λ→FADO), FADO better on {int((gain > 0).sum())}/{len(gain)} '
                    f'seeds, Wilcoxon p = {wilcoxon(gain).pvalue:.1e}')
claim('V-C', 'removing the adaptive constraint worsens every metric on at least 28 of 30 seeds (p < 1e-8); DP '
      '0.042→0.050, EO 0.025→0.035, F1 and accuracy fall', '; '.join(ab_lines),
      'fado vs fado_no_lambda (λ fixed at λ_base, no fairness monitor, no reset) on the same 30 seeds; '
      'two-sided paired Wilcoxon', 'W&B arm metrics jo5dqqhy; F1 from local run folders jo5dqqhy')
ab_halves = []
for m in ('fado', 'fado_no_lambda'):
    h = np.array([(np.mean(ts['dp'][:len(ts['dp']) // 2]), np.mean(ts['dp'][len(ts['dp']) // 2:]))
                  for (mm, s), ts in folk_local.items() if mm == m])
    ab_halves.append(f'{m}: first half {fmt(h[:, 0].mean())}, second half {fmt(h[:, 1].mean())}')
claim('V-C', 'the gain is concentrated in the 2017 half; on the 2018 half the two variants reach a similar DP',
      '; '.join(ab_halves), 'mean rolling DP over each half of the stream, averaged over 30 seeds',
      'local run folders jo5dqqhy')

# ----------------------------------------------------------------------------- machines
hosts = pd.concat([mt.run_meta(v) for v in mt.SWEEPS.values() if v], ignore_index=True)
machines = hosts.groupby(['host', 'os', 'python', 'cpu_count']).sweep.agg(lambda x: ', '.join(sorted(set(x))))
claim('IV (setup)', 'runs executed on three machines (Linux, Windows, macOS), CPU only, Python 3.11 or 3.12',
      '; '.join(f'{h} ({o}, {py}, {c} CPUs): sweeps {sw}' for (h, o, py, c), sw in machines.items()),
      'W&B run metadata of every finished run of the paper\'s sweeps', 'W&B (make_tables.run_meta, cached meta.csv)')

# Before the onset, the four Adult scenarios of a seed are the same stream, so a deterministic method must
# produce exactly the same metrics on it in every scenario, wherever each scenario ran.
pre_cols = ['phase_pre_drift_accuracy', 'phase_pre_drift_dp', 'phase_pre_drift_eo']
tuned_adult = mt.arm_metrics(mt.SWEEPS['tuned_adult'], False)
tuned_adult = tuned_adult[tuned_adult.model == 'aranyani'].assign(model='tuned')
both = pd.concat([adult[adult.model.isin(['fado'] + mt.BASELINES)], tuned_adult], ignore_index=True)
both = both.merge(hosts[['run_id', 'host']], on='run_id', how='left')
equal = both.groupby(['model', 'seed'])[pre_cols].nunique().eq(1).all(axis=1).groupby('model').agg(['sum', 'size'])
spread = both.groupby(['model', 'seed']).phase_pre_drift_dp.agg(lambda x: x.max() - x.min()).groupby('model').max()
where = both.groupby(['model', 'scenario']).host.agg(lambda h: '/'.join(sorted(set(h.dropna()))))
claim('IV (setup)', 'FADO and both Aranyani variants give identical results on all three machines; ARF and RFR are '
      'reproducible on one machine but differ slightly across machines (pre-onset DP by at most 0.010)',
      '; '.join(f'{m}: {int(r["sum"])}/{int(r["size"])} seeds identical across the four scenarios, largest pre-onset DP '
                f'difference {spread[m]:.4f}' for m, r in equal.iterrows())
      + '. Scenarios ran on: ' + '; '.join(f'{m} {NAMES[sc]} {h}' for (m, sc), h in where.items()),
      'exact equality (no tolerance) of phase_pre_drift_* across the four Adult scenarios of each seed: the stream '
      'before the onset is the same in every scenario, and the scenarios ran on different machines',
      'W&B arm metrics of the main and leaked-λ Adult sweeps, with the run hosts')
claim('IV (setup)', 'the data stream of a seed is the same on every machine',
      f'x_exceed_skip labels regenerated on this machine equal the stored labels of the paper runs on {sum(same)}/{len(same)} '
      'seeds', 'exact equality of the regenerated and stored label streams', 'pipeline loader; local run folders l8c4i60o '
      '(run on tesla)')
lam_mean = pd.concat([adult[adult.model == 'fado'], folk_raw[folk_raw.model == 'fado'].assign(scenario='folktables')])
lam_mean = lam_mean.groupby('scenario').stream_final_lambda.mean()
claim('IV-F', 'the leaked constant is at least FADO\'s mean λ_t on each dataset: 0.5 on Folktables, 1.5 on Adult',
      f"FADO mean λ_t: Folktables {lam_mean['folktables']:.3f} (≤ 0.5: {lam_mean['folktables'] <= 0.5}); Adult "
      + ', '.join(f'{NAMES[sc]} {lam_mean[sc]:.3f}' for sc in ADULT)
      + f" (all ≤ 1.5: {bool((lam_mean[ADULT] <= 1.5).all())})",
      'mean over 30 seeds of stream_final_lambda (mean of λ_t over the stream)', 'W&B arm metrics')

# ----------------------------------------------------------------------------- arithmetic
claim('IV-G', 'ε = 0.05 allows a positive-prediction-rate gap of at most 0.10 between two groups',
      f'{2 * 0.05:.2f}', 'for two groups DP = |p0 − p1| / 2 (Section III-B), so the gap is 2ε', 'definition')
claim('IV-G', 'Δτ = 2e-3 returns τ from 0.1 to 1 in 450 steps, inside the Δ = 3000 horizon',
      f'{(1 - 0.1) / 2e-3:.0f} steps', '(1 − τ_drift) / Δτ', 'Table II values')

# ----------------------------------------------------------------------------- write
md = ['# Numbers quoted in the paper text', '',
      'Generated by `DOCS/paper_icde/scripts/paper_numbers.py`. Numbers printed in Table III and Table IV',
      'come from `make_tables.py` and are not repeated here, except where the text derives something from',
      'them (a ratio, a difference, a count). Percentages are computed from unrounded means.', '',
      '| Section | Claim in the text | Value computed | Computation | Source |', '|---|---|---|---|---|']
for r in rows:
    md.append('| ' + ' | '.join(str(x).replace('|', '/') for x in r) + ' |')
open(OUT, 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
print(f'-> {os.path.relpath(OUT, mt.REPO)}')
