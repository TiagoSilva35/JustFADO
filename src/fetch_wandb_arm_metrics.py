"""Download the per-run `arm_metrics` tables of a W&B sweep and summarise them
over seeds.

Each run logs `arm_metrics`: one row per (seed, model, scenario) with the
run's final metrics (accuracy, dp, eo, stream_final_*, post_drift_*,
phase_{pre_drift,transition,drift}_*, lambda, detections, fairness_resets,
ms_per_sample). Runs logged before the phase/detection columns existed get
them from the run summary (one-seed runs only). This script pulls those tables for every
finished run of a sweep, keeps only the (seed, scenario) pairs where EVERY
model finished, and reports mean and std over seeds.

    python -m src.fetch_wandb_arm_metrics --sweep ych8qely
    python -m src.fetch_wandb_arm_metrics --sweep ych8qely --metrics accuracy,dp,phase_drift_dp

Writes to files/experiments/wandb_arm_metrics/<sweep>/:
    raw.csv       every kept (seed, model, scenario) row
    summary.csv   mean, std and n_seeds per (scenario, model, metric), plus an
                  `average` scenario: each seed first averaged over the
                  scenarios, then mean and std over seeds
"""
import argparse
import os

import pandas as pd

from src.fetch_wandb_trajectories import complete_pairs, fetch

TABLE_NAME = 'arm_metrics'
AVERAGE = 'average'


# Added to the logged table on 2026-09-30. Earlier runs have them only in the
# run summary as `<model>/<metric>`, which holds the run's per-seed value when
# the run has one seed, and a mean over seeds otherwise.
BACKFILL = [
    *[f'phase_{phase}_{metric}'
      for phase in ('pre_drift', 'transition', 'drift')
      for metric in ('accuracy', 'dp', 'eo', 'lambda')],
    'accuracy_detections', 'accuracy_true_positives', 'accuracy_false_alarms',
    'fairness_detections', 'fairness_true_positives', 'fairness_false_alarms',
]


def backfill_from_summary(raw, entity, project):
  """Fill BACKFILL columns missing from old tables, one-seed runs only."""
  import wandb
  api = wandb.Api(timeout=60)
  raw = raw.copy()
  for column in BACKFILL:
    if column not in raw.columns:
      raw[column] = float('nan')
  multi_seed = []
  for run_id, block in raw.groupby('run_id'):
    if not block[BACKFILL].isna().all().all():
      continue  # new-style table, already has them
    if block['seed'].nunique() != 1:
      multi_seed.append(run_id)
      continue
    summary = api.run(f'{entity}/{project}/{run_id}').summary
    for idx, row in block.iterrows():
      for column in BACKFILL:
        value = summary.get(f"{row['model']}/{column}")
        if isinstance(value, (int, float)):
          raw.at[idx, column] = float(value)
  if multi_seed:
    print(f'[backfill] {len(multi_seed)} run(s) hold several seeds; their summary '
          f'is a mean over seeds, so phase/detection columns stay empty for them')
  return raw


def summarise(raw, metrics):
  long = raw.melt(id_vars=['seed', 'model', 'scenario'], value_vars=metrics,
                  var_name='metric', value_name='value').dropna(subset=['value'])
  # Across-scenario average: per seed first, so the std is seed-to-seed.
  per_seed = (long.groupby(['seed', 'model', 'metric'])['value'].mean()
              .reset_index().assign(scenario=AVERAGE))
  both = pd.concat([long, per_seed], ignore_index=True)
  summary = (both.groupby(['scenario', 'model', 'metric'])['value']
             .agg(mean='mean', std='std', n_seeds='count').reset_index())
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('--sweep', required=True, help='W&B sweep id')
  parser.add_argument('--entity', default=os.environ.get(
      'WANDB_ENTITY', 'tjcsilva04-universidade-de-coimbra'))
  parser.add_argument('--project', default='fado-ablations')
  parser.add_argument('--models', default='',
                      help='Models that must all be present (default: every '
                           'model found). Only these are kept.')
  parser.add_argument('--metrics', default='',
                      help='Metrics to summarise (default: every numeric column).')
  parser.add_argument('--out', default='files/experiments/wandb_arm_metrics')
  args = parser.parse_args()

  out_dir = os.path.join(args.out, args.sweep)
  os.makedirs(out_dir, exist_ok=True)
  raw = fetch(args.entity, args.project, args.sweep,
              cache_dir=os.path.join(out_dir, 'artifacts'), table_name=TABLE_NAME)
  models = ([m.strip() for m in args.models.split(',') if m.strip()]
            or sorted(raw['model'].unique()))
  raw = complete_pairs(raw, models)
  raw = backfill_from_summary(raw, args.entity, args.project)
  if raw.empty:
    raise SystemExit(f'No (seed, scenario) has all of {models}.')
  numeric = [c for c in raw.columns
             if c not in ('seed', 'model', 'scenario', 'run', 'run_id')
             and pd.api.types.is_numeric_dtype(raw[c]) and raw[c].notna().any()]
  metrics = [m.strip() for m in args.metrics.split(',') if m.strip()] or numeric
  unknown = sorted(set(metrics) - set(numeric))
  if unknown:
    raise SystemExit(f'Unknown metric(s) {unknown}; available: {numeric}')

  summary = summarise(raw, metrics)
  raw.to_csv(os.path.join(out_dir, 'raw.csv'), index=False)
  summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)

  print(f'models: {models}')
  for scenario, n in raw.groupby('scenario')['seed'].nunique().items():
    print(f'  {scenario}: {n} complete seed(s)')
  print(f'written to {out_dir}/')


if __name__ == '__main__':
  main()
