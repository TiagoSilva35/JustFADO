"""Download the per-run `metrics_over_time` tables of a W&B sweep and average
them over seeds.

Each run logs its trajectories as a W&B table (`metrics_over_time`: seed,
model, scenario, timestep, accuracy, dp, eo; 300 bins per stream). This
script pulls those tables for every finished run of a sweep, keeps only the
(seed, scenario) pairs where EVERY model finished, and averages over seeds per
(scenario, model, bin).

    python -m src.fetch_wandb_trajectories --sweep ych8qely
    python -m src.fetch_wandb_trajectories --sweep ych8qely --models fado,aranyani --plot

Writes to files/experiments/wandb_trajectories/<sweep>/:
    raw.csv    every bin of every kept (seed, model, scenario)
    mean.csv   mean and std over seeds per (scenario, model, bin), with n_seeds
    <scenario>_<metric>.png   with --plot

Bins are matched by index, not by timestep: streams of skip-type drifts lose
rows, so seeds can have different lengths. `timestep` in mean.csv is the mean
over seeds. The injected-drift onset also differs per seed (50-70% of the
stream), so the seed mean smooths the drift over that range.
"""
import argparse
import os

import numpy as np
import pandas as pd

TABLE_NAME = 'metrics_over_time'
# Runs from before the table was binned logged the full per-sample table under
# this name, which W&B truncated to 10,000 rows: those are skipped.
OLD_TABLE_NAME = 'accuracy_dp_over_time'
METRICS = ('accuracy', 'dp', 'eo')


def _run_table(run, cache_dir, table_name=TABLE_NAME):
  for artifact in run.logged_artifacts():
    if artifact.type != 'run_table':
      continue
    if artifact.name.split(':')[0].endswith(f'-{table_name}'):
      artifact.download(root=os.path.join(cache_dir, run.id))
      return artifact.get(table_name).get_dataframe()
  return None


def fetch(entity, project, sweep_id, cache_dir, table_name=TABLE_NAME):
  """Every finished run's `table_name` table, stacked, with a `run` column."""
  # artifact.get() downloads into ./artifacts unless told otherwise.
  os.environ['WANDB_ARTIFACT_DIR'] = cache_dir
  import wandb
  api = wandb.Api(timeout=60)
  sweep = api.sweep(f'{entity}/{project}/{sweep_id}')
  frames, skipped = [], []
  for run in sweep.runs:
    if run.state != 'finished':
      skipped.append((run.name, run.state))
      continue
    df = _run_table(run, cache_dir, table_name)
    if df is None:
      skipped.append((run.name, f'no {table_name} table (old run?)'))
      continue
    df['run'] = run.name
    df['run_id'] = run.id
    frames.append(df)
  for name, reason in skipped:
    print(f'[skip] {name}: {reason}')
  if not frames:
    raise SystemExit(f'No finished runs with a {table_name} table.')
  return pd.concat(frames, ignore_index=True)


def complete_pairs(raw, models):
  """Keep (seed, scenario) pairs where every requested model has a curve."""
  have = raw.groupby(['seed', 'scenario'])['model'].agg(set)
  keep = have[have.apply(lambda s: set(models) <= s)].index
  kept = raw.set_index(['seed', 'scenario']).loc[keep].reset_index()
  dropped = sorted(set(have.index) - set(keep))
  for seed, scenario in dropped:
    missing = sorted(set(models) - have[(seed, scenario)])
    print(f'[drop] seed {seed}, {scenario}: missing {missing}')
  return kept[kept['model'].isin(models)]


def average(raw):
  raw = raw.sort_values('timestep').copy()
  raw['bin'] = raw.groupby(['seed', 'scenario', 'model']).cumcount()
  # Seeds of different stream length have different bin counts; average only
  # the bins every seed has.
  n_bins = raw.groupby(['seed', 'scenario', 'model'])['bin'].max().groupby(
      ['scenario', 'model']).min() + 1
  raw = raw.merge(n_bins.rename('n_bins').reset_index(), on=['scenario', 'model'])
  raw = raw[raw['bin'] < raw['n_bins']]
  agg = {'timestep': 'mean', 'seed': 'nunique'}
  agg.update({m: ['mean', 'std'] for m in METRICS})
  mean = raw.groupby(['scenario', 'model', 'bin']).agg(agg)
  mean.columns = [
      'timestep' if c == ('timestep', 'mean') else
      'n_seeds' if c == ('seed', 'nunique') else f'{c[0]}_{c[1]}'
      for c in mean.columns
  ]
  return mean.reset_index()


def plot(mean, out_dir):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  labels = {'accuracy': 'Accuracy', 'dp': 'DP', 'eo': 'EO'}
  for scenario, block in mean.groupby('scenario'):
    for metric in METRICS:
      fig, ax = plt.subplots(figsize=(8, 4))
      for model, curve in block.groupby('model'):
        m, s = curve[f'{metric}_mean'], curve[f'{metric}_std'].fillna(0)
        line, = ax.plot(curve['timestep'], m, label=model, linewidth=1.4)
        ax.fill_between(curve['timestep'], m - s, m + s,
                        color=line.get_color(), alpha=0.12, linewidth=0)
      n = int(block['n_seeds'].min())
      ax.set_title(f'{labels[metric]} over time: {scenario} (mean ± std, {n} seeds)')
      ax.set_xlabel('timestep')
      ax.set_ylabel(labels[metric])
      ax.legend(frameon=False)
      ax.grid(alpha=0.3)
      fig.tight_layout()
      fig.savefig(os.path.join(out_dir, f'{scenario}_{metric}.png'), dpi=150)
      plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('--sweep', required=True, help='W&B sweep id')
  parser.add_argument('--entity', default=os.environ.get(
      'WANDB_ENTITY', 'tjcsilva04-universidade-de-coimbra'))
  parser.add_argument('--project', default='fado-ablations')
  parser.add_argument('--models', default='',
                      help='Models that must all be present (default: every '
                           'model found). Only these are kept.')
  parser.add_argument('--out', default='files/experiments/wandb_trajectories')
  parser.add_argument('--plot', action='store_true')
  args = parser.parse_args()

  out_dir = os.path.join(args.out, args.sweep)
  os.makedirs(out_dir, exist_ok=True)
  raw = fetch(args.entity, args.project, args.sweep,
              cache_dir=os.path.join(out_dir, 'artifacts'))
  models = ([m.strip() for m in args.models.split(',') if m.strip()]
            or sorted(raw['model'].unique()))
  raw = complete_pairs(raw, models)
  if raw.empty:
    raise SystemExit(f'No (seed, scenario) has all of {models}.')
  mean = average(raw)

  raw.to_csv(os.path.join(out_dir, 'raw.csv'), index=False)
  mean.to_csv(os.path.join(out_dir, 'mean.csv'), index=False)
  seeds = raw.groupby('scenario')['seed'].nunique()
  print(f'models: {models}')
  for scenario, n in seeds.items():
    print(f'  {scenario}: {n} complete seed(s)')
  if args.plot:
    plot(mean, out_dir)
  print(f'written to {out_dir}/')


if __name__ == '__main__':
  main()
