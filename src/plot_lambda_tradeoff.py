"""Accuracy-vs-DP trade-off curves over the lambda grid (decision 3.2).

No lambda is selected. Each Aranyani arm (FADO, Aranyani-Base) is drawn as a
curve through every lambda it ran with; ARF and RFR, which have no lambda, are
single reference points. The claim to read off is whether FADO's curve lies
beyond Base's -- up and to the left, i.e. more accurate at equal DP.

Two panels per (dataset, scenario), per decision 2.4: the whole-stream means,
and the same curves restricted to the post-drift part of the stream.

Reads every ``seed_pipeline_results.json`` under ``--inputs`` (a sweep run
writes one under ``files/experiments/wandb/<run id>/``), so run it after the
``sweep_lambda_budget_*`` and ``sweep_reference_*`` sweeps:

    python -m src.plot_lambda_tradeoff --inputs files/experiments/wandb

Points are the mean over seeds, bars one standard deviation. Runs are grouped
by their configuration apart from lambda and scenario; if the inputs mix
configurations (say an architecture-sweep run with depth=6) the script stops
and names the differing keys, unless ``--config key=value`` narrows them.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


LAMBDA_ARMS = ('aranyani', 'aranyani_base')
REFERENCE_ARMS = ('arf', 'rfr')

# Fixed order from the validated default categorical palette (dataviz skill,
# slots 1-4); each arm also has its own marker so identity is not colour alone.
STYLE = {
    'aranyani': {'label': 'FADO', 'color': '#2a78d6', 'marker': 'o'},
    'aranyani_base': {'label': 'Aranyani-Base', 'color': '#eb6834', 'marker': 's'},
    'arf': {'label': 'ARF', 'color': '#1baf7a', 'marker': '^'},
    'rfr': {'label': 'RFR', 'color': '#eda100', 'marker': 'D'},
}
TEXT = '#0b0b0b'
TEXT_MUTED = '#52514e'
GRID = '#e4e3df'

PANELS = (
    ('whole stream', 'stream_final_accuracy', 'stream_final_dp'),
    ('post-drift', 'post_drift_accuracy', 'post_drift_dp'),
)

# Config keys that are allowed to differ between runs plotted together.
VARYING_KEYS = {'lambda_const', 'drift_scenario', 'models', 'seeds',
                'wandb_run_id', 'wandb_sweep_id'}


def _signature(config):
  """The configuration a run shares with the others on one plot."""
  sig = {k: v for k, v in config.items() if k not in VARYING_KEYS}
  static = dict(sig.pop('static_params', None) or {})
  static.pop('lambda_const', None)
  sig.update({f'static.{k}': v for k, v in static.items()})
  return {k: json.dumps(v, sort_keys=True) for k, v in sig.items()}


def _parse_filters(pairs):
  filters = {}
  for pair in pairs or []:
    key, _, value = pair.partition('=')
    filters[key.strip()] = value.strip()
  return filters


def _matches(signature, filters):
  return all(json.loads(signature.get(k, 'null')) == _parse_value(v)
             for k, v in filters.items())


def _parse_value(raw):
  """'4' -> 4, 'true' -> True, 'abc' -> 'abc'."""
  try:
    return json.loads(raw)
  except ValueError:
    return raw


def load_points(inputs, filters):
  """``points[(dataset, scenario, model, lam)][seed] = {metric: value}``."""
  paths = sorted(glob(os.path.join(inputs, '**', 'seed_pipeline_results.json'),
                      recursive=True))
  points = defaultdict(dict)
  signatures = defaultdict(list)  # dataset -> [(path, signature)]
  skipped = 0
  for path in paths:
    with open(path) as f:
      payload = json.load(f)
    config = payload.get('config')
    if not config:
      skipped += 1
      continue
    signature = _signature(config)
    if not _matches(signature, filters):
      continue
    dataset = payload.get('dataset')
    signatures[dataset].append((path, signature))
    for run in payload.get('seed_runs') or []:
      model = run.get('model')
      if model not in LAMBDA_ARMS + REFERENCE_ARMS:
        continue
      for row in run.get('results') or []:
        lam = row.get('lambda_const') if model in LAMBDA_ARMS else None
        if model in LAMBDA_ARMS and lam is None:
          continue
        key = (dataset, row.get('scenario'), model, lam)
        seed = run.get('seed')
        if seed in points[key]:
          continue  # the same cell from a second run; keep the first
        points[key][seed] = {m: row.get(m) for _, a, d in PANELS for m in (a, d)}
  if skipped:
    print(f'Skipped {skipped} result file(s) without a saved config '
          '(written before configs were recorded).')
  _check_one_config(signatures)
  return points


def _check_one_config(signatures):
  for dataset, entries in signatures.items():
    keys = set().union(*(sig.keys() for _, sig in entries))
    differing = sorted(k for k in keys
                       if len({sig.get(k) for _, sig in entries}) > 1)
    if differing:
      raise SystemExit(
          f'{dataset}: runs differ in {differing}. Narrow them with '
          '--config key=value (e.g. --config depth=4), or point --inputs at '
          'the lambda-budget and reference runs only.'
      )


def _summarise(seed_values, metric):
  values = [v[metric] for v in seed_values.values()
            if isinstance(v.get(metric), (int, float))]
  if not values:
    return None, None, 0
  return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, len(values)


def build_table(points):
  rows = []
  for (dataset, scenario, model, lam), seed_values in sorted(
      points.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or 0)):
    row = {'dataset': dataset, 'scenario': scenario, 'model': model,
           'lambda_const': lam}
    for _, acc_key, dp_key in PANELS:
      for metric in (acc_key, dp_key):
        mean, std, n = _summarise(seed_values, metric)
        row[f'{metric}_mean'] = mean
        row[f'{metric}_std'] = std
        row['n_seeds'] = n
    rows.append(row)
  return rows


def _plot_panel(ax, rows, title, acc_key, dp_key):
  for model in LAMBDA_ARMS + REFERENCE_ARMS:
    style = STYLE[model]
    arm = [r for r in rows if r['model'] == model
           and r[f'{acc_key}_mean'] is not None and r[f'{dp_key}_mean'] is not None]
    if not arm:
      continue
    arm.sort(key=lambda r: r['lambda_const'] or 0)
    x = [r[f'{dp_key}_mean'] for r in arm]
    y = [r[f'{acc_key}_mean'] for r in arm]
    ax.errorbar(
        x, y, xerr=[r[f'{dp_key}_std'] for r in arm],
        yerr=[r[f'{acc_key}_std'] for r in arm],
        color=style['color'], marker=style['marker'],
        markersize=6 if model in LAMBDA_ARMS else 8,
        linewidth=2 if model in LAMBDA_ARMS else 0,
        elinewidth=0.8, capsize=0, alpha=0.95, label=style['label'],
        markeredgecolor='white', markeredgewidth=1.2, zorder=3,
    )
    if model in LAMBDA_ARMS:
      # Label only the two ends of each curve; the direction of lambda
      # is then readable without a number on every point.
      # FADO labels sit above its points, Base labels below, so the two
      # curves' labels do not collide where the curves run close.
      offset = (6, 5) if model == 'aranyani' else (6, -13)
      for r in (arm[0], arm[-1]) if len(arm) > 1 else arm:
        ax.annotate(f"λ={r['lambda_const']:g}",
                    (r[f'{dp_key}_mean'], r[f'{acc_key}_mean']),
                    textcoords='offset points', xytext=offset,
                    fontsize=8, color=TEXT_MUTED)
  ax.set_title(title, fontsize=10, color=TEXT, loc='left')
  ax.set_xlabel('demographic parity gap (lower is fairer)', fontsize=9, color=TEXT_MUTED)
  ax.set_ylabel('prequential accuracy', fontsize=9, color=TEXT_MUTED)
  ax.text(0.02, 0.97, '↖ better', transform=ax.transAxes, fontsize=8,
          color=TEXT_MUTED, va='top')
  ax.grid(True, color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  for side in ('left', 'bottom'):
    ax.spines[side].set_color(GRID)
  ax.tick_params(colors=TEXT_MUTED, labelsize=8)


def plot(table, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  written = []
  for dataset, scenario in sorted({(r['dataset'], r['scenario']) for r in table}):
    rows = [r for r in table if r['dataset'] == dataset and r['scenario'] == scenario]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(10, 4.2))
    for ax, (title, acc_key, dp_key) in zip(np.atleast_1d(axes), PANELS):
      _plot_panel(ax, rows, title, acc_key, dp_key)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    n_seeds = max((r['n_seeds'] for r in rows), default=0)
    fig.suptitle(f'{dataset} · {scenario} · mean ± sd over {n_seeds} seeds',
                 fontsize=10, color=TEXT, x=0.01, y=0.99, ha='left')
    fig.legend(handles, labels, loc='upper left', ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.0, 0.96))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    stem = os.path.join(out_dir, f'lambda_tradeoff_{dataset}_{scenario}')
    for ext in ('png', 'pdf'):
      fig.savefig(f'{stem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    written.append(f'{stem}.png')
  return written


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('--inputs', default='files/experiments/wandb',
                      help='Directory searched recursively for seed_pipeline_results.json.')
  parser.add_argument('--out', default='files/plots/lambda_tradeoff',
                      help='Where the figures and the CSV are written.')
  parser.add_argument('--config', action='append', metavar='KEY=VALUE',
                      help='Keep only runs whose saved config has KEY=VALUE '
                           '(repeatable; static params as static.<name>).')
  args = parser.parse_args()

  points = load_points(args.inputs, _parse_filters(args.config))
  if not points:
    raise SystemExit(f'No lambda-budget or reference results under {args.inputs}.')
  table = build_table(points)

  os.makedirs(args.out, exist_ok=True)
  csv_path = os.path.join(args.out, 'lambda_tradeoff.csv')
  with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
    writer.writeheader()
    writer.writerows(table)
  for path in plot(table, args.out):
    print(f'wrote {path}')
  print(f'wrote {csv_path}')


if __name__ == '__main__':
  main()
