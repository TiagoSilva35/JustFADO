"""Lambda and rolling DP over one stream (decision log 2.1).

Top panel: rolling DP for each arm, with the DP target epsilon and the phase
boundaries. Bottom panel: the FADO arm's lambda_t, with the samples where the
fairness statistics were reset. Reads one ``seed_pipeline_results.json``.

    python -m src.plot_lambda_trajectory --results <path> --seed 11 \\
        --out DOCS/presentation/second_meating/assets/lambda_trajectory.pdf
"""

import argparse
import json

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src.drift import compas_scenarios, scenarios  # noqa: E402

# Validated default categorical palette (dataviz skill), fixed order.
STYLE = {
    'aranyani': ('FADO (λ controller)', '#2a78d6'),
    'fado_no_lambda': ('FADO, λ fixed', '#1baf7a'),
    'aranyani_base': ('Aranyani-Base', '#eb6834'),
    'fado_no_reset': ('FADO, no reset', '#eda100'),
}
TEXT, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _style_axis(ax):
  ax.grid(True, color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  for side in ('left', 'bottom'):
    ax.spines[side].set_color(GRID)
  ax.tick_params(colors=MUTED, labelsize=8)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('--results', required=True)
  parser.add_argument('--seed', type=int, required=True)
  parser.add_argument('--scenario', default=None)
  parser.add_argument('--models', default='aranyani,fado_no_lambda,aranyani_base')
  parser.add_argument('--out', required=True)
  args = parser.parse_args()

  with open(args.results) as f:
    payload = json.load(f)
  dataset = payload.get('dataset')
  module = compas_scenarios if dataset == 'compas' else scenarios
  rows = {}
  for run in payload['seed_runs']:
    if run['seed'] != args.seed:
      continue
    for row in run['results']:
      if args.scenario in (None, row['scenario']):
        rows[run['model']] = row
  models = [m for m in args.models.split(',') if m in rows]
  if not models:
    raise SystemExit(f'No rows for seed {args.seed} in {args.results}.')
  scenario = rows[models[0]]['scenario']

  fig, (ax_dp, ax_lam) = plt.subplots(
      2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
  n = 0
  epsilon = None
  for model in models:
    ts = rows[model]['timestep_results']
    dp = np.asarray(ts['dp'], dtype=float)
    n = max(n, len(dp))
    label, color = STYLE.get(model, (model, MUTED))
    ax_dp.plot(dp, color=color, linewidth=2 if model == 'aranyani' else 1.5,
               label=label)
    used = ts.get('static_params_used') or {}
    if model == 'aranyani':
      epsilon = used.get('fairness_target')
      lam = np.asarray(ts.get('lambda') or [], dtype=float)
      ax_lam.plot(lam, color=color, linewidth=2)
      base = used.get('lambda_const')
      if base is not None:
        ax_lam.axhline(base, color=MUTED, linewidth=1, linestyle='--')
        ax_lam.text(n * 0.995, base, f'λ_base = {base:g}', fontsize=8,
                    color=MUTED, ha='right', va='bottom')
      for i, t in enumerate(ts.get('fairness_resets') or []):
        ax_lam.axvline(t, color=MUTED, linewidth=0.8, linestyle=':',
                       label='fairness-stat reset' if i == 0 else None)

  if epsilon is not None:
    ax_dp.axhline(epsilon, color=MUTED, linewidth=1, linestyle='--')
    ax_dp.text(n * 0.995, epsilon, f'ε = {epsilon:g}', fontsize=8, color=MUTED,
               ha='right', va='bottom')
  for split, label in zip(module.SPLITS[:-1], module.PHASE_LABELS[1:]):
    for ax in (ax_dp, ax_lam):
      ax.axvline(int(split * n), color=TEXT, linewidth=0.8, alpha=0.35)
    ax_dp.text(int(split * n) + n * 0.005, ax_dp.get_ylim()[1], label.lower(),
               fontsize=8, color=MUTED, va='top')

  ax_dp.set_ylabel('rolling DP', fontsize=9, color=MUTED)
  ax_lam.set_ylabel('λ_t', fontsize=9, color=MUTED)
  ax_lam.set_xlabel('sample in the prequential stream', fontsize=9, color=MUTED)
  for ax in (ax_dp, ax_lam):
    _style_axis(ax)
  ax_dp.legend(frameon=False, fontsize=9, ncol=len(models), loc='upper left',
               bbox_to_anchor=(0, 1.22))
  if ax_lam.get_legend_handles_labels()[0]:
    ax_lam.legend(frameon=False, fontsize=8, loc='upper right')
  fig.suptitle(f'{dataset} · {scenario} · seed {args.seed}', fontsize=10,
               color=TEXT, x=0.01, ha='left')
  fig.tight_layout()
  fig.savefig(args.out, dpi=200, bbox_inches='tight')
  print(f'wrote {args.out}')


if __name__ == '__main__':
  main()
