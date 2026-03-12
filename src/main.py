import os
import json
import time
import traceback

import tensorflow as tf

# Configure GPU memory growth before any TensorFlow operations
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

from absl import app
from src.forest.train import *
from src.helpers.data import load_drifted_test_set
from src.drift.scenarios import SCENARIOS, SCENARIO_DESCRIPTIONS
from src.helpers.utils import evaluate_over_timesteps, evaluate_arf_over_timesteps, get_test_performance
from src.helpers.plots import (
    plot_metrics_over_timesteps,
    plot_aranyani_vs_arf,
)

OUTPUT_DIR = 'files/experiments'


def _common_train_kwargs():
  """Collect the current flag values into a dict for train()."""
  return dict(
      lambda_const=FLAGS.lambda_const,
      dataset=FLAGS.dataset,
      max_iter=FLAGS.max_iter,
      depth=FLAGS.depth,
      num_trees=FLAGS.num_trees,
      compute_fairness=FLAGS.compute_fairness and FLAGS.lambda_const > 0,
      batch_size=FLAGS.batch_size,
      activation=FLAGS.activation,
      compute_mode=FLAGS.compute_mode,
      base_gamma=FLAGS.base_gamma,
      constraint_type=FLAGS.constraint_type,
      gradient_type=FLAGS.gradient_type,
      encoder_model=FLAGS.encoder_model,
      offline_loss_type=FLAGS.offline_loss_type,
      local_run=True,
      save_model=FLAGS.save_model,
      load_model=FLAGS.load_model,
      model_path=FLAGS.model_path,
      prequential=FLAGS.prequential,
  )


def _evaluate_scenario(model, data_dim, scenario_name):
  """Evaluate an already-trained model on one drift scenario's test set."""
  print(f"\n{'#'*80}")
  print(f"# Evaluating scenario: {scenario_name}")
  print(f"# {SCENARIO_DESCRIPTIONS.get(scenario_name, '')}")
  print(f"{'#'*80}\n")

  start = time.time()
  x_test, y_test, a_test = load_drifted_test_set(scenario_name)

  timestep_results = evaluate_over_timesteps(
      model, x_test, y_test, a_test, data_dim=data_dim,
  )
  test_metrics = get_test_performance(
      model, x_test, y_test, a_test, data_dim=data_dim,
  )

  # Run ARF baseline on the same stream
  arf_results = evaluate_arf_over_timesteps(x_test, y_test, a_test)

  elapsed = round(time.time() - start, 1)

  # Save per-scenario timestep plots
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  if timestep_results is not None:
    per_path = os.path.join(OUTPUT_DIR, f'timesteps_{scenario_name}.png')
    plot_metrics_over_timesteps(timestep_results, save_path=per_path)
    cmp_path = os.path.join(OUTPUT_DIR, f'aranyani_vs_arf_{scenario_name}.png')
    plot_aranyani_vs_arf(timestep_results, arf_results, save_path=cmp_path,
                         scenario_name=scenario_name)

  return {
      'scenario': scenario_name,
      'test_metrics': test_metrics,
      'timestep_results': timestep_results,
      'arf_results': arf_results,
      'elapsed_seconds': elapsed,
  }


def run_all_scenarios(kwargs):
  """Train Aranyani ONCE, then evaluate on every registered drift scenario."""
  scenario_names = list(SCENARIOS.keys())
  total = len(scenario_names)

  print(f"\n{'='*80}")
  print(f" Aranyani – Running all {total} drift scenarios")
  print(f" Scenarios: {scenario_names}")
  print(f" Output:    {os.path.abspath(OUTPUT_DIR)}/")
  print(f"{'='*80}\n")

  # ── Train once on clean (undrifted) data ────────────────────────────
  print(">>> Training model on clean data (no drift) ...")
  train_start = time.time()
  dps, accs, eos, _, _, trained_model, data_dim = train(
      drift=False,
      drift_scenario=None,
      **kwargs,
  )
  train_elapsed = round(time.time() - train_start, 1)
  print(f">>> Training completed in {train_elapsed}s\n")

  if trained_model is None:
    raise RuntimeError("Training returned no model — cannot evaluate scenarios.")

  # ── Evaluate on each drifted test set ───────────────────────────────
  all_results = []
  for idx, scenario in enumerate(scenario_names, 1):
    print(f"\n>>> [{idx}/{total}] {scenario}")
    try:
      result = _evaluate_scenario(trained_model, data_dim, scenario)
      all_results.append(result)
      print(f"    Done in {result['elapsed_seconds']}s")
      if result['test_metrics']:
        tm = result['test_metrics']
        print(f"    Acc={tm['accuracy']:.4f}  DP={tm['dp']:.4f}  "
              f"EO={tm['eo']:.4f}  F1={tm['f1']:.4f}")
    except Exception as e:
      print(f"    FAILED: {e}")
      traceback.print_exc()
      all_results.append({
          'scenario': scenario,
          'test_metrics': None,
          'timestep_results': None,
          'error': str(e),
      })

  # ── Comparison plots ────────────────────────────────────────────────
  print(f"\n{'='*80}")
  print(" Generating comparison plots ...")
  print(f"{'='*80}\n")

 
  # ── Save results JSON ───────────────────────────────────────────────
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  json_results = []
  for r in all_results:
    jr = {k: v for k, v in r.items() if k != 'timestep_results'}
    ts = r.get('timestep_results')
    if ts:
      jr['final_accuracy'] = ts['accuracy'][-1] if ts['accuracy'] else None
      jr['final_dp'] = ts['dp'][-1] if ts['dp'] else None
      jr['final_eo'] = ts['eo'][-1] if ts['eo'] else None
    json_results.append(jr)

  json_path = os.path.join(OUTPUT_DIR, 'results.json')
  with open(json_path, 'w') as f:
    json.dump(json_results, f, indent=2, default=str)
  print(f"Results saved to: {json_path}")

  # ── Summary table ───────────────────────────────────────────────────
  print(f"\n{'='*80}")
  print(f" SUMMARY")
  print(f"{'='*80}")
  print(f"{'Scenario':<20s} {'Acc':>7s} {'DP':>7s} {'EO':>7s} {'F1':>7s} {'Time':>7s}")
  print('-' * 60)
  for r in all_results:
    tm = r.get('test_metrics')
    if tm:
      print(f"{r['scenario']:<20s} "
            f"{tm['accuracy']:>7.4f} {tm['dp']:>7.4f} {tm['eo']:>7.4f} "
            f"{tm['f1']:>7.4f} {r.get('elapsed_seconds', 0):>6.1f}s")
    else:
      err = r.get('error', 'N/A')
      print(f"{r['scenario']:<20s}  FAILED ({err[:40]})")
  print(f"{'='*80}")
  print(f"All plots saved to: {os.path.abspath(OUTPUT_DIR)}/")


def main(_):
  kwargs = _common_train_kwargs()

  if FLAGS.run_all_scenarios:
    run_all_scenarios(kwargs)
  else:
    # Single run (original behaviour)
    drift_on = FLAGS.drift or (FLAGS.drift_scenario is not None)
    _, _, _, timestep_results, _, _, data_dim = train(
        drift=drift_on,
        drift_scenario=FLAGS.drift_scenario,
        **kwargs,
    )
    # If prequential evaluation ran, also run ARF on the same stream and
    # save a head-to-head comparison plot.
    if timestep_results is not None and FLAGS.drift_scenario:
      scenario = FLAGS.drift_scenario
      x_test, y_test, a_test = load_drifted_test_set(scenario)
      arf_results = evaluate_arf_over_timesteps(x_test, y_test, a_test)
      os.makedirs(OUTPUT_DIR, exist_ok=True)
      cmp_path = os.path.join(OUTPUT_DIR, f'aranyani_vs_arf_{scenario}.png')
      plot_aranyani_vs_arf(timestep_results, arf_results,
                           save_path=cmp_path, scenario_name=scenario)
      print(f"Aranyani vs ARF comparison saved to: {cmp_path}")


if __name__ == '__main__':
  app.run(main)