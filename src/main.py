import json
import numbers
import os
import random
import time
import traceback

import numpy as np
from absl import app, flags

from src.drift.scenarios import SCENARIO_DESCRIPTIONS, SCENARIOS, SPLITS, PHASE_LABELS
from src.helpers.data import load_drifted_test_set, read_adult
from src.helpers.plots import plot_metrics_over_timesteps
from src.helpers.utils import get_test_performance
from src.models.arf.arf import evaluate_arf_over_timesteps
from src.models.forest.evaluator import evaluate_over_timesteps
from src.models.rfr.evaluator import evaluate_rfr_over_timesteps
from src.models.forest.train import train, FLAGS
from src.helpers.constants import (
    OUTPUT_DIR,
    DEFAULT_SEED_RUNS,
    DEFAULT_RANDOM_SEED_MIN,
    DEFAULT_RANDOM_SEED_MAX,
    RFR_CONFIG,
)

flags.DEFINE_enum(
    'pipeline_model',
    'aranyani',
    ['aranyani', 'arf', 'rfr'],
    'Model to run in main pipeline: aranyani, arf, or rfr.',
)
flags.DEFINE_string(
    'pipeline_dataset',
    '',
    'Optional dataset override for pipeline runs. Empty keeps --dataset.',
)


def _dataset_name():
    override = str(FLAGS.pipeline_dataset).strip()
    return override if override else FLAGS.dataset


def _train_kwargs(seed, dataset_name):
    return {
        'dataset': dataset_name,
        'save_model': FLAGS.save_model,
        'load_model': FLAGS.load_model,
        'model_path': FLAGS.model_path,
        'prequential': FLAGS.prequential,
        'drift_scenario': FLAGS.drift_scenario ,
        'seed': int(seed),
    }


def _parse_seed_list():
    runs = int(DEFAULT_SEED_RUNS)
    seeds = list(set(str(FLAGS.seeds).strip().split(',')))
    rng = random.SystemRandom()
    seeds = seeds + [rng.randint(DEFAULT_RANDOM_SEED_MIN, DEFAULT_RANDOM_SEED_MAX) for _ in range(max(0, runs - len(seeds)))]
    return seeds



def _stats(values):
    if not values:
        return {'mean': None, 'std': None, 'n': 0}
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {'mean': float(arr.mean()), 'std': std, 'n': int(len(arr))}


def _fmt_stats(stats):
    if stats['mean'] is None:
        return 'n/a'
    return f"{stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']})"


def _evaluate_selected_model(
    model_name,
    x_test,
    y_test,
    a_test,
    model=None,
    data_dim=None,
    x_train=None,
    y_train=None,
):
    if model_name == 'aranyani':
        stream = evaluate_over_timesteps(model, x_test, y_test, a_test, data_dim=data_dim)
        test_metrics = get_test_performance(model, x_test, y_test, a_test, data_dim=data_dim)
        return stream, test_metrics

    if model_name == 'arf':
        stream = evaluate_arf_over_timesteps(
            x_test,
            y_test,
            a_test,
            x_train=x_train,
            y_train=y_train,
        )
        test_metrics = {
            'accuracy': float(stream.get('accuracy')[-1]) if stream.get('accuracy') else None,
            'dp': float(stream.get('dp')[-1]) if stream.get('dp') else None,
            'eo': float(stream.get('eo')[-1]) if stream.get('eo') else None,
        }
        return stream, test_metrics

    if model_name == 'rfr':
        stream = evaluate_rfr_over_timesteps(
            x_test,
            y_test,
            a_test,
            approach=RFR_CONFIG['approach'],
            backbone=RFR_CONFIG['backbone'],
            hidden_dim=RFR_CONFIG['hidden_dim'],
            n_ensemble=RFR_CONFIG['n_ensemble'],
            learning_rate=RFR_CONFIG['learning_rate'],
            rho=RFR_CONFIG['rho'],
            penalty_coefficient=RFR_CONFIG['penalty_coefficient'],
            fcr_threshold=RFR_CONFIG['fcr_threshold'],
            train_batch_size=RFR_CONFIG['train_batch_size'],
            buffer_size=RFR_CONFIG['buffer_size'],
            adv_hidden_dim=RFR_CONFIG['adv_hidden_dim'],
        )
        test_metrics = {
            'accuracy': float(stream.get('accuracy')[-1]) if stream.get('accuracy') else None,
            'dp': float(stream.get('dp')[-1]) if stream.get('dp') else None,
            'eo': float(stream.get('eo')[-1]) if stream.get('eo') else None,
        }
        return stream, test_metrics

    raise ValueError(f'Unsupported model: {model_name}')


def _single_scenario(
    model_name,
    scenario_name,
    output_dir,
    model=None,
    data_dim=None,
    x_train=None,
    y_train=None,
):
    print(f"\n{'#' * 80}")
    print(f"# Evaluating scenario ({model_name}): {scenario_name}")
    print(f"# {SCENARIO_DESCRIPTIONS.get(scenario_name, '')}")
    print(f"{'#' * 80}\n")
    start = time.time()
    
    x_test, y_test, a_test = load_drifted_test_set(scenario_name)
    stream, test_metrics = _evaluate_selected_model(
        model_name=model_name,
        x_test=x_test,
        y_test=y_test,
        a_test=a_test,
        model=model,
        data_dim=data_dim,
        x_train=x_train,
        y_train=y_train,
    )

    os.makedirs(output_dir, exist_ok=True)
    # plot_path = os.path.join(output_dir, f'timesteps_{model_name}_{scenario_name}.png')
    # plot_metrics_over_timesteps(
    #     stream,
    #     save_path=plot_path,
    #     stage_splits=SPLITS,
    #     stage_labels=PHASE_LABELS,
    # )

    return {
        'model': model_name,
        'scenario': scenario_name,
        'test_metrics': test_metrics,
        'timestep_results': stream,
        'elapsed_seconds': round(time.time() - start, 1),
    }


def run_scenarios(kwargs, model_name, dataset_name, output_dir=OUTPUT_DIR, scenario_name=None):
    if str(dataset_name).lower() != 'adult':
        raise ValueError(
            f"Drift scenario evaluation currently supports dataset='adult' only; got '{dataset_name}'."
        )

    scenarios = list(SCENARIOS.keys())
    print(f"\n{'=' * 80}")
    print(f" Running all {len(scenarios)} drift scenarios")
    print(f" Model: {model_name}")
    print(f" Dataset: {dataset_name}")
    print(f" Output: {os.path.abspath(output_dir)}/")
    print(f"{'=' * 80}\n")

    trained_model = None
    data_dim = None
    arf_x_train = None
    arf_y_train = None
    if model_name == 'aranyani':
        train_start = time.time()
        print('>>> Training model on clean data (no drift)...')
        _, _, _, _, _, trained_model, data_dim = train(
            drift=False,
            **kwargs,
        )
        print(f">>> Training completed in {round(time.time() - train_start, 1)}s")
        if trained_model is None:
            raise RuntimeError('Training returned no model - cannot evaluate scenarios.')
    elif model_name == 'arf':
        train_start = time.time()
        print('>>> Loading clean train split for ARF warm-start...')
        arf_x_train, _, arf_y_train, _, _, _ = read_adult(drift=False, drift_scenario=None)
        print(f">>> Loaded {len(arf_x_train)} clean train samples in {round(time.time() - train_start, 1)}s")

    results = []
    for idx, scenario_name in enumerate(scenarios, 1):
        if scenario_name != scenario_name and scenario_name is not None:
            continue
        print(f"\n>>> [{idx}/{len(scenarios)}] {scenario_name}")
        result = _single_scenario(
            model_name=model_name,
            scenario_name=scenario_name,
            output_dir=output_dir,
            model=trained_model,
            data_dim=data_dim,
            x_train=arf_x_train,
            y_train=arf_y_train,
        )
        results.append(result)
        tm = result['test_metrics']
        print(tm)
        print(
            f"Done in {result['elapsed_seconds']}s | "
            f"Acc={float(tm.get('accuracy'))} "
            f"DP={float(tm.get('dp'))} "
            f"EO={float(tm.get('eo'))}"
        )

    rows = []
    for result in results:
        tm = result.get('test_metrics') or {}
        ts = result.get('timestep_results') or {}
        rows.append({
            'model': result.get('model'),
            'scenario': result.get('scenario'),
            'accuracy': float(tm.get('accuracy')),
            'dp': float(tm.get('dp')),
            'eo': float(tm.get('eo')),
            'stream_final_accuracy': ts.get('accuracy'),
            'stream_final_dp': ts.get('dp'),
            'stream_final_eo': ts.get('eo'),
            'elapsed_seconds': result.get('elapsed_seconds'),
            'error': result.get('error'),
        })

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")

    print(f"\n{'=' * 80}")
    print(' SUMMARY')
    print(f"{'=' * 80}")
    print(f"{'Scenario':<22s} {'Acc':>10s} {'DP':>10s} {'EO':>10s}")
    print('-' * 80)
    for row in rows:
        acc = f"{row['accuracy']:.4f}"
        dp = f"{row['dp']:.4f}"
        eo = f"{row['eo']:.4f}"
        print(f"{str(row['scenario']):<22s} {acc:>10s} {dp:>10s} {eo:>10s}")

    return rows


def _aggregate_metrics(rows, metric_names):
    def _coerce_metric_values(value):
        if value is None:
            return []
        if isinstance(value, numbers.Number):
            return [float(value)]
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return []
            return [float(v) for v in value.reshape(-1) if isinstance(v, numbers.Number)]
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(_coerce_metric_values(item))
            return out
        return []

    flat_rows = []
    for row in rows:
        if isinstance(row, dict):
            flat_rows.append(row)
        elif isinstance(row, list):
            flat_rows.extend(r for r in row if isinstance(r, dict))

    aggregated = {}
    for metric_name in metric_names:
        values = []
        for row in flat_rows:
            value = row.get(metric_name)
            values.extend(_coerce_metric_values(value))
        aggregated[metric_name] = _stats(values)
    return aggregated


def main(_):
    dataset_name = _dataset_name()
    model_name = str(FLAGS.pipeline_model).lower()
    seeds = _parse_seed_list()

    print(f"\n{'=' * 80}")
    print(f" Multi-seed pipeline: {len(seeds)} runs")
    print(f" Model: {model_name}")
    print(f" Dataset: {dataset_name}")
    print(f" Seeds: {seeds}")
    print(f" Base output: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'=' * 80}\n")

    seed_runs = []
    base_output_dir = os.path.join(OUTPUT_DIR, f'model_{model_name}', f'dataset_{dataset_name}')
    for idx, seed in enumerate(seeds, 1):
        print(f"\n{'-' * 80}")
        print(f" Seed run [{idx}/{len(seeds)}]: {seed}")
        print(f"{'-' * 80}")
        kwargs = _train_kwargs(seed=seed, dataset_name=dataset_name)
        seed_output_dir = os.path.join(base_output_dir, f'seed_{seed}')

        results = run_scenarios(
            kwargs=kwargs,
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=seed_output_dir,
            scenario_name=FLAGS.drift_scenario if FLAGS.drift_scenario else None
        )
        seed_runs.append({'seed': seed, 'results': results})

    if FLAGS.run_all_scenarios:
        metric_names = [
            'accuracy', 'dp', 'eo',
            'stream_final_accuracy', 'stream_final_dp', 'stream_final_eo',
        ]
        grouped = {}
        for run in seed_runs:
            for row in run['results']:
                grouped.setdefault(row['scenario'], []).append(row)

        summary_rows = []
        for scenario_name in sorted(grouped.keys()):
            summary_rows.append({
                'scenario': scenario_name,
                'metrics': _aggregate_metrics(grouped[scenario_name], metric_names),
            })
        summary = {'mode': 'all_scenarios', 'rows': summary_rows}
    else:
        metric_names = [
            'accuracy', 'dp', 'eo',
            'stream_final_accuracy', 'stream_final_dp', 'stream_final_eo',
        ]
        rows = [
            row
            for run in seed_runs
            for row in run.get('results', [])
            if isinstance(row, dict)
        ]
        summary = {'mode': 'single', 'metrics': _aggregate_metrics(rows, metric_names)}

    payload = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'dataset': dataset_name,
        'seeds': seeds,
        'run_all_scenarios': bool(FLAGS.run_all_scenarios),
        'seed_runs': seed_runs,
        'summary': summary,
    }

    os.makedirs(base_output_dir, exist_ok=True)
    output_path = os.path.join(base_output_dir, 'seed_pipeline_results.json')
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSeed pipeline results saved to: {output_path}")

    print(f"\n{'=' * 80}")
    print(' Seed Pipeline Summary')
    print(f"{'=' * 80}")
    if summary['mode'] == 'all_scenarios':
        print(f"{'Scenario':<22s} {'Acc':>28s} {'DP':>28s} {'EO':>28s}")
        print('-' * 120)
        for row in summary['rows']:
            print(
                f"{row['scenario']:<22s} "
                f"{_fmt_stats(row['metrics']['accuracy']):>28s} "
                f"{_fmt_stats(row['metrics']['dp']):>28s} "
                f"{_fmt_stats(row['metrics']['eo']):>28s}"
            )
    else:
        for metric_name, metric_stats in summary['metrics'].items():
            print(f"{metric_name:<24s} {_fmt_stats(metric_stats)}")



if __name__ == '__main__':
    app.run(main)
