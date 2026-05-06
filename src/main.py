import json
import numbers
import os
import random
import time
import traceback

import numpy as np
from absl import app, flags

from src.drift.scenarios import SCENARIO_DESCRIPTIONS, SCENARIOS
from src.helpers.data import load_drifted_test_set, read_adult, read_diabetes, read_folktables
from src.models.arf.arf import evaluate_arf_over_timesteps
from src.models.forest.evaluator import evaluate_over_timesteps
from src.models.rfr.evaluator import evaluate_rfr_over_timesteps
from src.models.forest.train import train, FLAGS
import src.models.forest.forest as forest

from src.helpers.constants import (
    
    OUTPUT_DIR,
    DEFAULT_SEED_RUNS,
    DEFAULT_RANDOM_SEED_MIN,
    DEFAULT_RANDOM_SEED_MAX,
    RFR_CONFIG,
)

flags.DEFINE_string(
    'pipeline_model',
    '',
    'Optional model to run in main pipeline: aranyani, arf, or rfr. Empty runs all supported models for the dataset.',
)
flags.DEFINE_string(
    'pipeline_dataset',
    '',
    'Optional dataset override for pipeline runs. Empty keeps --dataset.',
)
flags.DEFINE_string(
    'diabetes_path',
    'data/diabetes/diabetic_data.csv',
    'Path, directory, or glob for diabetes CSV files. Defaults to diabetic_data.csv.',
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
        'seed': FLAGS.seed,
    }


def _supported_models_for_dataset(dataset_name):
    dataset_key = str(dataset_name).strip().lower()
    supported = {
        'adult': ['aranyani', 'arf', 'rfr'],
        'folktables': ['aranyani', 'arf', 'rfr'],
        'diabetes': ['aranyani', 'arf', 'rfr'],
    }
    if dataset_key not in supported:
        raise ValueError(
            f"Unsupported dataset for pipeline evaluation: '{dataset_name}'."
        )
    return supported[dataset_key]


def _resolve_pipeline_models(dataset_name):
    requested_model = str(FLAGS.pipeline_model).strip().lower()
    supported_models = _supported_models_for_dataset(dataset_name)
    if not requested_model:
        return supported_models
    if requested_model not in supported_models:
        raise ValueError(
            f"Model '{requested_model}' is not supported for dataset '{dataset_name}'. "
            f"Supported models: {supported_models}"
        )
    return [requested_model]


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
    data_dim=None,
    seed=None,
):
    if model_name == 'aranyani':
        model = forest.FairDecisionForest(
            num_trees=4,
            tree_depth=9,
            data_dim=data_dim,
            num_classes=2,
        )
        print("Evaluating Aranyani...")
        stream = evaluate_over_timesteps(model, x_test, y_test, a_test, data_dim=data_dim, lambda_const=0.1)
        test_metrics = {
            'accuracy': float(stream.get('accuracy')[-1]) if stream.get('accuracy') else None,
            'dp': float(stream.get('dp')[-1]) if stream.get('dp') else None,
            'eo': float(stream.get('eo')[-1]) if stream.get('eo') else None,
        }
        return stream, test_metrics

    if model_name == 'arf':
        stream = evaluate_arf_over_timesteps(
            x_test,
            y_test,
            a_test,
            seed=seed,
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
    data_dim=None,

    x_test=None,
    y_test=None,
    a_test=None,
    seed=None,
):
    print(f"\n{'#' * 80}")
    print(f"# Evaluating scenario ({model_name}): {scenario_name}")
    print(f"# {SCENARIO_DESCRIPTIONS.get(scenario_name, '')}")
    print(f"{'#' * 80}\n")
    start = time.time()

    if x_test is None or y_test is None or a_test is None:
        x_test, y_test, a_test = load_drifted_test_set(scenario_name)

    data_dim = x_test.shape[1]
    print(f"number of features: {data_dim}")
    
    stream, test_metrics = _evaluate_selected_model(
        model_name=model_name,
        x_test=x_test,
        y_test=y_test,
        a_test=a_test,
        data_dim=data_dim,
        seed=seed,
    )

    os.makedirs(output_dir, exist_ok=True)
    return {
        'model': model_name,
        'scenario': scenario_name,
        'test_metrics': test_metrics,
        'timestep_results': stream,
        'elapsed_seconds': round(time.time() - start, 1),
    }


def run_scenarios(kwargs, model_name, dataset_name, output_dir=OUTPUT_DIR, scenario_filter=None):
    dataset_key = str(dataset_name).lower()
    print(f"\n{'=' * 80}")
    if dataset_key == 'adult':
        scenarios = list(SCENARIOS.keys())
        print(f" Running all {len(scenarios)} drift scenarios")
    elif dataset_key == 'folktables':
        scenarios = ['folktables']
        print(" Running single Folktables evaluation")
    elif dataset_key == 'diabetes':
        scenarios = ['diabetes']
        print(" Running single Diabetes evaluation")
    else:
        raise ValueError(
            f"Unsupported dataset for pipeline evaluation: '{dataset_name}'."
        )
    print(f" Model: {model_name}")
    print(f" Dataset: {dataset_name}")
    print(f" Output: {os.path.abspath(output_dir)}/")
    print(f"{'=' * 80}\n")

    results = []
    if dataset_key == 'adult':
        for idx, scenario_name in enumerate(scenarios, 1):
            if scenario_filter is not None and scenario_name != scenario_filter:
                continue
            print(f"\n>>> [{idx}/{len(scenarios)}] {scenario_name}")
            result = _single_scenario(
                model_name=model_name,
                scenario_name=scenario_name,
                output_dir=output_dir,
                seed=kwargs.get('seed'),
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
    elif dataset_key == 'folktables':
        x_train, x_test, y_train, y_test, a_train, a_test, _ = read_folktables(
            train_year=FLAGS.folktables_train_year,
            test_years=tuple(
                int(year.strip()) for year in str(FLAGS.folktables_test_years).split(',') if year.strip()
            ),
            state=[state.strip() for state in str(FLAGS.folktables_states).split(',') if state.strip()],
            horizon=FLAGS.folktables_horizon,
            sensitive_attribute=FLAGS.folktables_sensitive_attribute,
        )
        result = _single_scenario(
            model_name=model_name,
            scenario_name='folktables',
            output_dir=output_dir,
            x_test=x_test,
            y_test=y_test,
            a_test=a_test,
            seed=kwargs.get('seed'),
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
    else:
        x_train, x_test, y_train, y_test, a_train, a_test = read_diabetes(
            path=FLAGS.diabetes_path,
        )
        result = _single_scenario(
            model_name=model_name,
            scenario_name='diabetes',
            output_dir=output_dir,
            x_test=x_test,
            y_test=y_test,
            a_test=a_test,
            seed=kwargs.get('seed'),
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
    models_to_run = _resolve_pipeline_models(dataset_name)
    seeds = _parse_seed_list()

    print(f"\n{'=' * 80}")
    print(f" Multi-seed pipeline: {len(seeds)} runs")
    print(f" Models: {models_to_run}")
    print(f" Dataset: {dataset_name}")
    print(f" Seeds: {seeds}")
    print(f" Base output: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'=' * 80}\n")

    seed_runs = []
    base_output_dir = os.path.join(OUTPUT_DIR, f'dataset_{dataset_name}')
    for idx, seed in enumerate(seeds, 1):
        print(f"\n{'-' * 80}")
        print(f" Seed run [{idx}/{len(seeds)}]: {seed}")
        print(f"{'-' * 80}")
        kwargs = _train_kwargs(seed=seed, dataset_name=dataset_name)
        for model_name in models_to_run:
            print(f"\n>>> Model run: {model_name}")
            seed_output_dir = os.path.join(base_output_dir, f'model_{model_name}', f'seed_{seed}')
            results = run_scenarios(
                kwargs=kwargs,
                model_name=model_name,
                dataset_name=dataset_name,
                output_dir=seed_output_dir,
                scenario_filter=FLAGS.drift_scenario if FLAGS.drift_scenario else None
            )
            seed_runs.append({'seed': seed, 'model': model_name, 'results': results})

    if FLAGS.run_all_scenarios:
        metric_names = [
            'accuracy', 'dp', 'eo',
            'stream_final_accuracy', 'stream_final_dp', 'stream_final_eo',
        ]
        grouped = {}
        for run in seed_runs:
            for row in run['results']:
                model_name = run.get('model')
                grouped.setdefault(model_name, {})
                grouped[model_name].setdefault(row['scenario'], []).append(row)

        summary_rows = []
        for model_name in sorted(grouped.keys()):
            scenario_rows = []
            for scenario_name in sorted(grouped[model_name].keys()):
                scenario_rows.append({
                    'scenario': scenario_name,
                    'metrics': _aggregate_metrics(grouped[model_name][scenario_name], metric_names),
                })
            summary_rows.append({
                'model': model_name,
                'rows': scenario_rows,
            })
        summary = {'mode': 'all_scenarios', 'rows': summary_rows}
    else:
        metric_names = [
            'accuracy', 'dp', 'eo',
            'stream_final_accuracy', 'stream_final_dp', 'stream_final_eo',
        ]
        metrics_by_model = {}
        for model_name in models_to_run:
            rows = [
                row
                for run in seed_runs
                if run.get('model') == model_name
                for row in run.get('results', [])
                if isinstance(row, dict)
            ]
            metrics_by_model[model_name] = _aggregate_metrics(rows, metric_names)
        summary = {'mode': 'single', 'metrics_by_model': metrics_by_model}

    payload = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': models_to_run,
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
        for model_row in summary['rows']:
            print(f"\nModel: {model_row['model']}")
            print(f"{'Scenario':<22s} {'Acc':>28s} {'DP':>28s} {'EO':>28s}")
            print('-' * 120)
            for row in model_row['rows']:
                print(
                    f"{row['scenario']:<22s} "
                    f"{_fmt_stats(row['metrics']['accuracy']):>28s} "
                    f"{_fmt_stats(row['metrics']['dp']):>28s} "
                    f"{_fmt_stats(row['metrics']['eo']):>28s}"
                )
    else:
        for model_name in sorted(summary['metrics_by_model'].keys()):
            print(f"\nModel: {model_name}")
            for metric_name, metric_stats in summary['metrics_by_model'][model_name].items():
                print(f"{metric_name:<24s} {_fmt_stats(metric_stats)}")



if __name__ == '__main__':
    app.run(main)
