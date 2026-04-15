import json
import numbers
import os
import random
import time
import traceback

import numpy as np
from absl import app, flags

from src.drift.scenarios import SCENARIO_DESCRIPTIONS, SCENARIOS, SPLITS, PHASE_LABELS
from src.helpers.data import load_drifted_test_set
from src.helpers.plots import plot_metrics_over_timesteps
from src.helpers.utils import get_test_performance
from src.models.arf.arf import evaluate_arf_over_timesteps
from src.models.forest.evaluator import evaluate_over_timesteps
from src.models.rfr.evaluator import evaluate_rfr_over_timesteps
from src.models.forest.train import train, FLAGS

OUTPUT_DIR = 'files/experiments'

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
flags.DEFINE_integer(
    'seed_runs',
    5,
    'Number of random seeds to auto-generate when --seeds is empty.',
)
flags.DEFINE_integer(
    'random_seed_min',
    1,
    'Lower bound (inclusive) for auto-generated random seeds.',
)
flags.DEFINE_integer(
    'random_seed_max',
    2_147_483_647,
    'Upper bound (inclusive) for auto-generated random seeds.',
)
flags.DEFINE_enum(
    'rfr_backbone',
    'netregression',
    ['netregression', 'neuralgbdt'],
    'RFR backbone network to use.',
)
flags.DEFINE_integer(
    'rfr_hidden_dim',
    50,
    'Hidden dimension for RFR backbones that use MLP blocks.',
)
flags.DEFINE_integer(
    'rfr_n_ensemble',
    4,
    'Number of ensemble learners for RFR NeuralGBDT backbone.',
)
flags.DEFINE_enum(
    'rfr_approach',
    'rfr',
    ['baseline', 'rfr', 'dnn', 'dnn_adv', 'dnn_fcr'],
    'RFR-family objective: baseline, rfr, dnn, dnn_adv, or dnn_fcr.',
)
flags.DEFINE_float(
    'rfr_penalty_coefficient',
    1.0,
    'Fairness penalty coefficient used by RFR-family objectives.',
)
flags.DEFINE_float(
    'rfr_learning_rate',
    1e-3,
    'Learning rate used by the RFR evaluator optimizer.',
)
flags.DEFINE_float(
    'rfr_rho',
    1e-4,
    'Rho used by the RFR optimizer in baseline/rfr objectives.',
)
flags.DEFINE_integer(
    'rfr_train_batch_size',
    64,
    'Replay mini-batch size used by online RFR-family updates.',
)
flags.DEFINE_integer(
    'rfr_buffer_size',
    512,
    'Replay buffer size for online RFR-family updates.',
)
flags.DEFINE_float(
    'rfr_fcr_threshold',
    0.8,
    'Confidence threshold used by dnn_fcr objective.',
)
flags.DEFINE_integer(
    'rfr_adv_hidden_dim',
    32,
    'Hidden size for the adversary network used by dnn_adv.',
)


def _dataset_name():
    override = str(FLAGS.pipeline_dataset).strip()
    return override if override else FLAGS.dataset


def _train_kwargs(seed, dataset_name):
    return {
        'dataset': dataset_name,
        'max_iter': FLAGS.max_iter,
        'compute_fairness': FLAGS.compute_fairness and FLAGS.lambda_const > 0,
        'compute_mode': FLAGS.compute_mode,
        'constraint_type': FLAGS.constraint_type,
        'gradient_type': FLAGS.gradient_type,
        'encoder_model': FLAGS.encoder_model,
        'offline_loss_type': FLAGS.offline_loss_type,
        'local_run': True,
        'save_model': FLAGS.save_model,
        'load_model': FLAGS.load_model,
        'model_path': FLAGS.model_path,
        'prequential': FLAGS.prequential,
        'folktables_sensitive_attribute': FLAGS.folktables_sensitive_attribute,
        'folktables_states': FLAGS.folktables_states,
        'folktables_train_year': FLAGS.folktables_train_year,
        'folktables_test_years': FLAGS.folktables_test_years,
        'folktables_horizon': FLAGS.folktables_horizon,
        'seed': int(seed),
    }


def _parse_seed_list():
    provided = str(FLAGS.seeds).strip()
    if provided:
        seeds = []
        seen = set()
        for token in provided.split(','):
            token = token.strip()
            if not token:
                continue
            seed = int(token) 
            if seed in seen:
                continue
            seen.add(seed)
            seeds.append(seed)
        if not seeds:
            raise ValueError('No valid seeds found. Provide --seeds like "11,22,33".')
        return seeds

    runs = int(FLAGS.seed_runs)
    low = int(FLAGS.random_seed_min)
    high = int(FLAGS.random_seed_max)
    if runs < 1:
        raise ValueError('--seed_runs must be >= 1 when --seeds is empty.')
    if low > high:
        raise ValueError('--random_seed_min must be <= --random_seed_max.')
    if runs > (high - low + 1):
        raise ValueError('Requested more unique random seeds than the configured range allows.')

    rng = random.SystemRandom()
    seeds = []
    seen = set()
    while len(seeds) < runs:
        seed = rng.randint(low, high)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _as_float(value, last=False):
    if value is None:
        return None
    if last and isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[-1]
    if isinstance(value, numbers.Real):
        return float(value)
    return None


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


def _evaluate_selected_model(model_name, x_test, y_test, a_test, model=None, data_dim=None):
    if model_name == 'aranyani':
        stream = evaluate_over_timesteps(model, x_test, y_test, a_test, data_dim=data_dim)
        test_metrics = get_test_performance(model, x_test, y_test, a_test, data_dim=data_dim)
        return stream, test_metrics

    if model_name == 'arf':
        stream = evaluate_arf_over_timesteps(x_test, y_test, a_test)
        test_metrics = {
            'accuracy': _as_float(stream.get('accuracy'), last=True),
            'dp': _as_float(stream.get('dp'), last=True),
            'eo': _as_float(stream.get('eo'), last=True),
            'f1': None,
        }
        return stream, test_metrics

    if model_name == 'rfr':
        stream = evaluate_rfr_over_timesteps(
            x_test,
            y_test,
            a_test,
            approach=FLAGS.rfr_approach,
            backbone=FLAGS.rfr_backbone,
            hidden_dim=FLAGS.rfr_hidden_dim,
            n_ensemble=FLAGS.rfr_n_ensemble,
            learning_rate=FLAGS.rfr_learning_rate,
            rho=FLAGS.rfr_rho,
            penalty_coefficient=FLAGS.rfr_penalty_coefficient,
            fcr_threshold=FLAGS.rfr_fcr_threshold,
            train_batch_size=FLAGS.rfr_train_batch_size,
            buffer_size=FLAGS.rfr_buffer_size,
            adv_hidden_dim=FLAGS.rfr_adv_hidden_dim,
        )
        test_metrics = {
            'accuracy': _as_float(stream.get('accuracy'), last=True),
            'dp': _as_float(stream.get('dp'), last=True),
            'eo': _as_float(stream.get('eo'), last=True),
            'f1': None,
        }
        return stream, test_metrics

    raise ValueError(f'Unsupported model: {model_name}')


def _run_scenario(model_name, scenario_name, output_dir, model=None, data_dim=None):
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
    )

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'timesteps_{model_name}_{scenario_name}.png')
    plot_metrics_over_timesteps(
        stream,
        save_path=plot_path,
        stage_splits=SPLITS,
        stage_labels=PHASE_LABELS,
    )

    return {
        'model': model_name,
        'scenario': scenario_name,
        'test_metrics': test_metrics,
        'timestep_results': stream,
        'elapsed_seconds': round(time.time() - start, 1),
    }


def run_all_scenarios(kwargs, model_name, dataset_name, output_dir=OUTPUT_DIR):
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
    if model_name == 'aranyani':
        train_start = time.time()
        print('>>> Training model on clean data (no drift)...')
        _, _, _, _, _, trained_model, data_dim = train(
            drift=False,
            drift_scenario=None,
            **kwargs,
        )
        print(f">>> Training completed in {round(time.time() - train_start, 1)}s")
        if trained_model is None:
            raise RuntimeError('Training returned no model - cannot evaluate scenarios.')

    results = []
    for idx, scenario_name in enumerate(scenarios, 1):
        print(f"\n>>> [{idx}/{len(scenarios)}] {scenario_name}")
        try:
            result = _run_scenario(
                model_name=model_name,
                scenario_name=scenario_name,
                output_dir=output_dir,
                model=trained_model,
                data_dim=data_dim,
            )
            results.append(result)
            tm = result['test_metrics'] or {}
            print(
                f"    Done in {result['elapsed_seconds']}s | "
                f"Acc={_as_float(tm.get('accuracy'))} "
                f"DP={_as_float(tm.get('dp'))} "
                f"EO={_as_float(tm.get('eo'))}"
            )
        except Exception as exc:
            print(f"    FAILED: {exc}")
            traceback.print_exc()
            results.append({
                'model': model_name,
                'scenario': scenario_name,
                'test_metrics': None,
                'timestep_results': None,
                'elapsed_seconds': None,
                'error': str(exc),
            })

    rows = []
    for result in results:
        tm = result.get('test_metrics') or {}
        ts = result.get('timestep_results') or {}
        rows.append({
            'model': result.get('model'),
            'scenario': result.get('scenario'),
            'accuracy': _as_float(tm.get('accuracy')),
            'dp': _as_float(tm.get('dp')),
            'eo': _as_float(tm.get('eo')),
            'f1': _as_float(tm.get('f1')),
            'stream_final_accuracy': _as_float(ts.get('accuracy'), last=True),
            'stream_final_dp': _as_float(ts.get('dp'), last=True),
            'stream_final_eo': _as_float(ts.get('eo'), last=True),
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
        acc = 'n/a' if row['accuracy'] is None else f"{row['accuracy']:.4f}"
        dp = 'n/a' if row['dp'] is None else f"{row['dp']:.4f}"
        eo = 'n/a' if row['eo'] is None else f"{row['eo']:.4f}"
        print(f"{str(row['scenario']):<22s} {acc:>10s} {dp:>10s} {eo:>10s}")

    return rows


def run_single(kwargs, model_name, dataset_name, output_dir=OUTPUT_DIR):
    if model_name in ('arf', 'rfr'):
        if str(dataset_name).lower() != 'adult':
            raise ValueError(
                f"{model_name.upper()} drift evaluation currently supports dataset='adult' only; got '{dataset_name}'."
            )
        if not FLAGS.drift_scenario:
            raise ValueError(f'{model_name.upper()} single-run mode requires --drift_scenario.')
        return _run_scenario(
            model_name=model_name,
            scenario_name=FLAGS.drift_scenario,
            output_dir=output_dir,
        )

    drift_on = FLAGS.drift or (FLAGS.drift_scenario is not None)
    _, _, _, stream, test_metrics, _, _ = train(
        drift=drift_on,
        drift_scenario=FLAGS.drift_scenario,
        **kwargs,
    )

    if stream is not None:
        os.makedirs(output_dir, exist_ok=True)
        scenario_name = FLAGS.drift_scenario or 'single'
        plot_path = os.path.join(output_dir, f'timesteps_{model_name}_{scenario_name}.png')
        plot_metrics_over_timesteps(
            stream,
            save_path=plot_path,
            stage_splits=SPLITS,
            stage_labels=PHASE_LABELS,
        )

    return {
        'model': model_name,
        'scenario': FLAGS.drift_scenario,
        'test_metrics': test_metrics,
        'timestep_results': stream,
        'elapsed_seconds': None,
    }


def _aggregate_metrics(rows, metric_names):
    aggregated = {}
    for metric_name in metric_names:
        values = []
        for row in rows:
            value = row.get(metric_name)
            if value is not None:
                values.append(float(value))
        aggregated[metric_name] = _stats(values)
    return aggregated


def run_seed_pipeline():
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
    for idx, seed in enumerate(seeds, 1):
        print(f"\n{'-' * 80}")
        print(f" Seed run [{idx}/{len(seeds)}]: {seed}")
        print(f"{'-' * 80}")
        kwargs = _train_kwargs(seed=seed, dataset_name=dataset_name)
        seed_output_dir = os.path.join(OUTPUT_DIR, f'seed_{seed}')

        if FLAGS.run_all_scenarios:
            results = run_all_scenarios(
                kwargs=kwargs,
                model_name=model_name,
                dataset_name=dataset_name,
                output_dir=seed_output_dir,
            )
            seed_runs.append({'seed': seed, 'results': results})
        else:
            result = run_single(
                kwargs=kwargs,
                model_name=model_name,
                dataset_name=dataset_name,
                output_dir=seed_output_dir,
            )
            tm = result.get('test_metrics') or {}
            ts = result.get('timestep_results') or {}
            seed_runs.append({
                'seed': seed,
                'results': {
                    'accuracy': _as_float(tm.get('accuracy')),
                    'dp': _as_float(tm.get('dp')),
                    'eo': _as_float(tm.get('eo')),
                    'f1': _as_float(tm.get('f1')),
                    'stream_final_accuracy': _as_float(ts.get('accuracy'), last=True),
                    'stream_final_dp': _as_float(ts.get('dp'), last=True),
                    'stream_final_eo': _as_float(ts.get('eo'), last=True),
                },
            })

    if FLAGS.run_all_scenarios:
        metric_names = [
            'accuracy', 'dp', 'eo', 'f1',
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
            'accuracy', 'dp', 'eo', 'f1',
            'stream_final_accuracy', 'stream_final_dp', 'stream_final_eo',
        ]
        rows = [run['results'] for run in seed_runs]
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'seed_pipeline_results.json')
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


def main(_):
    model_name = str(FLAGS.pipeline_model).lower()
    dataset_name = _dataset_name()

    if FLAGS.run_seed_pipeline:
        run_seed_pipeline()
        return

    kwargs = _train_kwargs(seed=FLAGS.seed, dataset_name=dataset_name)

    if FLAGS.run_all_scenarios:
        run_all_scenarios(
            kwargs=kwargs,
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=OUTPUT_DIR,
        )
    else:
        run_single(
            kwargs=kwargs,
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=OUTPUT_DIR,
        )


if __name__ == '__main__':
    app.run(main)
