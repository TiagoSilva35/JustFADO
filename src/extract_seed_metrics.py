import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_SCENARIO_ORDER = [
    'no_drift',
    'abrupt_gender',
    'gradual_gender',
    'occupation_gender_reversal',
    'gender_relationship_decouple',
]

DEFAULT_SCENARIO_LABELS = {
    'no_drift': 'No Drift',
    'abrupt_gender': 'Abrupt',
    'gradual_gender': 'Gradual',
    'occupation_gender_reversal': 'Reversal',
    'gender_relationship_decouple': 'Decouple',
}

DEFAULT_MODEL_ORDER = ['arf', 'rfr', 'aranyani']
DEFAULT_MODEL_LABELS = {
    'arf': 'ARF',
    'rfr': 'RFR',
    'aranyani': r'\textbf{Aranyani}',
}


def _parse_kv_mapping(raw):
    mapping = {}
    if not raw:
        return mapping
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"Invalid key=value mapping entry: '{item}'")
        key, value = item.split('=', 1)
        mapping[key.strip()] = value.strip()
    return mapping


def _normalize_runs(payload):
    if isinstance(payload, dict) and isinstance(payload.get('seed_runs'), list):
        default_model = payload.get('model')
        if not default_model:
            models = payload.get('models')
            if isinstance(models, list) and len(models) == 1:
                default_model = models[0]
        normalized_runs = []
        for run in payload['seed_runs']:
            run_copy = dict(run)
            if not run_copy.get('model') and default_model:
                run_copy['model'] = default_model
            normalized_runs.append(run_copy)
        return normalized_runs
    if isinstance(payload, list):
        model_hint = _infer_model_from_payload(payload, default='unknown')
        return [{'model': model_hint, 'seed': None, 'results': payload}]
    raise ValueError('Unsupported JSON format. Expected {"seed_runs":[...]} or list.')


def _infer_model_from_payload(payload, default=None):
    if isinstance(payload, dict):
        model = payload.get('model')
        if model:
            return model
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                model = entry.get('model')
                if model:
                    return model
    return default


def _parse_seed_from_dir(seed_dir):
    name = seed_dir.name
    if name.startswith('seed_'):
        token = name[len('seed_'):]
    else:
        token = name
    try:
        return int(token)
    except ValueError:
        return token


def _infer_model_from_path(path):
    for parent in path.parents:
        if parent.name.startswith('model_'):
            return parent.name[len('model_'):]
    return None


def _load_seed_runs_from_results_file(path):
    with path.open('r') as f:
        payload = json.load(f)
    if isinstance(payload, list):
        model_hint = _infer_model_from_payload(payload, default=_infer_model_from_path(path))
        return [{
            'seed': _parse_seed_from_dir(path.parent),
            'model': model_hint or 'unknown',
            'results': payload,
        }]
    return _normalize_runs(payload)


def _find_seed_results_files(base_dir):
    results_files = set()
    for seed_dir in base_dir.rglob('seed_*'):
        if not seed_dir.is_dir():
            continue
        results_path = seed_dir / 'results.json'
        if results_path.is_file():
            results_files.add(results_path.resolve())
    return sorted(results_files)


def _resolve_input_path(path, experiments_dir):
    if path.exists():
        return path
    candidate = experiments_dir / path
    if candidate.exists():
        return candidate
    return path


def _load_seed_runs(paths, experiments_dir):
    all_runs = []
    for raw_path in paths:
        path = _resolve_input_path(raw_path, experiments_dir)
        if path.is_dir():
            results_files = _find_seed_results_files(path)
            if not results_files:
                raise ValueError(f'No seed_*/results.json files found under: {path}')
            for results_file in results_files:
                all_runs.extend(_load_seed_runs_from_results_file(results_file))
            continue
        if not path.is_file():
            raise FileNotFoundError(f'Input path not found: {path}')
        all_runs.extend(_load_seed_runs_from_results_file(path))
    return all_runs


def _aggregate(seed_runs, metrics, model_filter=None):
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen = set()

    for run in seed_runs:
        model = str(run.get('model', 'unknown')).strip().lower()
        if model_filter and model not in model_filter:
            continue
        seed = run.get('seed')
        for entry in run.get('results', []):
            scenario = entry.get('scenario')
            if not scenario:
                continue
            for metric in metrics:
                value = entry.get(metric)
                if value is None:
                    continue
                dedupe_key = (model, seed, scenario, metric)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                values[model][scenario][metric].append(float(value))
    return values


def _stats(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _sort_models(models):
    order_index = {name: i for i, name in enumerate(DEFAULT_MODEL_ORDER)}
    return sorted(models, key=lambda m: (order_index.get(m, 10_000), m))


def _sort_scenarios(scenarios):
    order_index = {name: i for i, name in enumerate(DEFAULT_SCENARIO_ORDER)}
    return sorted(scenarios, key=lambda s: (order_index.get(s, 10_000), s))


def _fmt_metric(mean, std):
    return f'{mean:.4f} \\pm {std:.4f}'


def _latex_row(model, scenarios, metric, aggregated, model_labels):
    model_label = model_labels.get(model, model.upper())
    cells = []
    for scenario in scenarios:
        values = aggregated[model][scenario][metric]
        if not values:
            cells.append('--')
            continue
        mean, std = _stats(values)
        cells.append(f'${_fmt_metric(mean, std)}$')
    return f'{model_label} & ' + ' & '.join(cells) + r' \\'


def _print_latex_tables(aggregated, metrics, scenarios, scenario_labels, model_labels):
    models = _sort_models(aggregated.keys())
    scenario_headers = [scenario_labels.get(s, s.replace('_', ' ').title()) for s in scenarios]
    n_cols = len(scenarios)

    for metric in metrics:
        metric_label_map = {'dp': 'DP', 'eo': 'EO', 'accuracy': 'Accuracy'}
        metric_label = metric_label_map.get(metric, metric.replace('_', ' ').title())
        arrow = r'$\uparrow$' if 'accuracy' in metric else r'$\downarrow$'
        print(f'% {metric_label}')
        print(rf'\begin{{tabular}}{{l{"c" * n_cols}}}')
        print(r'\toprule')
        print(r'\textbf{Model} & ' + ' & '.join([rf'\textbf{{{h} ({metric_label} {arrow})}}' for h in scenario_headers]) + r' \\')
        print(r'\midrule')
        for model in models:
            print(_latex_row(model, scenarios, metric, aggregated, model_labels))
        print(r'\bottomrule')
        print(r'\end{tabular}')
        print()


def _print_summary(aggregated, metrics, scenarios):
    models = _sort_models(aggregated.keys())
    for model in models:
        print(f'[{model}]')
        for scenario in scenarios:
            line_parts = []
            for metric in metrics:
                values = aggregated[model][scenario][metric]
                if not values:
                    continue
                mean, std = _stats(values)
                line_parts.append(f'{metric}={mean:.4f}±{std:.4f}')
            if line_parts:
                print(f'  {scenario}: ' + ', '.join(line_parts))
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate per-seed pipeline results and print paper-ready tables.',
    )
    parser.add_argument(
        '--inputs',
        nargs='+',
        type=Path,
        default=[Path('files/experiments/dataset_folktables/seed_pipeline_results.json')],
        help='One or more seed_pipeline_results.json files or experiment directories.',
    )
    parser.add_argument(
        '--experiments-dir',
        type=Path,
        default=Path('files/experiments'),
        help='Base directory for experiment outputs (used to resolve relative --inputs).',
    )
    parser.add_argument(
        '--metrics',
        default='accuracy,dp',
        help='Comma-separated metrics to aggregate.',
    )
    parser.add_argument(
        '--scenarios',
        default='',
        help='Optional comma-separated scenario order. Defaults to known order + discovered scenarios.',
    )
    parser.add_argument(
        '--models',
        default='',
        help='Optional comma-separated model filter (e.g., arf,rfr,aranyani).',
    )
    parser.add_argument(
        '--scenario-labels',
        default='',
        help='Optional key=value pairs for scenario labels (comma-separated).',
    )
    parser.add_argument(
        '--model-labels',
        default='',
        help='Optional key=value pairs for model labels (comma-separated).',
    )
    parser.add_argument(
        '--format',
        choices=['latex', 'summary'],
        default='latex',
        help='Output format.',
    )
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    if not metrics:
        raise ValueError('At least one metric must be provided via --metrics.')

    model_filter = {m.strip().lower() for m in args.models.split(',') if m.strip()}
    if not model_filter:
        model_filter = None

    seed_runs = _load_seed_runs(args.inputs, args.experiments_dir)
    aggregated = _aggregate(seed_runs, metrics=metrics, model_filter=model_filter)
    if not aggregated:
        raise ValueError('No matching runs found. Check --inputs/--models.')

    discovered_scenarios = {
        scenario
        for model_data in aggregated.values()
        for scenario in model_data.keys()
    }
    if args.scenarios.strip():
        scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]
    else:
        scenarios = _sort_scenarios(discovered_scenarios)

    scenario_labels = dict(DEFAULT_SCENARIO_LABELS)
    scenario_labels.update(_parse_kv_mapping(args.scenario_labels))
    model_labels = dict(DEFAULT_MODEL_LABELS)
    model_labels.update(_parse_kv_mapping(args.model_labels))

    if args.format == 'latex':
        _print_latex_tables(
            aggregated=aggregated,
            metrics=metrics,
            scenarios=scenarios,
            scenario_labels=scenario_labels,
            model_labels=model_labels,
        )
    else:
        _print_summary(aggregated=aggregated, metrics=metrics, scenarios=scenarios)


if __name__ == '__main__':
    main()
