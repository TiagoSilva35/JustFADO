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

DEFAULT_MODEL_ORDER = ['arf', 'rfr', 'fermi', 'aranyani']
DEFAULT_MODEL_LABELS = {
    'arf': 'ARF',
    'rfr': 'RFR',
    'fermi': 'FERMI',
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
        return [{'model': 'unknown', 'seed': None, 'results': payload}]
    raise ValueError('Unsupported JSON format. Expected {"seed_runs":[...]} or list.')


def _load_seed_runs(paths):
    all_runs = []
    for path in paths:
        with path.open('r') as f:
            payload = json.load(f)
        all_runs.extend(_normalize_runs(payload))
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
        default=[Path('files/experiments/dataset_adult/seed_pipeline_results.json')],
        help='One or more seed_pipeline_results.json files.',
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
        help='Optional comma-separated model filter (e.g., arf,rfr,fermi,aranyani).',
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

    seed_runs = _load_seed_runs(args.inputs)
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
