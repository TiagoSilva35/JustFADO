"""Per-seed paired significance tests for the COMPAS prequential sweep.

Reads the seed-pipeline JSON produced by ``src/main.py`` (one entry per
(seed, model, scenario) cell), computes per-seed averages across the scenarios
that exist for each model, and runs the paired Wilcoxon signed-rank test
(the Section IV.H protocol) plus a Welch's two-sample t-test fall-back on the
same per-seed values. p-values are corrected within each metric column with
Holm--Bonferroni across the set of baseline comparisons.

The Welch result is what the paper currently quotes from summary statistics;
the Wilcoxon result is what the protocol formally specifies. They are reported
side-by-side so the reviewer can read both. The Welch column reproduces the
hand calculation that backs Table~\\ref{tab:compas} in DOCS/paper.tex.

Usage
-----
    python -m src.significance_tests \\
        --inputs files/experiments/dataset_compas \\
        --reference aranyani \\
        --baselines aranyani_base,arf,rfr \\
        --metrics accuracy,dp,eo

The defaults match the COMPAS protocol exactly, so the no-flags invocation is
the canonical reproduction of the paper's significance numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


# Mirrors src/extract_seed_metrics.py so the two utilities agree on what
# counts as a "model" / "scenario" by default. Keep in sync if either list
# changes.
DEFAULT_MODELS_REFERENCE = 'aranyani'
DEFAULT_MODELS_BASELINES = ('aranyani_base', 'arf', 'rfr')
DEFAULT_METRICS = ('accuracy', 'dp', 'eo')


def _load_seed_runs(inputs_dir: Path) -> List[dict]:
    """Discover per-seed results across all ``model_*/seed_*/results.json``.

    Each ``results.json`` holds a list of scenario records (one per scenario
    actually run for that seed and model). We flatten everything into the
    ``[{seed, model, results:[scenario records]}]`` shape that the rest of
    the script expects, so the per-seed pipeline JSON and the per-seed
    result files become drop-in equivalents from the caller's point of
    view.
    """
    if not inputs_dir.exists():
        raise FileNotFoundError(f'Inputs directory does not exist: {inputs_dir}')

    aggregated: Dict[Tuple[int, str], List[dict]] = {}
    for model_dir in sorted(inputs_dir.glob('model_*')):
        model_name = model_dir.name[len('model_'):]
        for seed_dir in sorted(model_dir.glob('seed_*')):
            try:
                seed = int(seed_dir.name[len('seed_'):])
            except ValueError:
                continue
            results_path = seed_dir / 'results.json'
            if not results_path.exists():
                continue
            with results_path.open() as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                # Tolerate older single-dict files.
                payload = [payload]
            aggregated[(seed, model_name)] = payload

    if not aggregated:
        raise FileNotFoundError(
            f'No model_*/seed_*/results.json files found under {inputs_dir}.'
        )

    runs: List[dict] = []
    for (seed, model_name), results in aggregated.items():
        runs.append({'seed': seed, 'model': model_name, 'results': results})
    return runs


def _common_scenarios(runs: Sequence[dict], models: Sequence[str]) -> List[str]:
    """Scenarios for which every model in ``models`` has at least one cell.

    The paired tests in this script require apples-to-apples per-seed
    averages, so we restrict to the intersection of scenarios across the
    models being compared. Scenarios that exist for some models but not
    others would inflate the baseline averages with extra cells the
    reference model never ran.
    """
    per_model: Dict[str, set] = {model: set() for model in models}
    for run in runs:
        model = str(run['model'])
        if model not in per_model:
            continue
        for blob in run.get('results', []) or []:
            scenario = blob.get('scenario')
            if scenario is not None:
                per_model[model].add(str(scenario))
    if not all(per_model[m] for m in models):
        missing = [m for m in models if not per_model[m]]
        raise ValueError(f'No scenarios found for model(s): {missing}')
    intersection = set.intersection(*[per_model[m] for m in models])
    return sorted(intersection)


def _per_seed_scenario_averages(
    runs: Sequence[dict],
    metrics: Sequence[str],
    scenarios: Sequence[str],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Return ``out[model][metric][seed] -> mean-over-scenarios value``.

    Only the scenarios in ``scenarios`` are aggregated. Cells with a missing
    metric value are skipped, and a seed is dropped from the average if it
    has zero contributing cells (rather than silently producing NaN).
    """
    out: Dict[str, Dict[str, Dict[int, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    scenario_set = set(scenarios)
    for run in runs:
        seed = int(run['seed'])
        model = str(run['model'])
        for scenario_blob in run.get('results', []) or []:
            if str(scenario_blob.get('scenario')) not in scenario_set:
                continue
            for metric in metrics:
                value = scenario_blob.get(metric)
                if value is None:
                    continue
                out[model][metric][seed].append(float(value))
    averaged: Dict[str, Dict[str, Dict[int, float]]] = {}
    for model, metric_blob in out.items():
        averaged[model] = {}
        for metric, seed_blob in metric_blob.items():
            averaged[model][metric] = {
                seed: float(np.mean(vals))
                for seed, vals in seed_blob.items()
                if vals
            }
    return averaged


def _paired_arrays(
    ref_by_seed: Dict[int, float],
    base_by_seed: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return ref, base arrays restricted to seeds present in both, ordered."""
    shared = sorted(set(ref_by_seed) & set(base_by_seed))
    if not shared:
        return np.array([]), np.array([]), []
    ref = np.array([ref_by_seed[s] for s in shared], dtype=float)
    base = np.array([base_by_seed[s] for s in shared], dtype=float)
    return ref, base, shared


def _holm(raw_pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down on a sequence of raw p-values.

    Returns adjusted p-values in the same order as the input. Adjustment uses
    the monotone correction: each rung is clamped up to the previous rung so
    the sequence is non-decreasing along the original ordering of p-values.
    """
    pvals = list(raw_pvals)
    m = len(pvals)
    if m == 0:
        return []
    ranked = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(ranked):
        factor = m - rank
        candidate = min(pvals[idx] * factor, 1.0)
        running = max(running, candidate)
        adj[idx] = running
    return adj


def _star(p: float) -> str:
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _format_p(p: float) -> str:
    if p < 1e-4:
        return f'{p:.2e}'
    if p < 0.01:
        return f'{p:.4f}'
    return f'{p:.3f}'


def _run_metric(
    averaged: Dict[str, Dict[str, Dict[int, float]]],
    metric: str,
    reference: str,
    baselines: Sequence[str],
) -> List[dict]:
    """Run paired Wilcoxon + Welch for every (reference, baseline) pair."""
    ref_by_seed = averaged.get(reference, {}).get(metric, {})
    if not ref_by_seed:
        raise ValueError(
            f'No per-seed values for reference model "{reference}" '
            f'on metric "{metric}".'
        )
    raw_wilcoxon: List[float] = []
    raw_welch: List[float] = []
    rows: List[dict] = []
    for base in baselines:
        base_by_seed = averaged.get(base, {}).get(metric, {})
        ref_arr, base_arr, shared = _paired_arrays(ref_by_seed, base_by_seed)
        n = len(shared)
        diff_mean = float(np.mean(ref_arr - base_arr)) if n else float('nan')
        if n >= 2 and np.any(ref_arr != base_arr):
            # zero_method='wilcox': drop zero-differences (the scipy default
            # for the modern Wilcoxon implementation). Two-sided test.
            try:
                w_stat, w_p = stats.wilcoxon(
                    ref_arr, base_arr, zero_method='wilcox', alternative='two-sided'
                )
            except ValueError:
                # Can happen if all differences are zero after filtering.
                w_stat, w_p = float('nan'), 1.0
            t_stat, t_p = stats.ttest_ind(
                ref_arr, base_arr, equal_var=False
            )
        else:
            w_stat, w_p = float('nan'), float('nan')
            t_stat, t_p = float('nan'), float('nan')
        raw_wilcoxon.append(w_p)
        raw_welch.append(t_p)
        rows.append({
            'baseline': base,
            'n_paired': n,
            'mean_diff': diff_mean,
            'wilcoxon_W': float(w_stat) if not math.isnan(w_stat) else None,
            'wilcoxon_p_raw': w_p,
            'welch_t': float(t_stat) if not math.isnan(t_stat) else None,
            'welch_p_raw': t_p,
        })
    holm_wilcoxon = _holm(raw_wilcoxon)
    holm_welch = _holm(raw_welch)
    for row, w_adj, t_adj in zip(rows, holm_wilcoxon, holm_welch):
        row['wilcoxon_p_holm'] = w_adj
        row['welch_p_holm'] = t_adj
        row['wilcoxon_stars'] = _star(w_adj)
        row['welch_stars'] = _star(t_adj)
    return rows


def _print_metric_block(metric: str, rows: Sequence[dict], reference: str) -> None:
    print(f'\n=== {metric.upper()}: {reference} vs each baseline ===')
    header = (
        f'  {"baseline":<14} {"n":>3} {"Δmean":>9}   '
        f'{"Wilcoxon W":>11} {"raw p":>10} {"Holm p":>10} {"":>4}   '
        f'{"Welch t":>9} {"raw p":>10} {"Holm p":>10} {"":>4}'
    )
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for row in rows:
        w_stat_str = (
            f'{row["wilcoxon_W"]:.1f}'
            if row['wilcoxon_W'] is not None
            else '   --   '
        )
        t_stat_str = (
            f'{row["welch_t"]:+.3f}'
            if row['welch_t'] is not None
            else '   --   '
        )
        print(
            f'  {row["baseline"]:<14} {row["n_paired"]:>3} '
            f'{row["mean_diff"]:+9.4f}   '
            f'{w_stat_str:>11} {_format_p(row["wilcoxon_p_raw"]):>10} '
            f'{_format_p(row["wilcoxon_p_holm"]):>10} {row["wilcoxon_stars"]:>4}   '
            f'{t_stat_str:>9} {_format_p(row["welch_p_raw"]):>10} '
            f'{_format_p(row["welch_p_holm"]):>10} {row["welch_stars"]:>4}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Paired Wilcoxon + Welch significance tests for the COMPAS '
            'prequential sweep, with Holm-Bonferroni correction.'
        )
    )
    parser.add_argument(
        '--inputs',
        type=Path,
        default=Path('files/experiments/dataset_compas'),
        help='Directory containing seed_pipeline_results.json.',
    )
    parser.add_argument(
        '--reference',
        default=DEFAULT_MODELS_REFERENCE,
        help='Model name used as the reference in every paired test '
             '(default: aranyani, i.e. FADO).',
    )
    parser.add_argument(
        '--baselines',
        default=','.join(DEFAULT_MODELS_BASELINES),
        help='Comma-separated baseline model names.',
    )
    parser.add_argument(
        '--metrics',
        default=','.join(DEFAULT_METRICS),
        help='Comma-separated metric keys (any subset of '
             'accuracy, dp, eo).',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Emit results as JSON instead of the human-readable table.',
    )
    args = parser.parse_args()

    baselines = [b.strip() for b in args.baselines.split(',') if b.strip()]
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]

    runs = _load_seed_runs(args.inputs)
    scenarios = _common_scenarios(runs, [args.reference] + baselines)
    averaged = _per_seed_scenario_averages(runs, metrics, scenarios)

    out: Dict[str, List[dict]] = {}
    for metric in metrics:
        out[metric] = _run_metric(averaged, metric, args.reference, baselines)

    if args.json:
        print(json.dumps({
            'reference': args.reference,
            'baselines': baselines,
            'metrics': metrics,
            'results': out,
        }, indent=2))
        return

    print(
        f'Paired Wilcoxon + Welch tests on per-seed scenario-averaged values\n'
        f'  Reference: {args.reference}\n'
        f'  Baselines: {", ".join(baselines)}\n'
        f'  Metrics:   {", ".join(metrics)}\n'
        f'  Scenarios: {", ".join(scenarios)} (intersection across all models)\n'
        f'  Holm-Bonferroni correction across the {len(baselines)} baseline '
        f'comparisons within each metric column.'
    )
    for metric in metrics:
        _print_metric_block(metric, out[metric], args.reference)
    print()


if __name__ == '__main__':
    main()
