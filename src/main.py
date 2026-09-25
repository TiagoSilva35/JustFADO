import atexit
import hashlib
import json
import math
import numbers
import os
import random
import shutil
import tempfile
import time

import numpy as np
from absl import app, flags
from sklearn.model_selection import train_test_split

import src.models.forest.aranyani as aranyani
import src.models.forest.forest as forest
from src.drift.compas_scenarios import (
    COMPAS_SCENARIO_DESCRIPTIONS,
    COMPAS_SCENARIOS,
)
from src.drift.scenarios import SCENARIO_DESCRIPTIONS, SCENARIOS
from src.drift import compas_scenarios as _compas_scenarios
from src.drift import scenarios as _adult_scenarios

# Unified description map for printing per-scenario headers regardless of dataset.
ALL_SCENARIO_DESCRIPTIONS = {
    **SCENARIO_DESCRIPTIONS,
    **COMPAS_SCENARIO_DESCRIPTIONS,
}
from src.helpers.constants import (
    DEFAULT_RANDOM_SEED_MAX,
    DEFAULT_RANDOM_SEED_MIN,
    DEFAULT_SEED_RUNS,
    OUTPUT_DIR,
    RFR_CONFIG,
)
from src.helpers.data import (
    load_drifted_test_set,
    read_adult,
    read_compas,
    read_compas_train_test,
    read_folktables,
)
from src.models.arf.arf import evaluate_arf_over_timesteps
from src.models.forest.baseline_evaluator import evaluate_aranyani_baseline_over_timesteps
from src.models.forest.evaluator import evaluate_over_timesteps
from src.models.forest.controller import ControllerConfig, PRESETS as CONTROLLER_PRESETS
from src.models.forest.train import FLAGS, _set_global_seed
from src.models.rfr.evaluator import evaluate_rfr_over_timesteps

try:
    import wandb
except ImportError:
    wandb = None

FOLKTABLES_PIPELINE_TRAIN_YEAR = 2015
FOLKTABLES_PIPELINE_TEST_YEARS = (2017, 2018)

flags.DEFINE_string(
    'pipeline_model',
    '',
    'Optional model to run in main pipeline: aranyani, aranyani_base, arf, or rfr. Empty runs all supported models for the dataset. '
    '"aranyani" is the full FADO framework (Aranyani + drift detection + reaction); "aranyani_base" is the pure Aranyani baseline without the controller.',
)
flags.DEFINE_string(
    'pipeline_dataset',
    '',
    'Optional dataset override for pipeline runs. Empty keeps --dataset.',
)
flags.DEFINE_float(
    'folktables_subsample_fraction',
    1.0,
    'If <1.0, uniformly subsample the Folktables train and test splits to '
    'this fraction, preserving their original proportions. The subsample is '
    'drawn deterministically from the per-seed RNG so all models within a '
    'seed see the same rows (paired tests remain valid) but seeds differ. '
    'Default 1.0 = use full splits. Set 0.10 to slim a ~187k+187k cell to '
    '~18.7k+18.7k for a ~10x wall-clock reduction.',
)
flags.DEFINE_string(
    'compas_path',
    'data/compas/*',
    'Path, directory, or glob for COMPAS CSV files. Defaults to data/compas/*.',
)
flags.DEFINE_bool(
    'wandb_log',
    False,
    'Enable Weights & Biases logging for pipeline runs (useful for sweeps/ablations).',
)
flags.DEFINE_bool(
    'intersectional',
    False,
    'Enable multi-protected-attribute (intersectional) mode. When True the '
    'COMPAS loader returns the composite race x sex code and the Folktables '
    'loader returns the composite SEX x RAC1P-binarised code, plus per-attribute '
    'marginal arrays for diagnostic DP/EO reporting. Default off for back-compat.',
)
flags.DEFINE_string(
    'wandb_project',
    'adult-drift-ablations',
    'W&B project name used when --wandb_log=True.',
)
flags.DEFINE_string(
    'wandb_entity',
    '',
    'Optional W&B entity/team used when --wandb_log=True.',
)


def _dataset_name():
    override = str(FLAGS.pipeline_dataset).strip()
    return override if override else FLAGS.dataset



_COMPAS_FADO_OVERRIDES = {
    # A1: warning must be the MORE sensitive detector (larger delta in river),
    # so these two were swapped relative to the previous values.
    'adwin_delta_warn': 0.2,
    'adwin_delta_confirm': 1e-3,
    'fairness_window': 250,
    'min_samples_per_stream': 20,
    'lr_decay_steps': 600,
    'lambda_const': 1.0,
}

_FOLKTABLES_FADO_OVERRIDES = {
    'drift_lr_spike_mult': 3.0,
    'temperature_on_drift': 0.5,
    'lr_decay_steps': 1000,
}


def _effective_fairness_window():
    """Fairness window shared by every arm, dataset defaults included."""
    return int(_build_aranyani_static_params()['fairness_window'])


def _effective_lambda():
    return float(_build_aranyani_static_params()['lambda_const'])


def _effective_accuracy_window():
    """Rolling window for the reported prequential accuracy, shared by every arm.

    B1/B2: accuracy used to be reported as a cumulative curve while DP/EO were
    windowed. Every evaluator now reports a rolling accuracy, and they all use
    this window so the arms stay comparable on one time scale. 0 means "follow
    the fairness window".
    """
    window = int(FLAGS.drift_accuracy_window)
    return window if window > 0 else _effective_fairness_window()


# Flag that sets each static param. The dataset defaults above apply only when
# that flag was not passed explicitly -- otherwise a sweep over, say,
# --lambda_const on COMPAS would silently run every cell at the override.
_STATIC_PARAM_FLAGS = {
    'adwin_delta_warn': ('drift_adwin_delta_warn', 'drift_adwin_delta_warn_multiplier'),
    'adwin_delta_confirm': ('drift_adwin_delta_confirm',),
    'fairness_window': ('drift_fairness_window',),
    'min_samples_per_stream': ('drift_min_samples_per_stream',),
    'lr_decay_steps': ('drift_lr_decay_steps',),
    'lambda_const': ('lambda_const',),
    'drift_lr_spike_mult': ('drift_lr_spike_mult',),
    'temperature_on_drift': ('drift_temperature_on_drift',),
}


def _explicitly_set(param):
    return any(FLAGS[name].present for name in _STATIC_PARAM_FLAGS[param])


def _build_aranyani_static_params():
    confirm_delta = float(FLAGS.drift_adwin_delta_confirm)
    warn_multiplier = float(FLAGS.drift_adwin_delta_warn_multiplier)
    warn_delta = (confirm_delta * warn_multiplier if warn_multiplier > 0
                  else float(FLAGS.drift_adwin_delta_warn))
    params = {
        'adwin_delta_warn': warn_delta,
        'adwin_delta_confirm': confirm_delta,
        'drift_lr_prewarm_mult': float(FLAGS.drift_lr_prewarm_mult),
        'drift_lr_spike_mult': float(FLAGS.drift_lr_spike_mult),
        'lr_decay_steps': int(FLAGS.drift_lr_decay_steps),
        'fairness_window': int(FLAGS.drift_fairness_window),
        # 0 -> evaluators fall back to the fairness window (B2).
        'accuracy_window': int(FLAGS.drift_accuracy_window),
        'cooldown': int(FLAGS.drift_cooldown),
        'min_samples_per_stream': int(FLAGS.drift_min_samples_per_stream),
        'temperature_on_drift': float(FLAGS.drift_temperature_on_drift),
        'temperature_recovery_target': float(FLAGS.drift_temperature_recovery_target),
        'temperature_recovery_step': float(FLAGS.drift_temperature_recovery_step),
        'lambda_const': float(FLAGS.lambda_const),
        'accuracy_detector': str(FLAGS.accuracy_detector),
        'fairness_detector': str(FLAGS.fairness_detector),
        'fairness_signal_mode': str(FLAGS.fairness_signal_mode),
        'fairness_target': float(FLAGS.fairness_target),
        'lambda_dual_lr': float(FLAGS.lambda_dual_lr),
        'lambda_max': float(FLAGS.lambda_max),
    }
    dataset_key = _dataset_name().lower()
    overrides = {'compas': _COMPAS_FADO_OVERRIDES,
                 'folktables': _FOLKTABLES_FADO_OVERRIDES}.get(dataset_key, {})
    params.update({k: v for k, v in overrides.items() if not _explicitly_set(k)})
    if warn_multiplier > 0:
        # Derive from the confirm delta actually in use, dataset default included.
        params['adwin_delta_warn'] = params['adwin_delta_confirm'] * warn_multiplier
    return params


# Ablation arms. These are NOT part of the default model set -- running every
# preset on every scenario would multiply the pipeline cost. Request one with
# --pipeline_model=fado_no_lambda, or several with --pipeline_model=fado_no_lambda,
# fado_lambda_only, or build one ad hoc with --controller_components.
_ABLATION_MODELS = sorted(name for name in CONTROLLER_PRESETS if name.startswith('fado'))


def _controller_for_model(model_name):
    """Controller configuration for a pipeline model name."""
    name = str(model_name).strip().lower()
    spec = str(FLAGS.controller_components).strip()
    if name in ('aranyani', 'fado', 'fado_full'):
        return ControllerConfig.from_spec(spec) if spec else ControllerConfig()
    if name in CONTROLLER_PRESETS:
        return CONTROLLER_PRESETS[name]
    raise ValueError(f"No controller configuration for model '{model_name}'.")


def _supported_models_for_dataset(dataset_name):
    dataset_key = str(dataset_name).strip().lower()
    supported = {
        'adult': ['aranyani', 'aranyani_base', 'arf', 'rfr'],
        'folktables': ['aranyani', 'aranyani_base', 'arf', 'rfr'],
        'compas': ['aranyani', 'aranyani_base', 'arf', 'rfr'],
    }
    if dataset_key not in supported:
        raise ValueError(
            f"Unsupported dataset for pipeline evaluation: '{dataset_name}'."
        )
    return supported[dataset_key]


def _resolve_pipeline_models(dataset_name):
    requested = str(FLAGS.pipeline_model).strip().lower()
    supported_models = _supported_models_for_dataset(dataset_name)
    if not requested:
        return supported_models
    selectable = supported_models + _ABLATION_MODELS
    models = [token.strip() for token in requested.split(',') if token.strip()]
    for model in models:
        if model not in selectable:
            raise ValueError(
                f"Model '{model}' is not supported for dataset '{dataset_name}'. "
                f"Supported models: {supported_models}; ablation arms: {_ABLATION_MODELS}"
            )
    return models


# Fixed default seeds. Drawing them from SystemRandom made a default run
# irreproducible unless results.json survived; the paper runs should pass
# --seeds explicitly anyway.
DEFAULT_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 101,
                 111, 122, 133, 144, 155, 166, 177, 188, 199, 202)


def _parse_seed_list():
    runs = int(DEFAULT_SEED_RUNS)
    seed_tokens = [
        s.strip() for s in str(FLAGS.seeds).strip().split(',') if s.strip()
    ]
    seeds = [int(token) for token in seed_tokens]
    if not seeds:
        seeds = list(DEFAULT_SEEDS[:max(1, runs)])
        print(f"[PIPELINE] --seeds not given; using the fixed default seeds {seeds}. "
              f"Pass --seeds explicitly for the reported runs.")
    seeds = list(dict.fromkeys(seeds))
    print(f"[PIPELINE] seeds: {seeds}")
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


def _to_numpy(values):
    if values is None:
        return np.asarray([])
    return np.asarray(values)


def _has_samples(values):
    return values is not None and len(values) > 0


def _is_valid_stratify_target(values):
    if values is None:
        return False
    uniques, counts = np.unique(values, return_counts=True)
    if len(uniques) < 2:
        return False
    return bool(np.all(counts >= 2))


def _build_stratify_target(y_values, a_values):
    y_arr = np.asarray(y_values)
    a_arr = np.asarray(a_values)
    if len(y_arr) != len(a_arr):
        return None

    y_as_str = y_arr.astype(str)
    a_as_str = a_arr.astype(str)
    joint = np.asarray([f'{y}_{a}' for y, a in zip(y_as_str, a_as_str)])
    if _is_valid_stratify_target(joint):
        return joint
    if _is_valid_stratify_target(y_arr):
        return y_arr
    return None


def _smart_split(x_values, y_values, a_values, seed, test_size=0.2):
    n_samples = len(x_values)
    if n_samples < 2:
        raise ValueError('Not enough samples to build train/test splits.')

    x_arr = np.asarray(x_values)
    y_arr = np.asarray(y_values)
    a_arr = np.asarray(a_values)
    indices = np.arange(n_samples)
    stratify = _build_stratify_target(y_arr, a_arr)

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(test_size),
            random_state=int(seed),
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(test_size),
            random_state=int(seed),
            shuffle=True,
            stratify=None,
        )

    return (
        x_arr[train_idx],
        x_arr[test_idx],
        y_arr[train_idx],
        y_arr[test_idx],
        a_arr[train_idx],
        a_arr[test_idx],
    )


def _ensure_train_test(x_train, x_test, y_train, y_test, a_train, a_test, seed):
    x_train = _to_numpy(x_train)
    x_test = _to_numpy(x_test)
    y_train = _to_numpy(y_train)
    y_test = _to_numpy(y_test)
    a_train = _to_numpy(a_train)
    a_test = _to_numpy(a_test)

    has_train = _has_samples(x_train) and _has_samples(y_train) and _has_samples(a_train)
    has_test = _has_samples(x_test) and _has_samples(y_test) and _has_samples(a_test)

    if has_train and has_test:
        return x_train, x_test, y_train, y_test, a_train, a_test

    if has_train:
        return _smart_split(x_train, y_train, a_train, seed=seed)
    if has_test:
        return _smart_split(x_test, y_test, a_test, seed=seed)

    raise ValueError('Dataset has no usable samples for train/test evaluation.')


def _maybe_subsample_folktables(
        x_train, y_train, a_train,
        x_test, y_test, a_test,
        seed,
        marginals_train=None,
        marginals_test=None,
):
    """Uniformly subsample both Folktables splits to
    ``FLAGS.folktables_subsample_fraction`` of their original size.

    The subsample is reproducible per seed (so all four models within a seed
    see identical rows and the paired Wilcoxon protocol holds) but varies
    across seeds (so the cross-seed variance reflects sampling uncertainty
    rather than a single fixed slice).

    When ``marginals_train`` / ``marginals_test`` are provided (intersectional
    mode), the same row indices are applied to those marginal-attribute
    arrays so they stay aligned with x/y/a row-for-row.
    """
    fraction = float(FLAGS.folktables_subsample_fraction)
    has_marginals = marginals_train is not None and marginals_test is not None
    if not (0.0 < fraction < 1.0):
        if has_marginals:
            return (x_train, y_train, a_train, x_test, y_test, a_test,
                    marginals_train, marginals_test)
        return x_train, y_train, a_train, x_test, y_test, a_test

    rng = np.random.default_rng(seed)

    def _take(x, y, a, m=None):
        n = len(y)
        k = max(1, int(round(n * fraction)))
        idx = rng.choice(n, size=k, replace=False)
        idx.sort()
        x_arr = np.asarray(x)
        if m is None:
            return x_arr[idx], np.asarray(y)[idx], np.asarray(a)[idx], None
        return (
            x_arr[idx], np.asarray(y)[idx], np.asarray(a)[idx],
            np.asarray(m)[idx],
        )

    x_train, y_train, a_train, marginals_train = _take(
        x_train, y_train, a_train, marginals_train
    )
    x_test, y_test, a_test, marginals_test = _take(
        x_test, y_test, a_test, marginals_test
    )

    print(
        f'[FOLKTABLES-SUBSAMPLE] fraction={fraction:.4f} seed={seed} '
        f'-> train={len(y_train)}, test={len(y_test)}'
    )
    if has_marginals:
        return (x_train, y_train, a_train, x_test, y_test, a_test,
                marginals_train, marginals_test)
    return x_train, y_train, a_train, x_test, y_test, a_test


def _load_dataset_splits(dataset_key, scenario_name, seed):
    intersectional = bool(FLAGS.intersectional)
    if dataset_key == 'adult':
        x_train, _, y_train, _, a_train, _ = read_adult(False, drift_scenario=None)
        x_test, y_test, a_test = load_drifted_test_set(scenario_name)
        print("size of adult train/test splits:", len(y_train), len(y_test))
        x_train, x_test, y_train, y_test, a_train, a_test = _ensure_train_test(
            x_train, x_test, y_train, y_test, a_train, a_test, seed=seed)
        return x_train, x_test, y_train, y_test, a_train, a_test, None

    if dataset_key == 'folktables':
        loader_out = read_folktables(
            train_year=FOLKTABLES_PIPELINE_TRAIN_YEAR,
            test_years=FOLKTABLES_PIPELINE_TEST_YEARS,
            state=[state.strip() for state in str(FLAGS.folktables_states).split(',') if state.strip()],
            horizon=FLAGS.folktables_horizon,
            sensitive_attribute=FLAGS.folktables_sensitive_attribute,
            intersectional=intersectional,
        )
        if intersectional:
            (x_train, x_test, y_train, y_test, a_train, a_test,
             _num_groups, a_marginals) = loader_out
            (x_train, y_train, a_train, x_test, y_test, a_test,
             m_train, m_test) = _maybe_subsample_folktables(
                x_train, y_train, a_train, x_test, y_test, a_test, seed=seed,
                marginals_train=a_marginals['train'],
                marginals_test=a_marginals['test'],
            )
            marginals = {
                'attr_names': a_marginals['attr_names'],
                'train': m_train, 'test': m_test,
            }
        else:
            x_train, x_test, y_train, y_test, a_train, a_test, _ = loader_out
            x_train, y_train, a_train, x_test, y_test, a_test = (
                _maybe_subsample_folktables(
                    x_train, y_train, a_train, x_test, y_test, a_test, seed=seed,
                )
            )
            marginals = None
        x_train, x_test, y_train, y_test, a_train, a_test = _ensure_train_test(
            x_train, x_test, y_train, y_test, a_train, a_test, seed=seed)
        return x_train, x_test, y_train, y_test, a_train, a_test, marginals

    if dataset_key == 'compas':
        # Drift-aware loader: splits raw rows by seed, applies the named
        # scenario to the test slice only, then fits the encoder on train
        # so the scenario edit on race / age_cat / c_charge_degree
        # propagates correctly through the one-hot transformer.
        compas_out = read_compas_train_test(
            scenario_name=scenario_name,
            path=FLAGS.compas_path,
            seed=seed,
            intersectional=intersectional,
        )
        if intersectional:
            (x_train, x_test, y_train, y_test, a_train, a_test,
             a_marginals) = compas_out
            return (x_train, x_test, y_train, y_test, a_train, a_test,
                    a_marginals)
        return (*compas_out, None)

    raise ValueError(f"Unsupported dataset for pipeline evaluation: '{dataset_key}'.")


def _phase_layout(dataset_key, n_samples):
    """The stream's phases as ``[(slug, start, end), ...]`` sample ranges.

    Adult and COMPAS scenarios lay a stream out as phases (``SPLITS`` holds the
    phase END fractions, ``PHASE_LABELS`` their names). Folktables has no
    injected drift: its stream is the test years in order, each contributing an
    equal share (see ``read_folktables``), so each year is a phase.
    """
    dataset_key = str(dataset_key).lower()
    if dataset_key == 'folktables':
        years = FOLKTABLES_PIPELINE_TEST_YEARS
        splits = [(i + 1) / len(years) for i in range(len(years))]
        labels = [f'year {year}' for year in years]
    else:
        module = _compas_scenarios if dataset_key == 'compas' else _adult_scenarios
        splits = list(getattr(module, 'SPLITS', []) or [])
        labels = list(getattr(module, 'PHASE_LABELS', []) or [])
    n_samples = int(n_samples or 0)
    if len(splits) < 2 or len(labels) != len(splits) or not n_samples:
        return []
    ends = [int(float(s) * n_samples) for s in splits]
    ends[-1] = n_samples
    starts = [0] + ends[:-1]
    slugs = [str(label).strip().lower().replace(' ', '_') for label in labels]
    return list(zip(slugs, starts, ends))


def _change_points(dataset_key, n_samples):
    """Sample indices at which the concept actually changes.

    Every phase boundary is a change event: COMPAS ``[0.30, 0.70, 1.0]``
    changes at 30% (drift onset) and again at 70% (the recovery edge, where the
    concept reverts). Scoring against a single onset would count a detection
    at the recovery edge as a very late detection of the first drift instead
    of a prompt detection of the second.
    """
    return [start for _, start, _ in _phase_layout(dataset_key, n_samples)[1:]]


# 'lambda' exists only for the FADO arm (2.1); the others skip it.
_PHASE_SOURCE_METRICS = ('accuracy', 'dp', 'eo', 'lambda')


def _phase_metrics(stream, dataset_key):
    """Per-phase and post-drift means of the prequential curves (decision 2.4).

    The whole-stream mean (``stream_final_*``) dilutes the post-drift period,
    the only place FADO differs from the baseline, into the warmup. This adds:

      phase_<slug>_<metric>   mean over one phase, e.g. phase_drift_dp
      post_drift_<metric>     mean from the first change point to the end

    They average the same rolling curves as ``stream_final_*``, so the
    whole-stream value is the length-weighted mean of the phases. The curves
    are rolling-window values, so the first window of a phase still carries
    part of the previous one. Phases are structural: a no_drift stream gets the
    same boundaries, which makes it the control for the drifted scenarios.
    """
    if not isinstance(stream, dict):
        return {}
    out = {}
    for metric in _PHASE_SOURCE_METRICS:
        series = stream.get(metric)
        if series is None:
            continue
        values = np.asarray(series, dtype=float).reshape(-1)
        phases = _phase_layout(dataset_key, len(values))
        for slug, start, end in phases:
            if end > start:
                out[f'phase_{slug}_{metric}'] = float(np.nanmean(values[start:end]))
        if len(phases) >= 2 and phases[1][1] < len(values):
            out[f'post_drift_{metric}'] = float(np.nanmean(values[phases[1][1]:]))
    return out


def _detector_metrics(points, change_points, n_samples, tolerance, precedence=0):
    """Stream-learning detection quality for ONE detector's event list.

    Follows the usual drift-detection convention: a detection inside the
    acceptable window ``[cp - precedence, cp + tolerance]`` around a change
    point is a true positive for that change point, everything else is a false
    alarm. Returns, for that detector:

      detections    raw number of alarms raised
      true_positives / false_alarms
      mdr    Missed Detection Rate -- change points never detected (lower better)
      mtd    Mean Time to Detection -- mean delay over detected change points,
             in samples (lower better)
      mtfa   Mean Time between False Alarms, in samples. Needs >= 2 false
             alarms to be defined; NaN otherwise (higher better)
      far    False Alarm Rate, false alarms per sample. Defined whenever there
             is at least one, so it is the metric to optimise a sweep on
             (lower better)
      mtr    Mean Time Ratio, (MTFA / MTD) * (1 - MDR), the aggregate from the
             drift-detection literature (higher better). NaN when MTFA or MTD
             is undefined.

    ``change_points`` empty (a no-drift stream) means every alarm is a false
    alarm by construction -- which is what makes the no_drift arm the clean
    measurement of the false-alarm rate.
    """
    points = sorted(int(v) for v in (points or []))
    n_samples = int(n_samples or 0)
    tolerance = max(1, int(tolerance))
    out = {'detections': len(points)}

    matched, used = {}, set()
    for cp in change_points:
        lo, hi = cp - int(precedence), cp + tolerance
        for i, point in enumerate(points):
            if i in used:
                continue
            if lo <= point <= hi:
                matched[cp] = point
                used.add(i)
                break
    false_alarms = [p for i, p in enumerate(points) if i not in used]

    out['true_positives'] = len(matched)
    out['false_alarms'] = len(false_alarms)
    out['far'] = (len(false_alarms) / n_samples) if n_samples else float('nan')

    if change_points:
        out['mdr'] = 1.0 - len(matched) / len(change_points)
        delays = [max(0, matched[cp] - cp) for cp in matched]
        out['mtd'] = float(np.mean(delays)) if delays else float('nan')
    else:
        out['mdr'] = float('nan')
        out['mtd'] = float('nan')

    if len(false_alarms) >= 2:
        gaps = np.diff(false_alarms)
        out['mtfa'] = float(np.mean(gaps))
    else:
        out['mtfa'] = float('nan')

    if (not math.isnan(out['mtfa']) and not math.isnan(out['mtd'])
            and out['mtd'] > 0 and not math.isnan(out['mdr'])):
        out['mtr'] = float(out['mtfa'] / out['mtd'] * (1.0 - out['mdr']))
    else:
        out['mtr'] = float('nan')
    return out


def _detection_metrics(stream, dataset_key, scenario_name):
    """Detection quality for BOTH detectors in a run, reported separately.

    The two detectors watch different things and are scored independently:

      ``accuracy_*``   the ADWIN pair on the prequential error stream -- the
                       drift detector that drives the LR/temperature reaction.
      ``fairness_*``   the fairness monitor (detector x signal design, see
                       src/models/forest/fairness_signal.py) -- observation
                       only, it records what it would have signalled.

    Every metric name is prefixed, so nothing in the results table is ambiguous
    about which detector it describes.
    """
    if not isinstance(stream, dict):
        return {}
    n_samples = int(stream.get('n_samples') or 0)
    no_drift = 'no_drift' in str(scenario_name).lower()
    change_points = [] if no_drift else _change_points(dataset_key, n_samples)

    # A detector cannot see a change before its own window has refilled, so the
    # acceptable delay defaults to one window rather than an arbitrary constant.
    used = stream.get('static_params_used') or {}
    tolerance = int(FLAGS.detection_tolerance)
    if tolerance <= 0:
        tolerance = int(used.get('accuracy_window')
                        or used.get('fairness_window')
                        or FLAGS.drift_fairness_window)

    out = {'n_change_points': len(change_points),
           'detection_tolerance': int(tolerance)}
    for label, key in (('accuracy', 'drifted_points'),
                       ('fairness', 'fairness_drifted_points')):
        points = stream.get(key)
        if points is None:
            continue
        for metric, value in _detector_metrics(
                points, change_points, n_samples, tolerance).items():
            out[f'{label}_{metric}'] = value
    return out


def _extract_test_metrics(stream):
    return {
        'accuracy': _mean_stream_metric(stream.get('accuracy')),
        'dp': _mean_stream_metric(stream.get('dp')),
        'eo': _mean_stream_metric(stream.get('eo')),
    }


def _compute_marginal_dp_eo(stream, a_marginals_test, attr_names):
    """Compute per-attribute (marginal) DP/EO from accumulated predictions.

    Intersectional mode reports composite-group DP/EO via the existing
    ``dp`` / ``eo`` stream keys (these already use ``a_test``, which the
    loader sets to the composite group index). This function adds the
    *marginal* DP/EO for each individual protected attribute as a single
    end-of-stream scalar per attribute, computed against the full recorded
    prediction stream rather than the rolling window. Returns
    ``{'<attr>': {'dp': float, 'eo': float}}`` for each attribute, or
    ``None`` if predictions weren't recorded.
    """
    from src.helpers import utils as _utils  # noqa: WPS433 -- local import
    y_preds = stream.get('y_preds_all') if isinstance(stream, dict) else None
    y_true = stream.get('y_true_all') if isinstance(stream, dict) else None
    if not y_preds or not y_true:
        return None
    a_marginals_test = np.asarray(a_marginals_test, dtype=np.int32)
    if a_marginals_test.ndim != 2 or a_marginals_test.shape[1] != len(attr_names):
        return None
    n = min(len(y_preds), len(y_true), a_marginals_test.shape[0])
    if n == 0:
        return None
    preds = list(y_preds[:n])
    trues = list(y_true[:n])
    out = {}
    for idx, attr in enumerate(attr_names):
        marginal_a = a_marginals_test[:n, idx].astype(np.int32).tolist()
        dp_val, _ = _utils.get_demographic_parity(preds, marginal_a)
        eo_val, _ = _utils.get_equalized_odds(preds, marginal_a, trues)
        out[str(attr)] = {
            'dp': float(dp_val),
            'eo': float(eo_val),
        }
    return out


def _mean_stream_metric(values):
    if values is None:
        return None
    if isinstance(values, numbers.Number):
        return float(values)
    if isinstance(values, np.ndarray):
        flattened = values.reshape(-1).tolist()
    elif isinstance(values, (list, tuple)):
        flattened = values
    else:
        return None

    numeric_values = [float(v) for v in flattened if isinstance(v, numbers.Number)]
    if not numeric_values:
        return None
    return float(np.asarray(numeric_values, dtype=float).mean())


def _numeric_or_nan(value):
    if isinstance(value, numbers.Number):
        return float(value)
    return float(np.nan)


_PRETRAIN_CACHE = {}
_PRETRAIN_DIR = None


def _pretrain_dir():
    global _PRETRAIN_DIR
    if _PRETRAIN_DIR is None:
        _PRETRAIN_DIR = tempfile.mkdtemp(prefix='fado_pretrain_')
        atexit.register(shutil.rmtree, _PRETRAIN_DIR, True)
    return _PRETRAIN_DIR


def _pretrain_cache_key(x_train, y_train, a_train, seed):
    digest = hashlib.sha1()
    for array in (x_train, y_train, a_train):
        digest.update(np.ascontiguousarray(array).tobytes())
    return (
        digest.hexdigest(), int(seed if seed is not None else -1),
        int(FLAGS.depth), int(FLAGS.num_trees), _effective_lambda(),
        int(FLAGS.batch_size), str(FLAGS.constraint_type),
        str(FLAGS.gradient_type), bool(FLAGS.compute_fairness),
        _effective_fairness_window(),
    )


def _pretrain_aranyani(x_train, y_train, a_train, data_dim, seed):
    """Pre-train the forest once and hand every arm an identical copy.

    D1: the arms used to re-train independently and rely on seed determinism to
    land on the same starting weights. Training once and reloading removes that
    assumption entirely (and halves the cost of a two-arm run). The shared
    pre-training necessarily runs on ONE leaf-probability path -- 'auto' -- so
    it sits outside the FADO-vs-Base comparison; its cost is reported
    separately as ``pretrain_timing``.
    """
    key = _pretrain_cache_key(x_train, y_train, a_train, seed)
    if key in _PRETRAIN_CACHE:
        checkpoint, timing = _PRETRAIN_CACHE[key]
        print(f"[PIPELINE] Reusing the pre-trained forest at {checkpoint}.pkl")
        return forest.FairDecisionForest.load(checkpoint), dict(timing)

    model = forest.FairDecisionForest(
        num_trees=int(FLAGS.num_trees),
        tree_depth=int(FLAGS.depth),
        data_dim=data_dim,
        num_classes=2,
        leaf_probability='auto',
    )
    timing = {}
    aranyani.train_online(
        model,
        x_train,
        y_train,
        a_train,
        data_dim=data_dim,
        batch_size=max(1, int(FLAGS.batch_size)),
        tree_depth=int(FLAGS.depth),
        compute_fairness=bool(FLAGS.compute_fairness),
        lambda_const=_effective_lambda(),
        num_trees=int(FLAGS.num_trees),
        constraint_type=FLAGS.constraint_type,
        gradient_type=FLAGS.gradient_type,
        local_run=True,
        # D3: the same window the evaluators use, so pre-training and
        # prequential evaluation regularise against one fairness signal.
        fairness_window=_effective_fairness_window(),
        use_incremental_fairness=True,
        timing_sink=timing,
    )
    checkpoint = os.path.join(_pretrain_dir(), f'pretrain_{abs(hash(key)):x}')
    model.save(checkpoint)
    _PRETRAIN_CACHE[key] = (checkpoint, dict(timing))
    return model, dict(timing)


def _run_aranyani_train_then_test(
    x_train,
    y_train,
    a_train,
    x_test,
    y_test,
    a_test,
    dataset_name,
    seed=None,
    use_drift_controller=True,
    controller=None,
):
    if seed is not None:
        _set_global_seed(int(seed))

    x_train_arr = np.asarray(x_train, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int32)
    a_train_arr = np.asarray(a_train, dtype=np.int32)
    x_test_arr = np.asarray(x_test, dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=np.int32)
    a_test_arr = np.asarray(a_test, dtype=np.int32)

    data_dim = int(x_train_arr.shape[1])
    tree_depth = int(FLAGS.depth)
    num_trees = int(FLAGS.num_trees)
    lambda_const = _effective_lambda()

    model, pretrain_timing = _pretrain_aranyani(
        x_train_arr, y_train_arr, a_train_arr, data_dim, seed)

    # The efficiency optimisations stay FADO-only for the prequential phase:
    # the baseline arm is pinned to the original mask path, FADO takes the
    # faster path for this depth.
    model.set_leaf_probability('auto' if use_drift_controller else 'mask')

    # Re-seed so both arms enter evaluation with the same RNG state regardless
    # of whether this call had to pre-train or reused the cached forest.
    if seed is not None:
        _set_global_seed(int(seed))

    def _with_pretrain_timing(stream):
        if isinstance(stream, dict) and pretrain_timing:
            stream['pretrain_timing'] = dict(pretrain_timing)
        return stream

    if use_drift_controller:
        return _with_pretrain_timing(evaluate_over_timesteps(
            model,
            x_test_arr,
            y_test_arr,
            a_test_arr,
            data_dim=data_dim,
            test_then_train=True,
            lambda_const=lambda_const,
            tree_depth=tree_depth,
            num_trees=num_trees,
            static_params=_build_aranyani_static_params(),
            controller=controller,
        ))

    print(
        "[PIPELINE][ARANYANI-BASE] Drift controller disabled; "
        "evaluating with pure Aranyani prequential loop (test-then-train)."
    )
    return _with_pretrain_timing(evaluate_aranyani_baseline_over_timesteps(
        model,
        x_test_arr,
        y_test_arr,
        a_test_arr,
        data_dim=data_dim,
        test_then_train=True,
        lambda_const=lambda_const,
        tree_depth=tree_depth,
        num_trees=num_trees,
        fairness_window=_effective_fairness_window(),
        static_params=_build_aranyani_static_params(),
        use_incremental_fairness=False,
    ))


def _run_arf_train_then_test(x_train, y_train, a_train, x_test, y_test, a_test, seed):
    # The ARF arm used to skip this, so it entered evaluation with whatever RNG
    # state the previous arm happened to leave behind.
    if seed is not None:
        _set_global_seed(int(seed))
    fairness_window = _effective_fairness_window()
    _, trained_model = evaluate_arf_over_timesteps(
        np.asarray(x_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.int32),
        np.asarray(a_train, dtype=np.int32),
        seed=seed,
        online_batch_size=1,
        accuracy_window=_effective_accuracy_window(),
        fairness_window=fairness_window,
        test_then_train=True,
        return_model=True,
    )
    return evaluate_arf_over_timesteps(
        np.asarray(x_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.int32),
        np.asarray(a_test, dtype=np.int32),
        seed=seed,
        online_batch_size=1,
        accuracy_window=_effective_accuracy_window(),
        fairness_window=fairness_window,
        model=trained_model,
        test_then_train=True,
    )


def _run_rfr_train_then_test(x_train, y_train, a_train, x_test, y_test, a_test, seed=None):
    if seed is not None:
        _set_global_seed(int(seed))
    fairness_window = _effective_fairness_window()
    _, trained_model = evaluate_rfr_over_timesteps(
        np.asarray(x_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.int32),
        np.asarray(a_train, dtype=np.int32),
        approach=RFR_CONFIG['approach'],
        backbone=RFR_CONFIG['backbone'],
        hidden_dim=RFR_CONFIG['hidden_dim'],
        n_ensemble=RFR_CONFIG['n_ensemble'],
        learning_rate=RFR_CONFIG['learning_rate'],
        rho=RFR_CONFIG['rho'],
        penalty_coefficient=RFR_CONFIG['penalty_coefficient'],
        fcr_threshold=RFR_CONFIG['fcr_threshold'],
        train_batch_size=1,
        buffer_size=RFR_CONFIG['buffer_size'],
        adv_hidden_dim=RFR_CONFIG['adv_hidden_dim'],
        accuracy_window=_effective_accuracy_window(),
        fairness_window=fairness_window,
        test_then_train=True,
        return_model=True,
    )
    return evaluate_rfr_over_timesteps(
        np.asarray(x_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.int32),
        np.asarray(a_test, dtype=np.int32),
        approach=RFR_CONFIG['approach'],
        backbone=RFR_CONFIG['backbone'],
        hidden_dim=RFR_CONFIG['hidden_dim'],
        n_ensemble=RFR_CONFIG['n_ensemble'],
        learning_rate=RFR_CONFIG['learning_rate'],
        rho=RFR_CONFIG['rho'],
        penalty_coefficient=RFR_CONFIG['penalty_coefficient'],
        fcr_threshold=RFR_CONFIG['fcr_threshold'],
        train_batch_size=1,
        buffer_size=RFR_CONFIG['buffer_size'],
        adv_hidden_dim=RFR_CONFIG['adv_hidden_dim'],
        accuracy_window=_effective_accuracy_window(),
        fairness_window=fairness_window,
        model=trained_model,
        test_then_train=True,
    )


def _evaluate_selected_model(
    model_name,
    dataset_name,
    x_train,
    y_train,
    a_train,
    x_test,
    y_test,
    a_test,
    seed=None,
):
    if model_name == 'aranyani' or model_name in CONTROLLER_PRESETS:
        return _run_aranyani_train_then_test(
            x_train,
            y_train,
            a_train,
            x_test,
            y_test,
            a_test,
            dataset_name=dataset_name,
            seed=seed,
            use_drift_controller=True,
            controller=_controller_for_model(model_name),
        )
    if model_name == 'aranyani_base':
        return _run_aranyani_train_then_test(
            x_train,
            y_train,
            a_train,
            x_test,
            y_test,
            a_test,
            dataset_name=dataset_name,
            seed=seed,
            use_drift_controller=False,
        )
    if model_name == 'arf':
        return _run_arf_train_then_test(x_train, y_train, a_train, x_test, y_test, a_test, seed=seed)
    if model_name == 'rfr':
        return _run_rfr_train_then_test(x_train, y_train, a_train, x_test, y_test, a_test, seed=seed)
    raise ValueError(f'Unsupported model: {model_name}')


def _single_scenario(
    model_name,
    dataset_name,
    scenario_name,
    output_dir,
    seed=None,
):
    print(f"\n{'#' * 80}")
    print(f"# Evaluating scenario ({model_name}): {scenario_name}")
    print(f"# {ALL_SCENARIO_DESCRIPTIONS.get(scenario_name, '')}")
    print(f"{'#' * 80}\n")
    start = time.time()

    x_train, x_test, y_train, y_test, a_train, a_test, a_marginals = (
        _load_dataset_splits(
            dataset_key=str(dataset_name).lower(),
            scenario_name=scenario_name,
            seed=seed,
        )
    )
    print(
        f"train_samples={len(x_train)} test_samples={len(x_test)} "
        f"features={x_train.shape[1] if len(x_train) > 0 else x_test.shape[1]}"
    )
    if a_marginals is not None:
        n_groups_composite = len({int(v) for v in np.asarray(a_train).tolist()})
        print(
            f"[INTERSECTIONAL] attrs={a_marginals['attr_names']} "
            f"composite_groups_train={n_groups_composite}"
        )

    stream = _evaluate_selected_model(
        model_name=model_name,
        dataset_name=dataset_name,
        x_train=x_train,
        y_train=y_train,
        a_train=a_train,
        x_test=x_test,
        y_test=y_test,
        a_test=a_test,
        seed=seed,
    )
    test_metrics = _extract_test_metrics(stream)
    if a_marginals is not None:
        marginal_metrics = _compute_marginal_dp_eo(
            stream=stream,
            a_marginals_test=a_marginals['test'],
            attr_names=a_marginals['attr_names'],
        )
        if marginal_metrics:
            test_metrics['marginal'] = marginal_metrics
            stream['marginal_metrics'] = marginal_metrics

    os.makedirs(output_dir, exist_ok=True)
    return {
        'model': model_name,
        'scenario': scenario_name,
        'test_metrics': test_metrics,
        'timestep_results': stream,
        'elapsed_seconds': round(time.time() - start, 1),
    }


def run_scenarios(model_name, dataset_name, output_dir=OUTPUT_DIR, scenario_filter=None, seed=None):
    dataset_key = str(dataset_name).lower()
    print(f"\n{'=' * 80}")
    if dataset_key == 'adult':
        scenarios = list(SCENARIOS.keys())
        print(f" Running all {len(scenarios)} drift scenarios")
    elif dataset_key == 'folktables':
        scenarios = ['folktables_2015_to_2017_2018']
        print(' Running Folktables train-then-test: train=2015, test=2017+2018')
    elif dataset_key == 'compas':
        scenarios = list(COMPAS_SCENARIOS.keys())
        print(f' Running all {len(scenarios)} COMPAS drift scenarios')
    else:
        raise ValueError(
            f"Unsupported dataset for pipeline evaluation: '{dataset_name}'."
        )
    print(f" Model: {model_name}")
    print(f" Dataset: {dataset_name}")
    print(f" Output: {os.path.abspath(output_dir)}/")
    print(f"{'=' * 80}\n")

    per_scenario_datasets = {'adult', 'compas'}
    results = []
    for idx, scenario_name in enumerate(scenarios, 1):
        if (
            dataset_key in per_scenario_datasets
            and scenario_filter is not None
            and scenario_name != scenario_filter
        ):
            continue
        print(f"\n>>> [{idx}/{len(scenarios)}] {scenario_name}")
        result = _single_scenario(
            model_name=model_name,
            dataset_name=dataset_key,
            scenario_name=(
                scenario_name if dataset_key in per_scenario_datasets else dataset_key
            ),
            output_dir=output_dir,
            seed=seed,
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
    rows_for_disk = []
    for result in results:
        tm = result.get('test_metrics') or {}
        ts = result.get('timestep_results') or {}
        marginal = tm.get('marginal') if isinstance(tm, dict) else None
        row = {
            'model': result.get('model'),
            'scenario': result.get('scenario'),
            'accuracy': float(tm.get('accuracy')),
            'dp': float(tm.get('dp')),
            'eo': float(tm.get('eo')),
            'stream_final_accuracy': _mean_stream_metric(ts.get('accuracy')),
            'stream_final_accuracy_cumulative': _mean_stream_metric(
                ts.get('accuracy_cumulative')
            ),
            'stream_final_dp': _mean_stream_metric(ts.get('dp')),
            'stream_final_eo': _mean_stream_metric(ts.get('eo')),
            'stream_final_lambda': _mean_stream_metric(ts.get('lambda')),
            'fairness_resets': len(ts.get('fairness_resets') or []),
            'elapsed_seconds': result.get('elapsed_seconds'),
            'error': result.get('error'),
            'test_metrics': tm,
            'timestep_results': ts,
        }
        # Runtime columns, so the FADO-only efficiency work (recursive leaf
        # probabilities + incremental fairness counters) is measurable in the
        # same results table as accuracy/DP/EO instead of being asserted.
        row.update(_detection_metrics(ts, dataset_key, result.get('scenario')))
        row.update(_phase_metrics(ts, dataset_key))
        # Top level so the lambda trade-off plot can read it from results.json;
        # None for the arms without a fairness penalty (ARF, RFR).
        row['lambda_const'] = (ts.get('static_params_used') or {}).get('lambda_const')
        timing = ts.get('timing') if isinstance(ts, dict) else None
        if isinstance(timing, dict):
            row['timing'] = timing
            row['wall_seconds'] = timing.get('wall_seconds')
            row['ms_per_sample'] = timing.get('ms_per_sample')
            row['samples_per_second'] = timing.get('samples_per_second')
            for phase_name, phase_stats in (timing.get('phases') or {}).items():
                if isinstance(phase_stats, dict):
                    row[f'seconds_{phase_name}'] = phase_stats.get('seconds')
        pretrain_timing = ts.get('pretrain_timing') if isinstance(ts, dict) else None
        if isinstance(pretrain_timing, dict):
            row['pretrain_timing'] = pretrain_timing
            row['pretrain_wall_seconds'] = pretrain_timing.get('wall_seconds')
        if marginal:
            row['marginal'] = marginal
            # Flatten marginal DP/EO into top-level keys so significance_tests
            # (and any consumer that reads scalar columns) can pick them up
            # without inspecting the nested ``marginal`` blob.
            for attr_name, vals in marginal.items():
                if not isinstance(vals, dict):
                    continue
                if vals.get('dp') is not None:
                    row[f'dp_{attr_name}'] = float(vals['dp'])
                if vals.get('eo') is not None:
                    row[f'eo_{attr_name}'] = float(vals['eo'])
        rows.append(row)
        rows_for_disk.append({
            k: v for k, v in row.items() if k not in ('test_metrics', 'timestep_results')
        })

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(rows_for_disk, f, indent=2, default=str)
    print(f'Results saved to: {results_path}')

    print(f"\n{'=' * 80}")
    print(' SUMMARY')
    print(f"{'=' * 80}")
    print(
        f"{'Scenario':<36s} {'Acc':>10s} {'DP':>10s} {'EO':>10s} "
        f"{'ms/sample':>11s} {'wall(s)':>9s}"
    )
    print('-' * 80)
    for row in rows:
        acc = f"{row['accuracy']:.4f}"
        dp = f"{row['dp']:.4f}"
        eo = f"{row['eo']:.4f}"
        mps = row.get('ms_per_sample')
        wall = row.get('wall_seconds')
        mps_s = f"{mps:.3f}" if isinstance(mps, numbers.Number) else '-'
        wall_s = f"{wall:.1f}" if isinstance(wall, numbers.Number) else '-'
        print(
            f"{str(row['scenario']):<36s} {acc:>10s} {dp:>10s} {eo:>10s} "
            f"{mps_s:>11s} {wall_s:>9s}"
        )

    return rows


# Arms whose scores the paired deltas are computed against.
_BASELINE_ARM = 'aranyani_base'


def _log_sweep_summary(summary, models_to_run):
    """Write flat, paired metrics to wandb.summary for sweep optimisation.

    A sweep reads ``wandb.summary``, not the per-row ``wandb.log`` stream, so
    the quantity a sweep should optimise has to live here. The paired deltas
    are what the controller has to be judged on: absolute DP moves with the
    scenario and the seed, the FADO-minus-baseline difference does not, and
    both arms in a run share one pre-trained forest so the pairing is exact.
    """
    if wandb is None:
        return
    by_model = {}
    if summary.get('mode') == 'single':
        by_model = summary.get('metrics_by_model') or {}
    else:
        for model_row in summary.get('rows') or []:
            name = model_row.get('model')
            scenario_rows = model_row.get('rows') or []
            merged = {}
            for scenario_row in scenario_rows:
                for metric, stats in (scenario_row.get('metrics') or {}).items():
                    merged.setdefault(metric, []).append(stats.get('mean'))
            by_model[name] = {
                metric: {'mean': float(np.nanmean([v for v in values if v is not None]))
                         if any(v is not None for v in values) else None}
                for metric, values in merged.items()
            }

    # Built as a plain dict and written once: wandb's Summary does not support
    # ``in``, so testing it for a key raised KeyError and aborted the run
    # before the sweep metric was written.
    out = {}
    for model_name, metrics in by_model.items():
        for metric_name, stats in (metrics or {}).items():
            if isinstance(stats, dict) and stats.get('mean') is not None:
                out[f'{model_name}/{metric_name}'] = stats['mean']

    base = by_model.get(_BASELINE_ARM) or {}
    treatments = [m for m in by_model if m != _BASELINE_ARM and m in models_to_run]
    for treatment in treatments:
        arm = by_model.get(treatment) or {}

        def _mean(metrics_dict, key):
            stats = (metrics_dict or {}).get(key)
            return stats.get('mean') if isinstance(stats, dict) else None

        for source, target in (('stream_final_dp', 'delta_dp'),
                               ('stream_final_eo', 'delta_eo'),
                               ('post_drift_dp', 'delta_dp_post_drift'),
                               ('post_drift_eo', 'delta_eo_post_drift')):
            b, a = _mean(base, source), _mean(arm, source)
            if b is not None and a is not None:
                # positive = the treatment arm is FAIRER than the baseline
                out[f'{treatment}/{target}'] = float(b) - float(a)
        for source, target in (('stream_final_accuracy', 'delta_accuracy'),
                               ('post_drift_accuracy', 'delta_accuracy_post_drift')):
            b, a = _mean(base, source), _mean(arm, source)
            if b is not None and a is not None:
                # positive = the treatment arm is MORE ACCURATE than the baseline
                out[f'{treatment}/{target}'] = float(a) - float(b)

    # Unprefixed aliases for the primary arm, so a sweep config can name a
    # metric without knowing which arm it is.
    primary = next((m for m in ('aranyani', 'fado_full') if m in by_model),
                   treatments[0] if treatments else None)
    if primary:
        for metric in ('delta_dp', 'delta_eo', 'delta_accuracy',
                       'delta_dp_post_drift', 'delta_eo_post_drift',
                       'delta_accuracy_post_drift'):
            key = f'{primary}/{metric}'
            if key in out:
                out[metric] = out[key]
        for metric in ('fairness_far', 'fairness_mdr', 'fairness_mtd',
                       'fairness_mtfa', 'fairness_mtr', 'fairness_false_alarms',
                       'accuracy_far', 'accuracy_mdr', 'accuracy_mtd',
                       'accuracy_mtfa', 'accuracy_mtr', 'accuracy_false_alarms',
                       'stream_final_dp', 'stream_final_accuracy',
                       'post_drift_dp', 'post_drift_accuracy'):
            key = f'{primary}/{metric}'
            if key in out:
                out[metric] = out[key]
    wandb.summary.update(out)


_SUMMARY_METRICS = [
    'accuracy', 'dp', 'eo',
    'stream_final_accuracy', 'stream_final_dp', 'stream_final_eo',
    'stream_final_accuracy_cumulative',
    'post_drift_accuracy', 'post_drift_dp', 'post_drift_eo',
    'stream_final_lambda', 'post_drift_lambda', 'fairness_resets',
    'wall_seconds', 'ms_per_sample', 'samples_per_second',
    'seconds_predict', 'seconds_fairness_metrics', 'seconds_train_step',
    'pretrain_wall_seconds',
    'accuracy_detections', 'accuracy_true_positives', 'accuracy_false_alarms',
    'accuracy_far', 'accuracy_mdr', 'accuracy_mtd', 'accuracy_mtfa', 'accuracy_mtr',
    'fairness_detections', 'fairness_true_positives', 'fairness_false_alarms',
    'fairness_far', 'fairness_mdr', 'fairness_mtd', 'fairness_mtfa', 'fairness_mtr',
]


def _summary_metric_names(seed_runs):
    """The fixed summary metrics plus every per-phase key the rows carry.

    Phase names differ per dataset (Adult has five, COMPAS three, Folktables
    one per test year), so they are collected rather than listed.
    """
    phase_keys = sorted({
        key
        for run in seed_runs
        for row in run.get('results') or []
        if isinstance(row, dict)
        for key in row
        if key.startswith('phase_')
    })
    return _SUMMARY_METRICS + phase_keys


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
    # Saved with the results as well as sent to W&B, so offline tools (the
    # lambda trade-off plot) can tell which runs share a configuration.
    run_config = {
        'dataset': dataset_name,
        'models': models_to_run,
        'batch_size': int(FLAGS.batch_size),
        'depth': int(FLAGS.depth),
        'num_trees': int(FLAGS.num_trees),
        'lambda_const': float(FLAGS.lambda_const),
        'drift_scenario': FLAGS.drift_scenario,
        'drift_adwin_delta_warn': float(FLAGS.drift_adwin_delta_warn),
        'drift_adwin_delta_confirm': float(FLAGS.drift_adwin_delta_confirm),
        'drift_lr_prewarm_mult': float(FLAGS.drift_lr_prewarm_mult),
        'drift_lr_spike_mult': float(FLAGS.drift_lr_spike_mult),
        'drift_lr_decay_steps': int(FLAGS.drift_lr_decay_steps),
        'drift_cooldown': int(FLAGS.drift_cooldown),
        'drift_min_samples_per_stream': int(FLAGS.drift_min_samples_per_stream),
        'drift_temperature_on_drift': float(FLAGS.drift_temperature_on_drift),
        'drift_temperature_recovery_target': float(FLAGS.drift_temperature_recovery_target),
        'drift_temperature_recovery_step': float(FLAGS.drift_temperature_recovery_step),
        'seeds': seeds,
        'controller_components': str(FLAGS.controller_components),
        # What the arms actually ran with, after dataset defaults.
        'static_params': _build_aranyani_static_params(),
    }
    wb_run = None
    if bool(FLAGS.wandb_log):
        if wandb is None:
            raise ImportError("wandb logging requested but wandb is not installed.")
        init_kwargs = {
            'project': str(FLAGS.wandb_project),
            'config': run_config,
            'reinit': True,
        }
        if str(FLAGS.wandb_entity).strip():
            init_kwargs['entity'] = str(FLAGS.wandb_entity).strip()
        wb_run = wandb.init(**init_kwargs)
        run_config['wandb_run_id'] = str(wb_run.id)
        run_config['wandb_sweep_id'] = getattr(wb_run, 'sweep_id', None)

    print(f"\n{'=' * 80}")
    print(f" Multi-seed pipeline: {len(seeds)} runs")
    print(f" Models: {models_to_run}")
    print(f" Dataset: {dataset_name}")
    print(f" Seeds: {seeds}")
    print(f" Base output: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'=' * 80}\n")

    seed_runs = []
    wandb_timestep_rows = []
    base_output_dir = os.path.join(OUTPUT_DIR, f'dataset_{dataset_name}')
    if wb_run is not None:
        # Sweep agents run in parallel; a shared directory would let every run
        # overwrite the others' results.json files.
        base_output_dir = os.path.join(
            OUTPUT_DIR, 'wandb', str(wb_run.id), f'dataset_{dataset_name}')
    for idx, seed in enumerate(seeds, 1):
        print(f"\n{'-' * 80}")
        print(f" Seed run [{idx}/{len(seeds)}]: {seed}")
        print(f"{'-' * 80}")
        for model_name in models_to_run:
            print(f"\n>>> Model run: {model_name}")
            seed_output_dir = os.path.join(base_output_dir, f'model_{model_name}', f'seed_{seed}')
            results = run_scenarios(
                model_name=model_name,
                dataset_name=dataset_name,
                output_dir=seed_output_dir,
                scenario_filter=FLAGS.drift_scenario if FLAGS.drift_scenario else None,
                seed=seed,
            )
            if wb_run is not None:
                for row in results:
                    tm = row.get('test_metrics') or {}
                    ts = row.get('timestep_results') or {}
                    stream_acc = _mean_stream_metric(ts.get('accuracy'))
                    stream_dp = _mean_stream_metric(ts.get('dp'))
                    stream_eo = _mean_stream_metric(ts.get('eo'))
                    wb_log = {
                        'seed': int(seed),
                        'model': str(model_name),
                        'scenario': str(row.get('scenario')),
                        'accuracy': _numeric_or_nan(tm.get('accuracy')),
                        'dp': _numeric_or_nan(tm.get('dp')),
                        'eo': _numeric_or_nan(tm.get('eo')),
                        'stream_accuracy': float(stream_acc) if stream_acc is not None else np.nan,
                        'stream_dp': float(stream_dp) if stream_dp is not None else np.nan,
                        'stream_eo': float(stream_eo) if stream_eo is not None else np.nan,
                    }
                    static_used = ts.get('static_params_used')
                    if isinstance(static_used, dict):
                        wb_log.update({f'static_{k}': v for k, v in static_used.items()})
                    wandb.log(wb_log)

                    acc_values = ts.get('accuracy')
                    dp_values = ts.get('dp')
                    if isinstance(acc_values, np.ndarray):
                        acc_values = acc_values.reshape(-1).tolist()
                    if isinstance(dp_values, np.ndarray):
                        dp_values = dp_values.reshape(-1).tolist()
                    if isinstance(acc_values, (list, tuple)) and isinstance(dp_values, (list, tuple)):
                        for timestep, (acc_value, dp_value) in enumerate(
                            zip(acc_values, dp_values), start=1
                        ):
                            if isinstance(acc_value, numbers.Number) and isinstance(dp_value, numbers.Number):
                                wandb_timestep_rows.append([
                                    int(seed),
                                    str(model_name),
                                    str(row.get('scenario')),
                                    int(timestep),
                                    float(acc_value),
                                    float(dp_value),
                                ])
            seed_runs.append({'seed': seed, 'model': model_name, 'results': results})

    if FLAGS.run_all_scenarios:
        metric_names = _summary_metric_names(seed_runs)
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
        metric_names = _summary_metric_names(seed_runs)
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
        'config': run_config,
        'seed_runs': seed_runs,
        'summary': summary,
    }

    os.makedirs(base_output_dir, exist_ok=True)
    output_path = os.path.join(base_output_dir, 'seed_pipeline_results.json')
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSeed pipeline results saved to: {output_path}")
    if wb_run is not None:
        if wandb_timestep_rows:
            wandb.log({
                'accuracy_dp_over_time': wandb.Table(
                    data=wandb_timestep_rows,
                    columns=['seed', 'model', 'scenario', 'timestep', 'accuracy', 'dp'],
                )
            })
        wandb.summary['results_file'] = output_path
        _log_sweep_summary(summary, models_to_run)
        if summary['mode'] == 'single':
            for model_name, metrics in summary['metrics_by_model'].items():
                for metric_name, metric_stats in metrics.items():
                    if metric_stats['mean'] is not None:
                        wandb.summary[f'{model_name}_{metric_name}_mean'] = metric_stats['mean']
                    if metric_stats['std'] is not None:
                        wandb.summary[f'{model_name}_{metric_name}_std'] = metric_stats['std']
        wandb.finish()

    print(f"\n{'=' * 80}")
    print(' Seed Pipeline Summary')
    print(f"{'=' * 80}")
    if summary['mode'] == 'all_scenarios':
        for model_row in summary['rows']:
            print(f"\nModel: {model_row['model']}")
            print(f"{'Scenario':<36s} {'Acc':>28s} {'DP':>28s} {'EO':>28s}")
            print('-' * 120)
            for row in model_row['rows']:
                print(
                    f"{row['scenario']:<36s} "
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
