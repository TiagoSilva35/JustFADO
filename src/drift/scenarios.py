"""Drift scenario definitions for automated experimentation.

Each scenario function takes a raw DataFrame (Adult test set format) and returns
a drifted DataFrame.  The phase splits follow the same convention used in
``create_drifted_ds.py``:

    [0 .. 0.15)  warmup
    [0.15 .. 0.50)  first drift phase
    [0.50 .. 0.60)  recovery 1
    [0.60 .. 0.75)  second drift phase
    [0.75 .. 1.0]   recovery 2
"""

import copy
import random
import numpy as np
import pandas as pd

SEED = 42
SPLITS = [0.15, 0.50, 0.60, 0.75, 1.0]

PHASE_LABELS = ['Warmup', 'Drift Phase 1', 'Recovery 1', 'Drift Phase 2', 'Recovery 2']


def _split_phases(X):
    """Split a list of samples into the 5 canonical phases."""
    n = len(X)
    boundaries = [int(s * n) for s in SPLITS]
    warmup = X[:boundaries[0]]
    phase1 = X[boundaries[0]:boundaries[1]]
    recovery1 = X[boundaries[1]:boundaries[2]]
    phase2 = X[boundaries[2]:boundaries[3]]
    recovery2 = X[boundaries[3]:]
    return warmup, phase1, recovery1, phase2, recovery2


# ---------------------------------------------------------------------------
# Scenario: no drift (baseline)
# ---------------------------------------------------------------------------
def no_drift(df):
    """Return the dataset unmodified (baseline)."""
    return df.copy()


# ---------------------------------------------------------------------------
# Scenario: abrupt gender drift
# ---------------------------------------------------------------------------
def abrupt_gender(df):
    """Abrupt + slow gender swap for high-income males (original create_drifted_ds)."""
    random.seed(SEED)
    columns = df.columns
    X = df.values.tolist()
    warmup, phase1, recovery1, phase2, recovery2 = _split_phases(X)

    # Abrupt drift – swap ALL qualifying males in phase 1
    for sample in phase1:
        income = str(sample[-1]).strip().rstrip('.')
        gender = str(sample[9]).strip()
        if income == '>50K' and gender == 'Male':
            sample[9] = ' Female'

    # Slow drift – decaying probability swap in phase 2
    temperature = 0.999
    for sample in phase2:
        income = str(sample[-1]).strip().rstrip('.')
        gender = str(sample[9]).strip()
        if income == '>50K' and gender == 'Male':
            if random.random() < temperature:
                sample[9] = ' Female'
            temperature *= 0.999

    return pd.DataFrame(warmup + phase1 + recovery1 + phase2 + recovery2, columns=columns)


# ---------------------------------------------------------------------------
# Scenario: gradual gender drift only
# ---------------------------------------------------------------------------
def gradual_gender(df):
    """Gradually increasing gender swap across both drift phases."""
    random.seed(SEED)
    columns = df.columns
    X = df.values.tolist()
    warmup, phase1, recovery1, phase2, recovery2 = _split_phases(X)

    # Phase 1: linearly increasing probability 0→0.8
    for i, sample in enumerate(phase1):
        prob = 0.8 * (i / max(len(phase1) - 1, 1))
        income = str(sample[-1]).strip().rstrip('.')
        gender = str(sample[9]).strip()
        if income == '>50K' and gender == 'Male' and random.random() < prob:
            sample[9] = ' Female'

    # Phase 2: linearly increasing probability 0→0.6
    for i, sample in enumerate(phase2):
        prob = 0.6 * (i / max(len(phase2) - 1, 1))
        income = str(sample[-1]).strip().rstrip('.')
        gender = str(sample[9]).strip()
        if income == '>50K' and gender == 'Male' and random.random() < prob:
            sample[9] = ' Female'

    return pd.DataFrame(warmup + phase1 + recovery1 + phase2 + recovery2, columns=columns)


# ---------------------------------------------------------------------------
# Scenario: label flip drift
# ---------------------------------------------------------------------------
def label_flip(df):
    """Flip income labels for a subset of samples in drift phases.

    In phase 1 (abrupt), 40% of labels are flipped.
    In phase 2 (gradual), flip probability decays from 30% to 5%.
    """
    random.seed(SEED)
    columns = df.columns
    X = df.values.tolist()
    warmup, phase1, recovery1, phase2, recovery2 = _split_phases(X)

    def _flip_income(sample):
        income = str(sample[-1]).strip().rstrip('.')
        if income == '>50K':
            sample[-1] = ' <=50K.'
        else:
            sample[-1] = ' >50K.'

    # Phase 1 – abrupt label flip (40%)
    for sample in phase1:
        if random.random() < 0.40:
            _flip_income(sample)

    # Phase 2 – gradual label flip (30% → 5%)
    for i, sample in enumerate(phase2):
        prob = 0.30 - 0.25 * (i / max(len(phase2) - 1, 1))
        if random.random() < prob:
            _flip_income(sample)

    return pd.DataFrame(warmup + phase1 + recovery1 + phase2 + recovery2, columns=columns)


# ---------------------------------------------------------------------------
# Scenario: feature noise drift
# ---------------------------------------------------------------------------
def feature_noise(df):
    """Add Gaussian noise to numerical features during drift phases.

    Affected columns: age (0), fnlwgt (2), education-num (4),
    capital gain (10), capital loss (11), hours per week (12).
    """
    random.seed(SEED)
    np.random.seed(SEED)
    columns = df.columns
    X = df.values.tolist()
    warmup, phase1, recovery1, phase2, recovery2 = _split_phases(X)

    numeric_idxs = [0, 2, 4, 10, 11, 12]

    def _add_noise(samples, scale):
        for sample in samples:
            for idx in numeric_idxs:
                try:
                    val = float(sample[idx])
                    sample[idx] = val + np.random.normal(0, scale * abs(val) + 1e-6)
                except (ValueError, TypeError):
                    pass

    # Phase 1 – strong noise
    _add_noise(phase1, scale=0.5)
    # Phase 2 – moderate noise
    _add_noise(phase2, scale=0.25)

    return pd.DataFrame(warmup + phase1 + recovery1 + phase2 + recovery2, columns=columns)


# ---------------------------------------------------------------------------
# Scenario: compound drift (gender + label noise)
# ---------------------------------------------------------------------------
def compound(df):
    """Combine abrupt gender drift with moderate label flipping."""
    random.seed(SEED)
    columns = df.columns
    X = df.values.tolist()
    warmup, phase1, recovery1, phase2, recovery2 = _split_phases(X)

    def _flip_income(sample):
        income = str(sample[-1]).strip().rstrip('.')
        if income == '>50K':
            sample[-1] = ' <=50K.'
        else:
            sample[-1] = ' >50K.'

    # Phase 1 – abrupt gender swap + 20% label flip
    for sample in phase1:
        income = str(sample[-1]).strip().rstrip('.')
        gender = str(sample[9]).strip()
        if income == '>50K' and gender == 'Male':
            sample[9] = ' Female'
        if random.random() < 0.20:
            _flip_income(sample)

    # Phase 2 – slow gender swap + 10% label flip
    temperature = 0.999
    for sample in phase2:
        income = str(sample[-1]).strip().rstrip('.')
        gender = str(sample[9]).strip()
        if income == '>50K' and gender == 'Male':
            if random.random() < temperature:
                sample[9] = ' Female'
            temperature *= 0.999
        if random.random() < 0.10:
            _flip_income(sample)

    return pd.DataFrame(warmup + phase1 + recovery1 + phase2 + recovery2, columns=columns)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SCENARIOS = {
    'no_drift': no_drift,
    'abrupt_gender': abrupt_gender,
    'gradual_gender': gradual_gender,
    'label_flip': label_flip,
    'feature_noise': feature_noise,
    'compound': compound,
}

SCENARIO_DESCRIPTIONS = {
    'no_drift': 'Baseline (no drift)',
    'abrupt_gender': 'Abrupt + slow gender swap for high-income males',
    'gradual_gender': 'Gradually increasing gender swap',
    'label_flip': 'Label flip in drift phases (40% abrupt, 30%→5% gradual)',
    'feature_noise': 'Gaussian noise on numerical features',
    'compound': 'Gender swap + label flip combined',
}


def get_scenario(name):
    """Return the drift function for the given scenario name."""
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown drift scenario '{name}'. "
            f"Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]
