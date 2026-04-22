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
# Scenario: gender–relationship decoupling
# ---------------------------------------------------------------------------
def gender_relationship_decouple(df):
    """Break the learned correlation between sex and relationship role.

    In the training data, 'Husband' is almost exclusively Male and 'Wife' is
    almost exclusively Female.  The model (and its fairness penalty) relies on
    this co-occurrence.  During drift phases we swap relationship roles between
    genders so that Male samples appear as 'Wife' and Female samples appear as
    'Husband'.  This invalidates the model's internal representation of sex
    without changing the income label, so accuracy drops are purely due to
    the broken feature correlation.

    Phase 1 (abrupt): 100% of qualifying samples are swapped.
    Phase 2 (gradual): swap probability decays 80% → 10%.
    """
    random.seed(SEED)
    columns = df.columns
    X = df.values.tolist()
    warmup, phase1, recovery1, phase2, recovery2 = _split_phases(X)

    # col indices (0-based): gender=9, relationship=7
    GENDER_IDX = 9
    REL_IDX = 7

    def _swap_rel(sample):
        gender = str(sample[GENDER_IDX]).strip()
        rel = str(sample[REL_IDX]).strip()
        if gender == 'Male' and rel == 'Husband':
            sample[REL_IDX] = ' Wife'
        elif gender == 'Female' and rel == 'Wife':
            sample[REL_IDX] = ' Husband'

    # Phase 1 – abrupt: swap all qualifying samples
    for sample in phase1:
        _swap_rel(sample)

    # Phase 2 – gradual: decaying probability
    for i, sample in enumerate(phase2):
        prob = 0.80 - 0.70 * (i / max(len(phase2) - 1, 1))
        if random.random() < prob:
            _swap_rel(sample)

    return pd.DataFrame(warmup + phase1 + recovery1 + phase2 + recovery2, columns=columns)



# ---------------------------------------------------------------------------
# Scenario: occupation gender reversal
# ---------------------------------------------------------------------------
def occupation_gender_reversal(df):
    """Swap gender in occupations that are strongly gender-stereotyped.

    The model learns that certain occupations (Exec-managerial, Craft-repair,
    Transport-moving) are Male-dominated and others (Adm-clerical,
    Priv-house-serv, Other-service) are Female-dominated.  This drift abruptly
    swaps the gender label for samples in these stereotyped occupations,
    breaking the occupation–gender co-occurrence the model relies on.

    Unlike abrupt_gender (which only touches high-income males), this hits
    *all* income levels and a wider range of occupations, causing a larger and
    more uniform accuracy drop.

    Phase 1 (abrupt): 100% swap in stereotyped occupations.
    Phase 2 (slow):   50% swap, decaying to 10%.
    """
    random.seed(SEED)
    columns = df.columns
    X = df.values.tolist()
    warmup, phase1, recovery1, phase2, recovery2 = _split_phases(X)

    # col indices
    GENDER_IDX = 9
    OCC_IDX    = 6

    # Occupations where gender is strongly stereotyped in this dataset
    MALE_DOMINATED   = {'Exec-managerial', 'Craft-repair', 'Transport-moving',
                        'Farming-fishing', 'Protective-serv'}
    FEMALE_DOMINATED = {'Adm-clerical', 'Other-service', 'Priv-house-serv'}

    def _swap_gender(sample):
        gender = str(sample[GENDER_IDX]).strip()
        occ    = str(sample[OCC_IDX]).strip()
        if gender == 'Male' and occ in MALE_DOMINATED:
            sample[GENDER_IDX] = ' Female'
        elif gender == 'Female' and occ in FEMALE_DOMINATED:
            sample[GENDER_IDX] = ' Male'

    # Phase 1 – abrupt: swap all stereotyped samples
    for sample in phase1:
        _swap_gender(sample)

    # Phase 2 – gradual: decaying probability
    for i, sample in enumerate(phase2):
        prob = 0.50 - 0.40 * (i / max(len(phase2) - 1, 1))
        if random.random() < prob:
            _swap_gender(sample)

    return pd.DataFrame(warmup + phase1 + recovery1 + phase2 + recovery2, columns=columns)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SCENARIOS = {
    'no_drift': no_drift,
    'abrupt_gender': abrupt_gender,
    'gradual_gender': gradual_gender,
    'gender_relationship_decouple': gender_relationship_decouple,
    'occupation_gender_reversal': occupation_gender_reversal,
}

SCENARIO_DESCRIPTIONS = {
    'no_drift': 'Baseline (no drift)',
    'abrupt_gender': 'Abrupt + slow gender swap for high-income males',
    'gradual_gender': 'Gradually increasing gender swap',
    'gender_relationship_decouple': 'Break Husband/Wife ↔ Male/Female co-occurrence',
    'occupation_gender_reversal': 'Swap gender in strongly stereotyped occupations',
}


def get_scenario(name):
    """Return the drift function for the given scenario name."""
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown drift scenario '{name}'. "
            f"Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]
