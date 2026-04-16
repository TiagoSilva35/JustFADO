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
# Scenario: female income parity shift
# ---------------------------------------------------------------------------
def female_income_parity(df):
    """Abruptly equalise the income distribution between genders.

    In the original data, ~30% of Males earn >50K vs ~11% of Females.
    This drift promotes Female samples: during phase 1 all Female <=50K
    samples are flipped to >50K (simulating a policy change / pay-gap
    closure).  This creates a real shift in P(Y|A=Female) that the model
    was never trained on, directly stressing the fairness-accuracy trade-off.

    Phase 1 (abrupt): all Female <=50K → >50K.
    Phase 2 (gradual): reversion — flip probability decays 70% → 0%.
    """
    random.seed(SEED)
    columns = df.columns
    X = df.values.tolist()
    warmup, phase1, recovery1, phase2, recovery2 = _split_phases(X)

    GENDER_IDX = 9
    INCOME_IDX = -1

    def _promote(sample):
        income = str(sample[INCOME_IDX]).strip().rstrip('.')
        if income == '<=50K':
            sample[INCOME_IDX] = ' >50K.'

    def _demote(sample):
        income = str(sample[INCOME_IDX]).strip().rstrip('.')
        if income == '>50K':
            sample[INCOME_IDX] = ' <=50K.'

    for sample in phase1:
        if str(sample[GENDER_IDX]).strip() == 'Female':
            _promote(sample)

    for i, sample in enumerate(phase2):
        prob = 0.70 * (1.0 - i / max(len(phase2) - 1, 1))
        if str(sample[GENDER_IDX]).strip() == 'Female' and random.random() < prob:
            _demote(sample)

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
# Registry
# ---------------------------------------------------------------------------
SCENARIOS = {
    'no_drift': no_drift,
    'abrupt_gender': abrupt_gender,
    'gradual_gender': gradual_gender,
    'label_flip': label_flip,
    'gender_relationship_decouple': gender_relationship_decouple,
    'female_income_parity': female_income_parity,
    'occupation_gender_reversal': occupation_gender_reversal,
}

SCENARIO_DESCRIPTIONS = {
    'no_drift': 'Baseline (no drift)',
    'abrupt_gender': 'Abrupt + slow gender swap for high-income males',
    'gradual_gender': 'Gradually increasing gender swap',
    'label_flip': 'Label flip in drift phases (40% abrupt, 30%→5% gradual)',
    'feature_noise': 'Gaussian noise on numerical features',
    'compound': 'Gender swap + label flip combined',
    'gender_relationship_decouple': 'Break Husband/Wife ↔ Male/Female co-occurrence',
    'female_income_parity': 'Abrupt female income promotion (P(Y|Female) equalised)',
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
