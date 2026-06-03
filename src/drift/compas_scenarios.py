"""Virtual drift scenarios for the COMPAS test stream.

These scenarios are the COMPAS analog of the Adult scenarios in
``src/drift/scenarios.py``. They are applied to the *raw, post-null-filter*
DataFrame (string-valued categoricals such as ``race``, ``age_cat``,
``c_charge_degree``, ``sex`` and numeric prior-count columns) BEFORE the
one-hot / frequency encoder is applied, so a deterministic row-level edit
propagates through the downstream transformer as a true feature-level
distribution shift.

Phase convention -- identical proportions to Adult so that ADWIN
detection thresholds tuned for Adult behave comparably:

    [0 .. 0.20)        warmup
    [0.20 .. 0.50)     drift phase 1 (abrupt component)
    [0.50 .. 0.60)     recovery 1
    [0.60 .. 0.75)     drift phase 2 (gradual / slow component)
    [0.75 .. 1.0]      recovery 2

Adjustment vs Adult: warmup is bumped from 15% to 20% because the COMPAS
test stream is roughly an order of magnitude smaller (1.4k vs 16k
samples). This gives the warm-started forest a few more samples to
stabilise before the first drift phase fires while keeping the remaining
phases proportionally close to the Adult layout.

Scenario design:

* ``no_drift`` -- baseline, returns the test slice unmodified.
* ``abrupt_race`` -- direct analog of Adult ``abrupt_gender``: every
  recidivist defendant whose race is African-American is relabeled
  Caucasian during phase 1, then the same swap fires with an
  exponentially decaying temperature during phase 2.
* ``gradual_race`` -- analog of Adult ``gradual_gender``: linearly
  increasing race-swap probability across both drift phases (0->0.8,
  0->0.6).
* ``age_race_decouple`` -- analog of Adult
  ``gender_relationship_decouple``: breaks the ``age_cat`` <-> ``race``
  co-occurrence the model relies on by swapping ``Less than 25`` and
  ``Greater than 45`` between African-American and Caucasian
  defendants.
* ``charge_degree_race_swap`` -- analog of Adult
  ``occupation_gender_reversal``: swaps race in stereotyped
  charge-degree combinations (African-American + Felony -> Caucasian,
  Caucasian + Misdemeanor -> African-American).

All scenarios use a fixed PRNG seed (``SEED = 42``) so they are
reproducible across runs given identical input row order.
"""

import random

import pandas as pd

SEED = 42
SPLITS = [0.20, 0.50, 0.60, 0.75, 1.0]
PHASE_LABELS = ['Warmup', 'Drift Phase 1', 'Recovery 1', 'Drift Phase 2', 'Recovery 2']


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------
def _split_phases(df):
  """Split a raw DataFrame into the 5 canonical drift phases."""
  n = len(df)
  b = [int(s * n) for s in SPLITS]
  return (
      df.iloc[: b[0]].copy(),
      df.iloc[b[0]: b[1]].copy(),
      df.iloc[b[1]: b[2]].copy(),
      df.iloc[b[2]: b[3]].copy(),
      df.iloc[b[3]:].copy(),
  )


def _concat(parts):
  return pd.concat(parts, axis=0).reset_index(drop=True)


def _is_recid_positive(value):
  """Treat 1 / '1' / 'yes' / 'true' as positive recidivism labels."""
  try:
    return int(value) == 1
  except (TypeError, ValueError):
    return str(value).strip().lower() in {'yes', 'true', '1'}


# ---------------------------------------------------------------------------
# Scenario: no drift (baseline)
# ---------------------------------------------------------------------------
def no_drift(df, target_col='two_year_recid'):
  """Return the dataset unmodified (baseline)."""
  return df.copy()


# ---------------------------------------------------------------------------
# Scenario: abrupt race drift
# ---------------------------------------------------------------------------
def abrupt_race(df, target_col='two_year_recid'):
  """Direct COMPAS analog of Adult ``abrupt_gender``.

  Phase 1 (abrupt): every recidivist defendant whose race is
  African-American is relabeled Caucasian. This breaks the
  ``race = African-American => recid = 1`` association the model
  learns on the no-drift training partition.

  Phase 2 (slow): the same swap fires with an exponentially decaying
  temperature, mirroring the slow phase of ``abrupt_gender``.
  """
  random.seed(SEED)
  warmup, phase1, recovery1, phase2, recovery2 = _split_phases(df)

  if target_col in phase1.columns:
    mask = (
        phase1[target_col].apply(_is_recid_positive)
        & (phase1['race'].astype(str).str.strip() == 'African-American')
    )
    phase1.loc[mask, 'race'] = 'Caucasian'

  temperature = 0.999
  for idx in phase2.index:
    if (
        _is_recid_positive(phase2.at[idx, target_col])
        and str(phase2.at[idx, 'race']).strip() == 'African-American'
    ):
      if random.random() < temperature:
        phase2.at[idx, 'race'] = 'Caucasian'
      temperature *= 0.999

  return _concat([warmup, phase1, recovery1, phase2, recovery2])


# ---------------------------------------------------------------------------
# Scenario: gradual race drift
# ---------------------------------------------------------------------------
def gradual_race(df, target_col='two_year_recid'):
  """Linearly increasing race swap across both drift phases.

  Counterpart of Adult ``gradual_gender``: probability of relabeling an
  African-American recidivist as Caucasian ramps 0->0.8 across phase 1
  and 0->0.6 across phase 2.
  """
  random.seed(SEED)
  warmup, phase1, recovery1, phase2, recovery2 = _split_phases(df)

  n1 = max(len(phase1) - 1, 1)
  for i, idx in enumerate(phase1.index):
    prob = 0.8 * (i / n1)
    if (
        _is_recid_positive(phase1.at[idx, target_col])
        and str(phase1.at[idx, 'race']).strip() == 'African-American'
        and random.random() < prob
    ):
      phase1.at[idx, 'race'] = 'Caucasian'

  n2 = max(len(phase2) - 1, 1)
  for i, idx in enumerate(phase2.index):
    prob = 0.6 * (i / n2)
    if (
        _is_recid_positive(phase2.at[idx, target_col])
        and str(phase2.at[idx, 'race']).strip() == 'African-American'
        and random.random() < prob
    ):
      phase2.at[idx, 'race'] = 'Caucasian'

  return _concat([warmup, phase1, recovery1, phase2, recovery2])


# ---------------------------------------------------------------------------
# Scenario: age <-> race decoupling
# ---------------------------------------------------------------------------
def age_race_decouple(df, target_col='two_year_recid'):
  """Break the age_cat <-> race co-occurrence used by the model.

  On COMPAS, ``Less than 25`` is over-represented among African-American
  defendants while ``Greater than 45`` is over-represented among
  Caucasians. We swap ``age_cat`` between the two race groups so that
  the joint distribution ``P(age_cat, race)`` is inverted relative to
  what the model learned, without modifying recidivism labels. This is
  the COMPAS counterpart of Adult ``gender_relationship_decouple``.

  Phase 1 (abrupt): 100% of qualifying samples swapped.
  Phase 2 (slow): probability decays 0.80 -> 0.10.
  """
  random.seed(SEED)
  warmup, phase1, recovery1, phase2, recovery2 = _split_phases(df)

  def _swap(df_phase, prob_fn):
    n = max(len(df_phase) - 1, 1)
    for i, idx in enumerate(df_phase.index):
      prob = prob_fn(i, n)
      if random.random() >= prob:
        continue
      race = str(df_phase.at[idx, 'race']).strip()
      age = str(df_phase.at[idx, 'age_cat']).strip()
      if race == 'African-American' and age == 'Less than 25':
        df_phase.at[idx, 'age_cat'] = 'Greater than 45'
      elif race == 'Caucasian' and age == 'Greater than 45':
        df_phase.at[idx, 'age_cat'] = 'Less than 25'

  _swap(phase1, prob_fn=lambda i, n: 1.0)
  _swap(phase2, prob_fn=lambda i, n: 0.80 - 0.70 * (i / n))

  return _concat([warmup, phase1, recovery1, phase2, recovery2])


# ---------------------------------------------------------------------------
# Scenario: charge-degree race reversal
# ---------------------------------------------------------------------------
def charge_degree_race_swap(df, target_col='two_year_recid'):
  """Swap race in stereotyped charge-degree combinations.

  COMPAS counterpart of Adult ``occupation_gender_reversal``.
  African-American defendants whose lead charge is a Felony
  (``c_charge_degree = 'F'``) are relabeled Caucasian, and Caucasian
  defendants whose lead charge is a Misdemeanor
  (``c_charge_degree = 'M'``) are relabeled African-American. This
  inverts the charge-severity <-> race co-occurrence the model relies on
  while leaving prior-count features untouched.

  Phase 1 (abrupt): 100% swap among stereotyped samples.
  Phase 2 (slow): 50% -> 10% decaying swap.
  """
  random.seed(SEED)
  warmup, phase1, recovery1, phase2, recovery2 = _split_phases(df)

  def _swap(df_phase, base_prob, end_prob):
    n = max(len(df_phase) - 1, 1)
    for i, idx in enumerate(df_phase.index):
      prob = base_prob - (base_prob - end_prob) * (i / n)
      if random.random() >= prob:
        continue
      race = str(df_phase.at[idx, 'race']).strip()
      degree = str(df_phase.at[idx, 'c_charge_degree']).strip()
      if race == 'African-American' and degree == 'F':
        df_phase.at[idx, 'race'] = 'Caucasian'
      elif race == 'Caucasian' and degree == 'M':
        df_phase.at[idx, 'race'] = 'African-American'

  _swap(phase1, base_prob=1.0, end_prob=1.0)
  _swap(phase2, base_prob=0.50, end_prob=0.10)

  return _concat([warmup, phase1, recovery1, phase2, recovery2])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
COMPAS_SCENARIOS = {
    'no_drift': no_drift,
    'abrupt_race': abrupt_race,
    'gradual_race': gradual_race,
    'age_race_decouple': age_race_decouple,
    'charge_degree_race_swap': charge_degree_race_swap,
}

COMPAS_SCENARIO_DESCRIPTIONS = {
    'no_drift': 'Baseline (no drift)',
    'abrupt_race': 'Abrupt + slow race swap for African-American recidivists',
    'gradual_race': 'Gradually increasing race swap for recidivist defendants',
    'age_race_decouple': 'Break age_cat<->race co-occurrence (AA<25 / Cauc>45)',
    'charge_degree_race_swap': 'Swap race in felonies (AA->Cauc) and misdemeanors (Cauc->AA)',
}


def get_compas_scenario(name):
  """Return the drift function for the given COMPAS scenario name."""
  if name not in COMPAS_SCENARIOS:
    raise ValueError(
        f"Unknown COMPAS drift scenario '{name}'. "
        f"Available: {list(COMPAS_SCENARIOS.keys())}"
    )
  return COMPAS_SCENARIOS[name]
