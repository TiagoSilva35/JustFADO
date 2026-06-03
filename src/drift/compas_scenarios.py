"""Virtual drift scenarios for the COMPAS test stream.

Layout (3 phases, ``test_size=0.30``, test stream ~2.1k rows):

    [0 .. 0.20)        warmup           (~433 samples)
    [0.20 .. 0.60)     drift            (~866 samples)
    [0.60 .. 1.0]      recovery         (~866 samples)

Each scenario applies its perturbation only during the drift phase;
the recovery phase reverts to undrifted rows so the controller has a
clean window to demonstrate adaptation back to baseline behaviour.

Why each scenario carries both a feature edit AND a label-flip
component: the Aranyani forest on COMPAS leans heavily on the numeric
``priors_count`` / ``age`` features, so the original race / age_cat /
charge_degree edits perturbed only weakly-used categoricals and left
``P(y | x)`` (and therefore accuracy) almost unchanged. With accuracy
flat ADWIN never trips, defeating the whole point of FADO on this
dataset. To make the drifts strong enough to be detectable we follow
each row-level feature edit with a deterministic (probability 1.0)
flip of ``two_year_recid`` on the SAME edited rows -- a per-subgroup concept
drift that gives ADWIN a clear accuracy signal. Each scenario's
fairness narrative is preserved (race / age_cat / charge_degree still
shift, sensitive-attribute distribution still moves) while the
detectable signal is added. ``no_drift`` is left untouched as the true
baseline.

Scenarios operate on the raw, post-null-filter DataFrame BEFORE the
one-hot / frequency encoder is applied, so row-level edits propagate
downstream as true feature-level distribution shifts. The recomputed
``y_test`` in ``read_compas_train_test`` picks up the label flips.

Scenario set:

* ``no_drift`` -- baseline, identity.
* ``abrupt_race`` -- analog of Adult ``abrupt_gender``: AA recidivists
  in the drift phase are relabeled Caucasian + label-flipped at 100%.
* ``gradual_race`` -- analog of Adult ``gradual_gender``: linearly
  ramping race-swap probability 0->1 across the drift phase, with 100%
  label flip on the rows that got swapped.
* ``age_race_decouple`` -- analog of Adult
  ``gender_relationship_decouple``: invert age_cat<->race
  co-occurrence (AA<25 / Cauc>45) with 100% label flip on the swapped
  rows.
* ``charge_degree_race_swap`` -- analog of Adult
  ``occupation_gender_reversal``: AA+Felony -> Caucasian and
  Caucasian+Misdemeanor -> AA, with 100% label flip on the swapped
  rows.

PRNG seed (``SEED = 42``) is fixed so the random swap and label-flip
decisions are reproducible across runs given identical input row order.
"""

import random

import pandas as pd

SEED = 42
SPLITS = [0.20, 0.60, 1.0]
PHASE_LABELS = ['Warmup', 'Drift', 'Recovery']

# Probability of flipping ``two_year_recid`` on rows whose feature
# columns were edited by a scenario. Set to 1.0 because COMPAS's test
# stream is small (~2.1k) and FADO's prequential test-then-train
# protocol lets the model partially re-fit to mislabeled rows within
# the drift phase, dampening the spike ADWIN sees. A deterministic
# 100% flip on the affected subgroup keeps the spike high enough for
# the (slightly relaxed) COMPAS ADWIN params to trigger -- see
# ``_COMPAS_FADO_OVERRIDES`` in ``src/main.py``.
DEFAULT_LABEL_FLIP_PROB = 1.0


# ---------------------------------------------------------------------------
# Phase / row helpers
# ---------------------------------------------------------------------------
def _split_phases(df):
  """Split a raw DataFrame into warmup / drift / recovery slices."""
  n = len(df)
  b = [int(s * n) for s in SPLITS]
  return (
      df.iloc[: b[0]].copy(),
      df.iloc[b[0]: b[1]].copy(),
      df.iloc[b[1]:].copy(),
  )


def _concat(parts):
  return pd.concat(parts, axis=0).reset_index(drop=True)


def _is_recid_positive(value):
  """Treat 1 / '1' / 'yes' / 'true' as positive recidivism labels."""
  try:
    return int(value) == 1
  except (TypeError, ValueError):
    return str(value).strip().lower() in {'yes', 'true', '1'}


def _flip_recid(value):
  """Flip a recidivism label (0<->1 for ints; yes<->no for strings)."""
  try:
    return 1 - int(value)
  except (TypeError, ValueError):
    pass
  s = str(value).strip().lower()
  if s in {'yes', 'true', '1'}:
    return 'No'
  if s in {'no', 'false', '0'}:
    return 'Yes'
  return value


def _apply_label_flips(df_phase, edited_indices, target_col, prob, rng):
  """Bernoulli(prob) flip ``target_col`` for each row in ``edited_indices``."""
  if not edited_indices or target_col not in df_phase.columns:
    return
  for idx in edited_indices:
    if rng.random() < prob:
      df_phase.at[idx, target_col] = _flip_recid(df_phase.at[idx, target_col])


# ---------------------------------------------------------------------------
# Scenario: no drift (baseline)
# ---------------------------------------------------------------------------
def no_drift(df, target_col='two_year_recid'):
  """Return the dataset unmodified (baseline)."""
  return df.copy()


# ---------------------------------------------------------------------------
# Scenario: abrupt race + concept drift
# ---------------------------------------------------------------------------
def abrupt_race(df, target_col='two_year_recid'):
  """Sustained abrupt race swap + 100% label flip on edited rows.

  Phase 1 (drift): every recidivist defendant whose race is
  African-American is relabeled Caucasian and has their
  ``two_year_recid`` label flipped (deterministic 100%). The race swap breaks
  ``race=AA -> recid=1`` and the label flip injects per-subgroup concept
  drift so ADWIN sees a clear accuracy hit. Recovery reverts to
  undrifted rows.
  """
  rng = random.Random(SEED)
  warmup, drift, recovery = _split_phases(df)
  if target_col in drift.columns:
    mask = (
        drift[target_col].apply(_is_recid_positive)
        & (drift['race'].astype(str).str.strip() == 'African-American')
    )
    edited_indices = list(drift.index[mask])
    drift.loc[edited_indices, 'race'] = 'Caucasian'
    _apply_label_flips(
        drift, edited_indices, target_col, DEFAULT_LABEL_FLIP_PROB, rng,
    )
  return _concat([warmup, drift, recovery])


# ---------------------------------------------------------------------------
# Scenario: gradual race + concept drift
# ---------------------------------------------------------------------------
def gradual_race(df, target_col='two_year_recid'):
  """Linearly ramping race swap + 100% label flip on rows that got swapped."""
  rng = random.Random(SEED)
  warmup, drift, recovery = _split_phases(df)
  n = max(len(drift) - 1, 1)
  edited_indices = []
  for i, idx in enumerate(drift.index):
    prob = i / n  # 0 -> 1
    if (
        _is_recid_positive(drift.at[idx, target_col])
        and str(drift.at[idx, 'race']).strip() == 'African-American'
        and rng.random() < prob
    ):
      drift.at[idx, 'race'] = 'Caucasian'
      edited_indices.append(idx)
  _apply_label_flips(
      drift, edited_indices, target_col, DEFAULT_LABEL_FLIP_PROB, rng,
  )
  return _concat([warmup, drift, recovery])


# ---------------------------------------------------------------------------
# Scenario: age <-> race decoupling + concept drift
# ---------------------------------------------------------------------------
def age_race_decouple(df, target_col='two_year_recid'):
  """Invert age_cat<->race co-occurrence + 100% label flip on swapped rows."""
  rng = random.Random(SEED)
  warmup, drift, recovery = _split_phases(df)
  edited_indices = []
  for idx in drift.index:
    race = str(drift.at[idx, 'race']).strip()
    age = str(drift.at[idx, 'age_cat']).strip()
    if race == 'African-American' and age == 'Less than 25':
      drift.at[idx, 'age_cat'] = 'Greater than 45'
      edited_indices.append(idx)
    elif race == 'Caucasian' and age == 'Greater than 45':
      drift.at[idx, 'age_cat'] = 'Less than 25'
      edited_indices.append(idx)
  _apply_label_flips(
      drift, edited_indices, target_col, DEFAULT_LABEL_FLIP_PROB, rng,
  )
  return _concat([warmup, drift, recovery])


# ---------------------------------------------------------------------------
# Scenario: charge-degree race reversal + concept drift
# ---------------------------------------------------------------------------
def charge_degree_race_swap(df, target_col='two_year_recid'):
  """Swap race in stereotyped charge-degree combos + 100% label flip on swapped rows."""
  rng = random.Random(SEED)
  warmup, drift, recovery = _split_phases(df)
  edited_indices = []
  for idx in drift.index:
    race = str(drift.at[idx, 'race']).strip()
    degree = str(drift.at[idx, 'c_charge_degree']).strip()
    if race == 'African-American' and degree == 'F':
      drift.at[idx, 'race'] = 'Caucasian'
      edited_indices.append(idx)
    elif race == 'Caucasian' and degree == 'M':
      drift.at[idx, 'race'] = 'African-American'
      edited_indices.append(idx)
  _apply_label_flips(
      drift, edited_indices, target_col, DEFAULT_LABEL_FLIP_PROB, rng,
  )
  return _concat([warmup, drift, recovery])


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
    'abrupt_race': 'Sustained abrupt race swap + 100% label flip on edited rows',
    'gradual_race': 'Linearly ramping race swap + 100% label flip on swapped rows',
    'age_race_decouple': 'Invert age_cat<->race co-occurrence + 100% label flip on swapped rows',
    'charge_degree_race_swap': 'Swap race in stereotyped charge-degree combos + 100% label flip on swapped rows',
}


def get_compas_scenario(name):
  """Return the drift function for the given COMPAS scenario name."""
  if name not in COMPAS_SCENARIOS:
    raise ValueError(
        f"Unknown COMPAS drift scenario '{name}'. "
        f"Available: {list(COMPAS_SCENARIOS.keys())}"
    )
  return COMPAS_SCENARIOS[name]
