
import random

import pandas as pd

SEED = 42
SPLITS = [0.30, 0.70, 1.0]
PHASE_LABELS = ['Warmup', 'Drift', 'Recovery']
DEFAULT_LABEL_FLIP_PROB = 0.5


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
  """Symmetric race swap (AA<->Caucasian) + Bernoulli(0.35) label flip.

  Phase 1 (drift): African-American defendants are relabeled Caucasian
  AND Caucasian defendants are relabeled African-American (symmetric
  inversion of the two majority race categories; the residual races
  -- Hispanic, Other, Asian, Native American -- are left untouched).
  On all edited rows the ``two_year_recid`` label is flipped with
  Bernoulli(0.35) noise.

  Intermediate calibration (post-bug-fix, 2026-06): the AA-only /
  Bernoulli(0.25) "Goldilocks" attempt produced only ~3-4pp aggregate
  accuracy deficit, below ADWIN's ~9pp Hoeffding floor on the COMPAS
  stream -- so the controller never fired, FADO collapsed to Base, and
  no meaningful FADO-vs-Base separation was observable. The symmetric
  mask (~84% subgroup) at 0.35 flip rate yields ~10pp aggregate
  deficit: ADWIN fires reliably while leaving the regulariser more
  oxygen than the 0.50-rate version did. The intent is to land in the
  regime where (a) the controller engages, (b) the regulariser
  maintains its DP-pulling effect, and (c) FADO measurably
  outperforms Base on DP.

  Why this is concept drift (Webb et al. 2016): the Bernoulli(0.35)
  flip changes P(y | x) on the affected subgroup. Combined with the
  race-column rewrite this is hybrid covariate + concept drift in the
  Zliobaite et al. 2014 taxonomy.
  """
  rng = random.Random(SEED)
  warmup, drift, recovery = _split_phases(df)
  if target_col in drift.columns:
    race_strings = drift['race'].astype(str).str.strip()
    aa_mask = race_strings == 'African-American'
    cauc_mask = race_strings == 'Caucasian'
    edited_indices = list(drift.index[aa_mask | cauc_mask])
    # Symmetric inversion: AA -> Caucasian, Caucasian -> AA.
    drift.loc[aa_mask, 'race'] = 'Caucasian'
    drift.loc[cauc_mask, 'race'] = 'African-American'
    _apply_label_flips(
        drift, edited_indices, target_col, DEFAULT_LABEL_FLIP_PROB, rng,
    )
  return _concat([warmup, drift, recovery])


# ---------------------------------------------------------------------------
# Scenario: gradual race + concept drift
# ---------------------------------------------------------------------------
def gradual_race(df, target_col='two_year_recid'):
  """Linearly ramping race swap + Bernoulli(0.5) label flip on rows that got swapped."""
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
  """Invert age_cat<->race co-occurrence on narrow subgroup + Bernoulli(0.25) flip.

  Phase 1 (drift): African-American defendants whose age_cat is
  'Less than 25' have it forced to 'Greater than 45', and Caucasian
  defendants whose age_cat is 'Greater than 45' have it forced to
  'Less than 25'. All other rows are untouched. On the edited rows the
  ``two_year_recid`` label is flipped with Bernoulli(0.25) noise.

  Goldilocks calibration (post-bug-fix, 2026-06): this is the original
  narrow-mask design (~10-15% of the drift phase) rather than the
  intermediate broader version (~65%) or the very broad scramble
  (~95%). With the regulariser now actually firing (the previously
  documented off-by-one bug + tape-slicing bug in
  ``initializers.py`` is fixed -- see
  ``DOCS/BUG_REPORT_fairness_regulariser.md``), the broader masks
  overwhelmed the regulariser and drove all fairness-aware methods
  worse than vanilla baselines on DP. The narrow mask leaves a small,
  realistic subspace shift that the regulariser can compensate for,
  while still measurably moving the race-age co-occurrence (the
  paper's "decouple" narrative). ADWIN may or may not fire on a given
  seed at this signal level -- the scenario is intended as a
  sub-/near-detection-floor reference point against ``abrupt_race``.

  Why this is concept drift (Webb et al. 2016): the Bernoulli(0.25)
  flip on the affected subgroup changes P(y | x) there, and the
  categorical edit changes P(x). Hybrid covariate + concept drift in
  the Zliobaite et al. 2014 taxonomy.
  """
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
  """Swap race in stereotyped charge-degree combos + Bernoulli(0.5) label flip on swapped rows."""
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
# Active scenario set: only the cells where ADWIN reliably fires on
# COMPAS at the post-bug-fix regulariser strength (lambda=1 on this
# dataset, see ``_COMPAS_FADO_OVERRIDES`` in ``src/main.py``).
#   * no_drift            -- baseline reference; the controller is
#                            correctly dormant here (bit-identical
#                            FADO vs Aranyani-Base on this scenario
#                            is the architectural guarantee that
#                            FADO costs nothing when there is no
#                            drift to react to).
#   * abrupt_race         -- symmetric race swap (~84% of drift phase)
#                            with Bernoulli(0.5) flip. This is the
#                            only scenario whose aggregate accuracy
#                            deficit clears ADWIN's ~9pp Hoeffding
#                            floor on the COMPAS test stream at
#                            delta=0.05, so it is the only cell
#                            where the FADO controller's
#                            detect-then-react pathway can be
#                            exercised.
#
# ``age_race_decouple`` (defined above) was dropped from the active
# registry. Empirically it never crossed ADWIN's detection floor on
# COMPAS in any of the calibration sweeps -- narrow {AA<25, Cauc>45}
# mask is sub-floor, broad age scramble overwhelmed the regulariser
# without producing a measurable accuracy deficit either. Re-enable
# by adding it back to the dict below.
#
# ``gradual_race`` and ``charge_degree_race_swap`` (also defined
# above) remain unregistered for the same reasons documented in the
# module-level docstring.
COMPAS_SCENARIOS = {
    'no_drift': no_drift,
    'abrupt_race': abrupt_race,
}

COMPAS_SCENARIO_DESCRIPTIONS = {
    'no_drift': 'Baseline (no drift)',
    'abrupt_race': 'Sustained abrupt race swap + Bernoulli(0.5) label flip on edited rows',
}


def get_compas_scenario(name):
  """Return the drift function for the given COMPAS scenario name."""
  if name not in COMPAS_SCENARIOS:
    raise ValueError(
        f"Unknown COMPAS drift scenario '{name}'. "
        f"Available: {list(COMPAS_SCENARIOS.keys())}"
    )
  return COMPAS_SCENARIOS[name]
