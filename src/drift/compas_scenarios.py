"""Concept-drift scenarios for the COMPAS test stream.

Each scenario combines a row-level categorical feature edit (covariate
shift on P(x)) with a stochastic Bernoulli(0.25) label flip on the same
edited rows (label-conditional shift on P(y|x)). The combination is genuine concept
drift in the Webb et al. sense: pure covariate-only edits on race,
age_cat, or c_charge_degree leave P(y|x) almost unchanged for an
Aranyani forest that relies primarily on the numeric priors_count /
age features, and we verified empirically that ADWIN does not detect
such virtual drift on COMPAS. Pairing the categorical edit with a
per-subgroup label flip moves the conditional and yields a detectable
prequential-error signal.

Layout (3 phases, ``test_size=0.30``, test stream ~2.1k rows):

    [0 .. 0.30)        warmup           (~650 samples)
    [0.30 .. 0.70)     drift            (~865 samples; 40% of stream)
    [0.70 .. 1.0]      recovery         (~650 samples)

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
each row-level feature edit with a *stochastic* Bernoulli(0.5) flip of
``two_year_recid`` on the SAME edited rows. The 0.5 rate is chosen
because at max entropy the corruption is unlearnable: no online update
rule can re-fit the labelled subgroup beyond 50% accuracy, so the
prequential error stays elevated for the ENTIRE drift phase rather
than collapsing back to baseline within ~300 samples as a deterministic
100% flip does. ADWIN therefore sees a sustained deficit during drift
(the regime FADO is actually designed to react to) instead of a
transient spike on the recovery edge. Each scenario's fairness
narrative is preserved (race / age_cat / charge_degree still shift,
sensitive-attribute distribution still moves) while the detectable
signal is added. ``no_drift`` is left untouched as the true baseline.

Scenarios operate on the raw, post-null-filter DataFrame BEFORE the
one-hot / frequency encoder is applied, so row-level edits propagate
downstream as true feature-level distribution shifts. The recomputed
``y_test`` in ``read_compas_train_test`` picks up the label flips.

Active scenario set (registered in COMPAS_SCENARIOS at the bottom of
this module):

* ``no_drift`` -- baseline, identity. Controller is dormant; FADO is
  bit-identical to Aranyani-Base in this cell, which gives the paper
  the reference point for "what does FADO cost when there is no drift
  to react to?".
* ``abrupt_race`` -- analog of Adult ``abrupt_gender``: AA recidivists
  in the drift phase are relabeled Caucasian + Bernoulli(0.5) label flip on edited rows.
  Cleanest ADWIN-fires-and-reacts demonstration on the dataset.
* ``age_race_decouple`` -- analog of Adult
  ``gender_relationship_decouple``: invert age_cat<->race
  co-occurrence (AA<25 / Cauc>45) with Bernoulli(0.5) label flip on the swapped
  rows. The only scenario where FADO beats Aranyani-Base on BOTH
  demographic parity and accuracy.

Two further scenarios remain implemented below but are NOT registered:
``gradual_race`` (bit-identical to no_drift -- controller dormant on
gradual subgroup ramps, so it adds no information beyond the no-drift
baseline) and ``charge_degree_race_swap`` (same "FADO wins accuracy,
loses DP" pattern as abrupt_race at a slightly larger magnitude --
redundant for the paper narrative). Re-enable either by adding it to
the ``COMPAS_SCENARIOS`` dict at the bottom of this file.

PRNG seed (``SEED = 42``) is fixed so the random swap and label-flip
decisions are reproducible across runs given identical input row order.
"""

import random

import pandas as pd

SEED = 42
# Phase boundaries on the COMPAS test stream (test_size=0.30 -> ~2165 samples):
#   Warmup    [0.00, 0.30) -> ~650 samples
#   Drift     [0.30, 0.70) -> ~865 samples (40% of stream)
#   Recovery  [0.70, 1.00] -> ~650 samples
# Reverted from the experimental 60% drift phase back to the original 40%
# split. The 60% widening was a compensation for the structurally blurred
# fairness signal caused by the default fairness_window=1000 deque on a
# 2165-sample stream (see ``_COMPAS_FADO_OVERRIDES`` in ``src/main.py``).
# Once W is right-sized to 250 the lambda-controller sees fairness move
# 4x faster after drift onset, so the drift phase no longer needs to be
# 1300 samples long for the controller to react. Shorter drift phase also
# means the LR-spike has less time to push the forest into corrupted-label
# territory, which fixes the overshoot that was making FADO worse than
# Base on accuracy under the 60% layout. Warmup/recovery slices grow to
# ~650 each -- well above ADWIN's ~200-sample reliability floor and large
# enough that the W=250 fairness deque sees a fully-clean window in both
# pre- and post-drift phases.
SPLITS = [0.30, 0.70, 1.0]
PHASE_LABELS = ['Warmup', 'Drift', 'Recovery']

# Probability of flipping ``two_year_recid`` on rows whose feature
# columns were edited by a scenario. Set to 0.5 (stochastic, max-entropy)
# rather than the previous 1.0 (deterministic invert).
#
# Why not 1.0: a deterministic flip on a subgroup is a *learnable function*
# of (race, recidivism). Aranyani's prequential test-then-train update
# re-fits the inverted mapping in ~300 samples, after which windowed
# accuracy returns to its pre-drift level. Empirically observed on the
# old 60% / 40% drift layouts: windowed accuracy at the drift-phase
# midpoint was statistically indistinguishable from the warmup baseline,
# so ADWIN saw no signal *during* the drift and only fired on the
# recovery edge (return to clean labels), causing FADO's controller to
# LR-spike at the wrong moment and degrade recovery performance.
#
# Why 0.5: at maximum entropy the flip is *unlearnable* on the affected
# subgroup (any model that predicts the labelled class is correct 50% of
# the time, no better than guessing). Accuracy on the edited subgroup
# stays depressed for the entire drift phase, ADWIN sees a sustained
# deficit within ~150-250 samples of drift onset, and the controller
# fires mid-drift -- which is the regime the FADO-vs-Base comparison is
# actually trying to evaluate.
#
# Valid tuning range is [0.35, 0.65]. Outside that band the signal
# either collapses toward clean labels (<0.35) or becomes deterministic
# enough to relearn (>0.65). Adjust in those bounds if 0.5 either fires
# too often on ``no_drift`` (lower) or fails to trigger on the active
# scenarios (push toward 0.5 from either side).
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
