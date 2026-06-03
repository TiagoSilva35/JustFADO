#!/usr/bin/env python
# coding: utf-8
# %%

# %%


"""Dataloaders."""


# %%


import os
from glob import glob


# %%

import pickle
import numpy as np
import pandas as pd
from folktables import ACSDataSource, BasicProblem

from sklearn.model_selection import train_test_split

from src.drift.create_drifted_ds import generate_drifted_dataset
from src.drift.scenarios import get_scenario, SCENARIOS
from src.drift.compas_scenarios import COMPAS_SCENARIOS, get_compas_scenario


# %%


IM_WIDTH = IM_HEIGHT = 160


# %%


def _normalize_categorical_values(values):
  normalized = values.where(values.notna(), '__missing__')
  normalized = normalized.astype(str).str.strip().replace('', '__missing__')
  return normalized


def _fit_tabular_transformer(
    features,
    categorical_columns,
    frequency_encode_columns=None,
    rare_category_min_count=1,
):
  """Fit a train-only tabular transformer and transform features."""
  work = features.copy()
  categorical = [column for column in categorical_columns if column in work.columns]
  numeric = [column for column in work.columns if column not in categorical]
  frequency_encode_columns = [
      column for column in (frequency_encode_columns or [])
      if column in categorical
  ]
  frequency_encode_set = set(frequency_encode_columns)
  one_hot_columns = [
      column for column in categorical
      if column not in frequency_encode_set
  ]
  min_count = int(max(1, rare_category_min_count))

  numeric_medians, numeric_means, numeric_stds = {}, {}, {}
  numeric_frame = pd.DataFrame(index=work.index)
  for column in numeric:
    values = pd.to_numeric(work[column], errors='coerce')
    median = float(values.median()) if values.notna().any() else 0.0
    values = values.fillna(median)
    mean = float(values.mean()) if values.notna().any() else 0.0
    std = float(values.std()) if values.notna().any() else 1.0
    if not np.isfinite(std) or std <= 0:
      std = 1.0
    numeric_medians[column] = median
    numeric_means[column] = mean
    numeric_stds[column] = std
    numeric_frame[column] = (values - mean) / std

  categorical_dummy_columns = {}
  categorical_frequency_maps = {}
  categorical_frames = []
  for column in one_hot_columns:
    values = _normalize_categorical_values(work[column])
    dummies = pd.get_dummies(values, prefix=column, dtype=np.float32)
    categorical_dummy_columns[column] = list(dummies.columns)
    categorical_frames.append(dummies)

  for column in frequency_encode_columns:
    values = _normalize_categorical_values(work[column])
    if min_count > 1:
      counts = values.value_counts()
      rare_values = counts[counts < min_count].index
      if len(rare_values):
        values = values.where(~values.isin(rare_values), '__other__')
    frequencies = values.value_counts(normalize=True).astype(np.float32).to_dict()
    categorical_frequency_maps[column] = frequencies
    encoded = values.map(frequencies).fillna(0.0).astype(np.float32)
    categorical_frames.append(pd.DataFrame({f'{column}__freq': encoded}, index=work.index))

  categorical_frame = (
      pd.concat(categorical_frames, axis=1)
      if categorical_frames
      else pd.DataFrame(index=work.index)
  )
  transformed = pd.concat([numeric_frame, categorical_frame], axis=1)
  transformed = transformed.astype(np.float32)
  transformer = {
      'feature_columns': list(transformed.columns),
      'numeric_columns': numeric,
      'categorical_columns': categorical,
      'frequency_encoded_columns': frequency_encode_columns,
      'numeric_medians': numeric_medians,
      'numeric_means': numeric_means,
      'numeric_stds': numeric_stds,
      'categorical_dummy_columns': categorical_dummy_columns,
      'categorical_frequency_maps': categorical_frequency_maps,
      'categorical_rare_min_count': min_count,
  }
  return transformed.to_numpy(dtype=np.float32), transformer


def _transform_tabular_features(features, transformer):
  """Apply a fitted tabular transformer to new feature data."""
  work = features.copy()
  categorical_columns = transformer.get('categorical_columns', [])
  frequency_encoded_columns = set(transformer.get('frequency_encoded_columns', []))
  categorical_frequency_maps = transformer.get('categorical_frequency_maps', {})
  numeric_frame = pd.DataFrame(index=work.index)
  for column in transformer['numeric_columns']:
    if column in work.columns:
      values = pd.to_numeric(work[column], errors='coerce')
    else:
      values = pd.Series(np.nan, index=work.index)
    values = values.fillna(transformer['numeric_medians'][column])
    mean = transformer['numeric_means'][column]
    std = transformer['numeric_stds'][column]
    numeric_frame[column] = (values - mean) / std

  categorical_frames = []
  for column in categorical_columns:
    if column in work.columns:
      values = _normalize_categorical_values(work[column])
    else:
      values = pd.Series('__missing__', index=work.index, dtype='object')
    if column in frequency_encoded_columns:
      frequencies = categorical_frequency_maps.get(column, {})
      if '__other__' in frequencies:
        values = values.where(values.isin(set(frequencies.keys())), '__other__')
      encoded = values.map(frequencies).fillna(0.0).astype(np.float32)
      categorical_frames.append(pd.DataFrame({f'{column}__freq': encoded}, index=work.index))
    else:
      dummies = pd.get_dummies(values, prefix=column, dtype=np.float32)
      dummies = dummies.reindex(
          columns=transformer['categorical_dummy_columns'][column],
          fill_value=0.0,
      )
      categorical_frames.append(dummies)

  categorical_frame = (
      pd.concat(categorical_frames, axis=1)
      if categorical_frames
      else pd.DataFrame(index=work.index)
  )
  transformed = pd.concat([numeric_frame, categorical_frame], axis=1)
  transformed = transformed.reindex(
      columns=transformer['feature_columns'],
      fill_value=0.0,
  )
  return transformed.to_numpy(dtype=np.float32)


def _adult_income_to_binary(series):
  normalized = (
      pd.Series(series).astype(str).str.strip().str.replace('.', '', regex=False).str.lower()
  )
  unique_values = set(normalized.unique())
  if unique_values.issubset({'<=50k', '>50k'}):
    return (normalized == '>50k').astype(np.int32).to_numpy()
  return _normalize_binary_series(normalized, 'target')


def _adult_gender_to_binary(series):
  normalized = pd.Series(series).astype(str).str.strip().str.lower()
  unique_values = set(normalized.unique())
  if unique_values.issubset({'male', 'female'}):
    return (normalized == 'male').astype(np.int32).to_numpy()
  return _normalize_binary_series(normalized, 'sensitive attribute')


def _prepare_adult_frame(df):
  frame = df.dropna().copy()
  if 'marital-status' in frame.columns:
    frame['marital-status'] = frame['marital-status'].replace(
        {
            'Divorced': 'not married',
            'Married-AF-spouse': 'married',
            'Married-civ-spouse': 'married',
            'Married-spouse-absent': 'married',
            'Never-married': 'not married',
            'Separated': 'not married',
            'Widowed': 'not married',
        }
    )
  return frame


def _preprocess_adult_frame(df, feature_transformer=None, fit_feature_transformer=False):
  frame = _prepare_adult_frame(df)
  y = _adult_income_to_binary(frame['income'])
  a = _adult_gender_to_binary(frame['gender'])

  features = frame.drop(columns=['income']).copy()
  categorical_features = [
      'workclass',
      'education',
      'marital-status',
      'occupation',
      'relationship',
      'race',
      'gender',
      'native-country',
  ]
  if fit_feature_transformer or feature_transformer is None:
    x, feature_transformer = _fit_tabular_transformer(features, categorical_features)
  else:
    x = _transform_tabular_features(features, feature_transformer)
  return x, y, a, feature_transformer


def preprocess_adult(df):
  """Pre-process the Adult dataset.

  Args:
    df: pandas data frame.

  Returns:
  """
  x, y, a, _ = _preprocess_adult_frame(
      df,
      feature_transformer=None,
      fit_feature_transformer=True,
  )
  return x, y, a


# %%


def read_adult(drift, path='data/adult', drift_scenario=None):
  """Read the Adult dataset.

  Args:
    drift: bool or str – if True uses default abrupt_gender scenario,
           if a string it selects the named scenario from drift.scenarios.
    path: path to the adult data directory.
    drift_scenario: explicit scenario name (overrides *drift* when given).

  Returns:
    x_train, x_test, y_train, y_test, a_train, a_test
  """

  columns = [
      'age',
      'workclass',
      'fnlwgt',
      'education',
      'education-num',
      'marital-status',
      'occupation',
      'relationship',
      'race',
      'gender',
      'capital gain',
      'capital loss',
      'hours per week',
      'native-country',
      'income',
  ]

  with open(os.path.join(path, 'adult.data'), 'rb') as f:
    train_df = pd.read_csv(f, names=columns)


  x_train, y_train, a_train, adult_transformer = _preprocess_adult_frame(
      train_df,
      feature_transformer=None,
      fit_feature_transformer=True,
  )

  # Load test data if available
  test_file_path = os.path.join(path, 'adult.test')
  if os.path.exists(test_file_path):
    with open(test_file_path, 'rb') as f:
        # Skip the first line if it contains a header or comment
        df = pd.read_csv(f, names=columns, skiprows=1)
        print(f"This set contains, {len(df[df['gender'] == ' Female'])} female samples")
        print(f"Income column unique values: {df['income'].unique()}")
        print(f"Samples with income >50K: {len(df[df['income'].str.strip() == '>50K'])}")
        print(f"Samples with income >50K.: {len(df[df['income'].str.strip() == '>50K.'])}")
        print(f"Male samples with >50K income: {len(df[(df['gender'].str.strip() == 'Male') & (df['income'].str.strip() == '>50K')])}")
        print(f"Male samples with >50K. income: {len(df[(df['gender'].str.strip() == 'Male') & (df['income'].str.strip() == '>50K.')])}")
        _scenario_name = drift_scenario  
        if _scenario_name is None and isinstance(drift, str) and drift in SCENARIOS:
            _scenario_name = drift
        elif _scenario_name is None and drift is True:
            _scenario_name = 'abrupt_gender'  # backwards-compatible default

        if _scenario_name:
            scenario_fn = get_scenario(_scenario_name)
            print(f"Applying drift scenario: {_scenario_name}")
            test_df = scenario_fn(df)
            print(f"This drifted set contains {len(test_df[test_df['gender'] == ' Female'])} female examples")
        else:
            test_df = df

    x_test, y_test, a_test, _ = _preprocess_adult_frame(
        test_df,
        feature_transformer=adult_transformer,
        fit_feature_transformer=False,
    )
  else:
    x_test, y_test, a_test = [], [], []

  return x_train, x_test, y_train, y_test, a_train, a_test


# %%


def load_drifted_test_set(scenario_name, path='data/adult'):
  """Load and preprocess only the adult test set with a drift scenario applied.

  This is a lightweight alternative to read_adult() for evaluating an
  already-trained model against different drift scenarios without reloading
  the training data.

  Args:
    scenario_name: name of the drift scenario (key in SCENARIOS dict).
    path: path to the adult data directory.

  Returns:
    x_test, y_test, a_test  (NumPy arrays, preprocessed)
  """

  columns = [
      'age', 'workclass', 'fnlwgt', 'education', 'education-num',
      'marital-status', 'occupation', 'relationship', 'race', 'gender',
      'capital gain', 'capital loss', 'hours per week', 'native-country',
      'income',
  ]

  test_file_path = os.path.join(path, 'adult.test')
  train_file_path = os.path.join(path, 'adult.data')
  with open(train_file_path, 'rb') as f:
    train_df = pd.read_csv(f, names=columns)
  _, _, _, adult_transformer = _preprocess_adult_frame(
      train_df,
      feature_transformer=None,
      fit_feature_transformer=True,
  )

  with open(test_file_path, 'rb') as f:
    df = pd.read_csv(f, names=columns, skiprows=1)

  if scenario_name and scenario_name != 'no_drift':
    scenario_fn = get_scenario(scenario_name)
    print(f"Applying drift scenario: {scenario_name}")
    df = scenario_fn(df)
  else:
    print("No drift applied (baseline)")

  x_test, y_test, a_test, _ = _preprocess_adult_frame(
      df,
      feature_transformer=adult_transformer,
      fit_feature_transformer=False,
  )
  return x_test, y_test, a_test


# %%


def _preprocess_census_frame(df, feature_transformer=None, fit_feature_transformer=False):
  """Pre-process the Census dataset.

  Args:
    df:

  Returns:
  """
  frame = df.dropna().copy()
  categorical_features = [
      'class_worker',
      'education',
      'hs_college',
      'marital_stat',
      'major_ind_code',
      'major_occ_code',
      'race',
      'hisp_origin',
      'sex',
      'union_member',
      'unemp_reason',
      'full_or_part_emp',
      'tax_filer_stat',
      'region_prev_res',
      'state_prev_res',
      'det_hh_fam_stat',
      'det_hh_summ',
      'mig_chg_msa',
      'mig_chg_reg',
      'mig_move_reg',
      'mig_same',
      'mig_prev_sunbelt',
      'fam_under_18',
      'country_father',
      'country_mother',
      'country_self',
      'citizenship',
      'vet_question',
  ]
  y = _normalize_binary_series(frame['income_50k'], 'target')
  a = _normalize_binary_series(frame['sex'], 'sensitive attribute')
  features = frame.drop(columns=['income_50k']).copy()
  if 'unk' in features.columns:
    features = features.drop(columns=['unk'])

  if fit_feature_transformer or feature_transformer is None:
    x, feature_transformer = _fit_tabular_transformer(features, categorical_features)
  else:
    x = _transform_tabular_features(features, feature_transformer)

  return x, np.asarray(y, dtype=np.int32), np.asarray(a, dtype=np.int32), feature_transformer


def preprocess_census(df):
  x, y, a, _ = _preprocess_census_frame(
      df,
      feature_transformer=None,
      fit_feature_transformer=True,
  )
  return x, y, a


# %%


def read_census(path='../data/census/'):
  """Read the Census dataset.

  Column names borrowed from:
  https://docs.1010data.com/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
  1 unidentified column name marked as 'unk' and dropped later.

  Args:
    path:

  Returns:

  """
  column_names = [
      'age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education',
      'wage_per_hour', 'hs_college', 'marital_stat', 'major_ind_code',
      'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
      'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses',
      'stock_dividends', 'tax_filer_stat', 'region_prev_res', 'state_prev_res',
      'det_hh_fam_stat', 'det_hh_summ', 'unk', 'mig_chg_msa', 'mig_chg_reg',
      'mig_move_reg', 'mig_same', 'mig_prev_sunbelt', 'num_emp', 'fam_under_18',
      'country_father', 'country_mother', 'country_self', 'citizenship',
      'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked',
      'year', 'income_50k',
  ]

  # we only use the test set for online learning
  with open(os.path.join(path, 'census-income.data'), 'rb') as f:
    df = pd.read_csv(f, names=column_names)
  x, y, a = preprocess_census(df)
  return x, [], y, [], a, []


def read_jigsaw(path='../data/jigsaw/'):
  with open(os.path.join(path, "jigsaw.pkl"), "rb") as f:
    inputs, texts, Y, A = pickle.load(f)

  X = np.array(inputs)
  Y = np.array(Y)
  A = np.array(A)
  return X, Y, A


def _resolve_compas_files(path):
  """Resolve COMPAS CSV inputs from a file, directory, or glob path."""
  path = str(path).strip()
  if not path:
    path = 'data/compas/*'

  if any(token in path for token in ['*', '?', '[']):
    files = sorted(glob(path))
  elif os.path.isdir(path):
    files = sorted(glob(os.path.join(path, '*.csv')))
  elif os.path.isfile(path):
    files = [path]
  else:
    files = []

  files = [
      file_path for file_path in files
      if os.path.isfile(file_path) and str(file_path).lower().endswith('.csv')
  ]
  if not files:
    raise FileNotFoundError(
        f"No COMPAS data files found for path '{path}'. "
        "Provide a valid file, directory, or glob (e.g., data/compas/*)."
    )
  return files


_COMPAS_TARGET_CANDIDATES = [
    'two_year_recid', 'is_recid', 'recid', 'label', 'target', 'y',
]
_COMPAS_SENSITIVE_CANDIDATES = [
    'race', 'ethnicity', 'sensitive', 'sensitive_attribute', 'group', 'a',
]
_COMPAS_LEGACY_FEATURE_NAMES = [
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "age",
    "c_charge_degree",
    "c_charge_desc",
    "age_cat",
    "sex",
    "race",
    "is_recid",
]
_COMPAS_PREFERRED_FEATURE_COLUMNS = [
    'juv_fel_count',
    'juv_misd_count',
    'juv_other_count',
    'priors_count',
    'age',
    'c_charge_degree',
    'c_charge_desc',
    'age_cat',
    'sex',
    'race',
]


def _read_compas_work_df(path):
  """Read raw COMPAS CSV(s) and return a cleaned, un-encoded DataFrame.

  Returns:
    work_df: pandas DataFrame containing only the columns we use
      (features + target + sensitive), with rows missing target/sensitive
      values dropped. Categorical columns are still raw strings so drift
      scenarios in ``src/drift/compas_scenarios.py`` can manipulate them
      before encoding.
    feature_columns: list of feature column names in canonical order.
    target_col, sensitive_col: resolved column names.
  """

  def _has_any_column(frame, candidates):
    columns = {str(column).strip().lower() for column in frame.columns}
    return any(candidate in columns for candidate in candidates)

  files = _resolve_compas_files(path)
  frames = []
  for file_path in files:
    frame = pd.read_csv(file_path)
    if not (
        _has_any_column(frame, _COMPAS_TARGET_CANDIDATES)
        and _has_any_column(frame, _COMPAS_SENSITIVE_CANDIDATES)
    ):
      frame = pd.read_csv(
          file_path, names=_COMPAS_LEGACY_FEATURE_NAMES, header=None
      )
    frames.append(frame)
  df = pd.concat(frames, ignore_index=True).copy()

  target_col = _resolve_column_name(df, _COMPAS_TARGET_CANDIDATES, 'target')
  sensitive_col = _resolve_column_name(
      df, _COMPAS_SENSITIVE_CANDIDATES, 'sensitive attribute'
  )

  present_preferred = [
      column for column in _COMPAS_PREFERRED_FEATURE_COLUMNS if column in df.columns
  ]
  if len(present_preferred) >= 3:
    feature_columns = list(dict.fromkeys(present_preferred))
  else:
    feature_columns = [
        column for column in df.columns
        if column != target_col
        and str(column).strip().lower() not in set(_COMPAS_TARGET_CANDIDATES)
    ]
  required_columns = list(
      dict.fromkeys(feature_columns + [target_col, sensitive_col])
  )
  work_df = df[required_columns].copy()
  work_df = work_df.dropna(subset=[target_col, sensitive_col]).copy()
  if work_df.empty:
    raise ValueError(
        "COMPAS data has no usable rows after filtering missing target/sensitive values."
    )
  return work_df, feature_columns, target_col, sensitive_col


def _compas_target_to_binary(values):
  """COMPAS target (``two_year_recid``) -> np.int32 with deterministic 0/1.

  ``_normalize_binary_series`` uses ``pd.factorize`` which assigns code 0
  to whichever value appears first -- order-dependent, and silently
  inverts when called on disjoint train/test halves. For COMPAS the
  target is always int {0, 1} (or a clean yes/no string), so we can map
  it deterministically without factorize, which lets us re-encode
  ``y_test`` after a scenario has flipped some labels and stay
  consistent with the pre-split ``y_full`` encoding.
  """
  series = pd.Series(values)
  if pd.api.types.is_numeric_dtype(series):
    return np.asarray(
        pd.to_numeric(series, errors='coerce').fillna(0).astype(np.int32)
    )
  normalized = series.astype(str).str.strip().str.lower()
  if normalized.isin({'0', '1'}).all():
    return normalized.astype(np.int32).to_numpy()
  if normalized.isin({'yes', 'no', 'true', 'false', '0', '1'}).any():
    return normalized.isin({'yes', 'true', '1'}).astype(np.int32).to_numpy()
  # Last-resort fallback: order-dependent factorize. Caller should
  # avoid this path on disjoint slices.
  return np.asarray(_normalize_binary_series(values, 'target'), dtype=np.int32)


def _compas_sensitive_to_binary(values):
  """Encode COMPAS sensitive attribute as binary (1=white/Caucasian, 0=other)."""
  values = pd.Series(values)
  if values.dtype.kind in {'O', 'U', 'S'}:
    normalized = values.astype(str).str.strip().str.lower()
    known_groups = {
        'white', 'caucasian', 'black', 'african-american',
        'other', 'asian', 'hispanic', 'native american',
    }
    if normalized.isin(known_groups).any():
      return normalized.isin({'white', 'caucasian'}).astype(np.int32).to_numpy()
    return _normalize_binary_series(normalized, 'sensitive attribute')
  return _normalize_binary_series(values, 'sensitive attribute')


def _encode_compas_features(work_df, target_col, feature_transformer=None, fit=False):
  """Encode only the feature columns of a COMPAS frame and return x + transformer.

  Split out from ``_encode_compas_frame`` so callers that pre-encode the
  binary target and sensitive labels globally (to avoid order-dependent
  ``pd.factorize`` between disjoint train/test halves) can reuse the
  same tabular transformer for features alone.
  """
  features = work_df.drop(columns=[target_col]).copy()
  if features.empty:
    raise ValueError("COMPAS data has no feature columns after dropping target column.")

  categorical_features = [
      column for column in features.columns
      if not pd.api.types.is_numeric_dtype(features[column])
  ]
  if fit or feature_transformer is None:
    x, feature_transformer = _fit_tabular_transformer(
        features,
        categorical_features,
        frequency_encode_columns=['c_charge_desc'],
        rare_category_min_count=10,
    )
  else:
    x = _transform_tabular_features(features, feature_transformer)
  return np.asarray(x, dtype=np.float32), feature_transformer


def _encode_compas_frame(
    work_df, target_col, sensitive_col,
    feature_transformer=None, fit=False,
):
  """Encode a COMPAS frame end-to-end (x, y, a) using the tabular transformer.

  WARNING: ``y`` and ``a`` are factorised on this frame in isolation, so
  calling this on disjoint train/test halves can produce inconsistent
  binary encodings (whichever class appears first in each half gets code
  0). For drift-aware splits use ``_encode_compas_features`` and
  pre-encode y / a globally before splitting (see
  ``read_compas_train_test``).
  """
  y = np.asarray(
      _normalize_binary_series(work_df[target_col], 'target'), dtype=np.int32
  )
  a = np.asarray(
      _compas_sensitive_to_binary(work_df[sensitive_col]), dtype=np.int32
  )
  x, feature_transformer = _encode_compas_features(
      work_df,
      target_col=target_col,
      feature_transformer=feature_transformer,
      fit=fit,
  )
  return x, y, a, feature_transformer


def read_compas(path="data/compas/*"):
  """Legacy COMPAS loader: encode the full dataset, return (x, y, a).

  Used by code paths that build train/test via post-hoc array splits
  (e.g. ``_smart_split`` in ``main.py``). For drift-aware evaluation use
  ``read_compas_train_test`` instead, which splits raw rows before
  encoding so per-scenario shifts can be applied to the test slice.
  """
  work_df, feature_columns, target_col, sensitive_col = _read_compas_work_df(path)
  print(f"Using {len(feature_columns)} features: {feature_columns}")
  x, y, a, _ = _encode_compas_frame(
      work_df, target_col=target_col, sensitive_col=sensitive_col, fit=True,
  )
  return x, y, a


def read_compas_train_test(
    scenario_name=None,
    path='data/compas/*',
    seed=42,
    test_size=0.3,
):
  """Read COMPAS, split train/test by seed, apply a drift scenario to test.

  The training slice is always kept drift-free (matching the Adult
  pipeline). The drift scenario is applied to the test slice in its raw
  string form, then the tabular transformer is fitted on the train slice
  alone and reused to encode the test slice -- this both avoids
  train/test leakage and ensures the scenario edit on ``race`` /
  ``age_cat`` / ``c_charge_degree`` propagates correctly through the
  one-hot encoder.

  ``test_size`` defaults to 0.30 (~2.1k test samples on COMPAS), giving
  each of the 3 drift phases declared in ``compas_scenarios.py`` enough
  rows (warmup ~430, drift ~865, recovery ~865) to stay above ADWIN's
  reliability floor. Training still has ~5k rows, which is comfortably
  more than enough for the 64-parameter Aranyani forest.

  Stratification matches ``_smart_split``: joint ``(y, a)`` when valid,
  otherwise plain ``y``, otherwise unstratified shuffle. The same
  ``random_state=seed`` is used so seeded paired comparisons share
  identical train/test row partitions across scenarios. The target
  encoding ``y`` is fit on the *full* DataFrame before the split to
  avoid ``pd.factorize`` order-dependence flipping labels between train
  and test (see ``_encode_compas_frame`` docstring).
  """
  work_df, feature_columns, target_col, sensitive_col = _read_compas_work_df(path)
  print(f"Using {len(feature_columns)} features: {feature_columns}")

  # Deterministic target encoding so y_full and the post-scenario
  # y_test recomputation share the same 0/1 mapping (factorize would
  # be order-dependent on disjoint slices -- see _compas_target_to_binary).
  y_full = _compas_target_to_binary(work_df[target_col])
  a_full = _compas_sensitive_to_binary(work_df[sensitive_col])

  joint = np.asarray([f'{int(y)}_{int(a)}' for y, a in zip(y_full, a_full)])
  uniques, counts = np.unique(joint, return_counts=True)
  if len(uniques) >= 2 and np.all(counts >= 2):
    stratify = joint
  else:
    y_uniques, y_counts = np.unique(y_full, return_counts=True)
    stratify = y_full if len(y_uniques) >= 2 and np.all(y_counts >= 2) else None

  indices = np.arange(len(work_df))
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

  train_df = work_df.iloc[train_idx].reset_index(drop=True)
  test_df = work_df.iloc[test_idx].reset_index(drop=True)

  scenario = (scenario_name or '').strip()
  if scenario and scenario != 'no_drift' and scenario in COMPAS_SCENARIOS:
    scenario_fn = get_compas_scenario(scenario)
    print(f"Applying COMPAS drift scenario: {scenario}")
    test_df = scenario_fn(test_df, target_col=target_col)
  else:
    print("COMPAS: no drift applied (baseline)")

  # y_train is sliced from the pre-split deterministic encoding (train
  # is never edited). y_test is RE-encoded from the post-scenario
  # ``test_df`` so concept-drift label flips inside scenarios are
  # honoured; the same deterministic mapping keeps it consistent with
  # y_train / y_full.
  y_train = np.asarray(y_full[train_idx], dtype=np.int32)
  y_test = _compas_target_to_binary(test_df[target_col])

  # a_train is sliced from the pre-split encoding; a_test is recomputed
  # from the (possibly drift-modified) race column so that swaps like
  # African-American -> Caucasian flip the sensitive bit accordingly.
  # _compas_sensitive_to_binary uses a deterministic isin check on known
  # race strings, so it stays consistent with a_full.
  a_train = np.asarray(a_full[train_idx], dtype=np.int32)
  a_test = np.asarray(
      _compas_sensitive_to_binary(test_df[sensitive_col]), dtype=np.int32
  )

  x_train, transformer = _encode_compas_features(
      train_df, target_col=target_col, fit=True,
  )
  x_test, _ = _encode_compas_features(
      test_df, target_col=target_col,
      feature_transformer=transformer, fit=False,
  )
  return x_train, x_test, y_train, y_test, a_train, a_test


# CelebA

def load_dump(file_path):
  with open(file_path, "rb") as f:
    data = pickle.load(f)
  return data


def read_celeba(path="../data/celeba/"):
  X, Y, A = load_dump(os.path.join(path, "clip_data.pkl"))
  x_train, y_train, a_train = X, Y[: len(X)], A[: len(X)]
  return x_train, y_train, a_train


def _resolve_diabetes_files(path):
  """Resolve diabetes CSV inputs from a file, directory, or glob path."""
  path = str(path).strip()
  if not path:
    path = 'data/diabetes/diabetic_data.csv'

  if any(token in path for token in ['*', '?', '[']):
    files = sorted(glob(path))
  elif os.path.isdir(path):
    files = sorted(glob(os.path.join(path, '*.csv')))
  elif os.path.isfile(path):
    files = [path]
  else:
    files = []

  files = [file_path for file_path in files if os.path.isfile(file_path)]
  if not files:
    raise FileNotFoundError(
        f"No diabetes data files found for path '{path}'. "
        "Provide a valid file, directory, or glob (e.g., data/diabetes/diabetic_data.csv)."
    )

  dataset_files = []
  for file_path in files:
    columns = {
        str(column).strip().lower()
        for column in pd.read_csv(file_path, nrows=0).columns
    }
    if 'readmitted' in columns:
      dataset_files.append(file_path)

  if not dataset_files:
    raise ValueError(
        f"No diabetes dataset CSV was found for path '{path}'. "
        "Expected at least one CSV with a 'readmitted' column."
    )
  return dataset_files


def _resolve_column_name(df, candidates, label):
  column_map = {str(column).strip().lower(): column for column in df.columns}
  for candidate in candidates:
    if candidate in column_map:
      return column_map[candidate]
  raise ValueError(
      f"Could not infer {label} column. Available columns: {list(df.columns)}"
  )


def _normalize_binary_series(series, label):
  values = pd.Series(series)
  if values.empty:
    raise ValueError(f"Cannot parse empty {label} values.")
  if values.dtype.kind in {'O', 'U', 'S'}:
    normalized = values.astype(str).str.strip().str.lower()
  else:
    normalized = values

  encoded = pd.factorize(normalized)[0].astype(np.int32)
  if len(np.unique(encoded)) != 2:
    raise ValueError(
        f"Expected binary {label} values, found {len(np.unique(encoded))} classes."
    )
  return encoded


def _preprocess_diabetes_frame(df, feature_transformer=None, fit_feature_transformer=False):
  df = df.dropna().copy()
  target_col = _resolve_column_name(
      df,
      ['readmitted'],
      'target',
  )
  sensitive_col = _resolve_column_name(
      df,
      ['gender', 'sex', 'sensitive', 'sensitive_attribute', 'group', 'a'],
      'sensitive attribute',
  )

  sensitive_values = df[sensitive_col]
  if sensitive_values.dtype.kind in {'O', 'U', 'S'}:
    normalized_sensitive = sensitive_values.astype(str).str.strip().str.lower()
    binary_sensitive_mask = normalized_sensitive.isin({'male', 'female'})
    if binary_sensitive_mask.any() and not binary_sensitive_mask.all():
      df = df.loc[binary_sensitive_mask].copy()
      normalized_sensitive = normalized_sensitive.loc[df.index]
    sensitive_values = normalized_sensitive

  target_values = df[target_col]
  if target_values.dtype.kind in {'O', 'U', 'S'}:
    normalized_target = target_values.astype(str).str.strip().str.lower()
    if set(normalized_target.unique()).issubset({'no', '>30', '<30'}):
      target_values = normalized_target.replace({'no': 'no', '>30': 'yes', '<30': 'yes'})

  y = _normalize_binary_series(target_values, 'target')
  a = _normalize_binary_series(sensitive_values, 'sensitive attribute')

  features = df.drop(columns=[target_col])
  if features.empty:
    raise ValueError("Diabetes data has no feature columns after dropping target column.")

  categorical_features = [
      column for column in features.columns
      if not pd.api.types.is_numeric_dtype(features[column])
  ]
  if fit_feature_transformer or feature_transformer is None:
    x, feature_transformer = _fit_tabular_transformer(features, categorical_features)
  else:
    x = _transform_tabular_features(features, feature_transformer)
  return x, y, a, feature_transformer


def read_diabetes(path='data/diabetes/diabetic_data.csv'):
  """Read diabetes data from a file, folder, or glob path."""
  files = _resolve_diabetes_files(path)
  train_frames = []
  test_frames = []
  for file_path in files:
    file_name = os.path.basename(file_path).lower()
    frame = pd.read_csv(file_path)
    if 'test' in file_name:
      test_frames.append(frame)
    elif 'train' in file_name:
      train_frames.append(frame)
    else:
      train_frames.append(frame)

  if test_frames:
    test_df = pd.concat(test_frames, ignore_index=True)
    if train_frames:
      train_df = pd.concat(train_frames, ignore_index=True)
    else:
      split_idx = max(1, int(0.8 * len(test_df)))
      if split_idx >= len(test_df):
        split_idx = len(test_df) - 1
      if split_idx <= 0:
        raise ValueError("Not enough diabetes samples to build train/test splits.")
      train_df = test_df.iloc[:split_idx].copy()
      test_df = test_df.iloc[split_idx:].copy()
  else:
    merged_df = pd.concat(train_frames, ignore_index=True)
    split_idx = max(1, int(0.8 * len(merged_df)))
    if split_idx >= len(merged_df):
      split_idx = len(merged_df) - 1
    if split_idx <= 0:
      raise ValueError("Not enough diabetes samples to build train/test splits.")
    train_df = merged_df.iloc[:split_idx].copy()
    test_df = merged_df.iloc[split_idx:].copy()

  x_train, y_train, a_train, diabetes_transformer = _preprocess_diabetes_frame(
      train_df,
      feature_transformer=None,
      fit_feature_transformer=True,
  )
  x_test, y_test, a_test, _ = _preprocess_diabetes_frame(
      test_df,
      feature_transformer=diabetes_transformer,
      fit_feature_transformer=False,
  )
  return x_train, x_test, y_train, y_test, a_train, a_test


def _adult_income_filter(data):
  """Mimic Adult dataset filtering for ACS Income."""
  df = data
  df = df[df['AGEP'] > 16]
  df = df[df['PINCP'] > 100]
  df = df[df['WKHP'] > 0]
  df = df[df['PWGTP'] >= 1]
  return df


def _acs_income_problem(sensitive_attribute='sex'):
  sensitive_attribute = str(sensitive_attribute).lower()
  features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
  if sensitive_attribute == 'race':
    group = 'RAC1P'
  else:
    group = 'SEX'

  return BasicProblem(
      features=features,
      target='PINCP',
      target_transform=lambda x: x > 50000,
      group=group,
      preprocess=_adult_income_filter,
      postprocess=lambda x: np.nan_to_num(x, -1),
  )


def _normalize_sensitive_attribute(values, sensitive_attribute):
  values = np.asarray(values)
  sensitive_attribute = str(sensitive_attribute).lower()
  if sensitive_attribute == 'race':
    # Binary race split: white (1) vs non-white (0)
    return (values == 1).astype(np.int32)
  # SEX in ACS is {1, 2}; convert to {0, 1}
  return (values - 1).astype(np.int32)


def _preprocess_folktables_frame(
    frame,
    sensitive_attribute,
    feature_transformer=None,
    fit_feature_transformer=False,
):
  work = _adult_income_filter(frame).copy()
  sensitive_attribute = str(sensitive_attribute).lower()
  sensitive_column = 'RAC1P' if sensitive_attribute == 'race' else 'SEX'
  feature_columns = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
  required_columns = list(dict.fromkeys(feature_columns + ['PINCP', sensitive_column]))
  work = work.dropna(subset=required_columns).copy()

  y = (pd.to_numeric(work['PINCP'], errors='coerce') > 50000).astype(np.int32).to_numpy()
  a = _normalize_sensitive_attribute(
      pd.to_numeric(work[sensitive_column], errors='coerce').to_numpy(),
      sensitive_attribute,
  )
  features = work[feature_columns].copy()
  categorical_features = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
  if fit_feature_transformer or feature_transformer is None:
    x, feature_transformer = _fit_tabular_transformer(features, categorical_features)
  else:
    x = _transform_tabular_features(features, feature_transformer)
  return x, y, a, feature_transformer


def read_folktables(path='data/acs-folktables', train_year=2015, test_years=(2017, 2018),
                    state='CA', horizon='1-Year', sensitive_attribute='sex',
                    download=True):
  """Read Folktables ACS Income data from local cache."""
  train_year = int(train_year)
  test_years = tuple(int(year) for year in test_years)
  if isinstance(state, (list, tuple)):
    states = [str(s).strip() for s in state if str(s).strip()]
  else:
    states = [str(state).strip()]
  if not states:
    states = ['CA']

  train_source = ACSDataSource(
      survey_year=train_year, horizon=horizon, survey='person', root_dir=path
  )
  train_df = train_source.get_data(states=states, download=download)
  x_train, y_train, a_train, folktables_transformer = _preprocess_folktables_frame(
      train_df,
      sensitive_attribute=sensitive_attribute,
      feature_transformer=None,
      fit_feature_transformer=True,
  )
  x_tests, y_tests, a_tests = [], [], []
  for year in test_years:
    test_source = ACSDataSource(
        survey_year=year, horizon=horizon, survey='person', root_dir=path
    )
    test_df = test_source.get_data(states=states, download=download)
    x_t, y_t, a_t, _ = _preprocess_folktables_frame(
        test_df,
        sensitive_attribute=sensitive_attribute,
        feature_transformer=folktables_transformer,
        fit_feature_transformer=False,
    )
    # append only a subset of the test data
    split_size = len(x_train) // max(len(test_years), 1)
    x_tests.append(x_t[:split_size])
    y_tests.append(y_t[:split_size])
    a_tests.append(a_t[:split_size])

  x_test = np.concatenate(x_tests, axis=0) if x_tests else np.array([], dtype=np.float32)
  y_test = np.concatenate(y_tests, axis=0) if y_tests else np.array([], dtype=np.int32)
  a_test = np.concatenate(a_tests, axis=0) if a_tests else np.array([], dtype=np.int32)

  return (
      np.asarray(x_train, dtype=np.float32),
      np.asarray(x_test, dtype=np.float32),
      np.asarray(y_train, dtype=np.int32),
      np.asarray(y_test, dtype=np.int32),
      np.asarray(a_train, dtype=np.int32),
      np.asarray(a_test, dtype=np.int32),
      len(set(a_train))
  )
