import os
from glob import glob

import numpy as np
import pandas as pd
from folktables import ACSDataSource
from sklearn.model_selection import train_test_split

from src.drift.compas_scenarios import COMPAS_SCENARIOS, get_compas_scenario
from src.drift.scenarios import SCENARIOS, get_scenario


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_csv_files(path, label):
  path = str(path).strip()
  if any(token in path for token in ['*', '?', '[']):
    files = sorted(glob(path))
  elif os.path.isdir(path):
    files = sorted(glob(os.path.join(path, '*.csv')))
  elif os.path.isfile(path):
    files = [path]
  else:
    files = []
  files = [f for f in files if os.path.isfile(f) and f.lower().endswith('.csv')]
  if not files:
    raise FileNotFoundError(
        f"No {label} data files found for path '{path}'. "
        "Provide a valid file, directory, or glob."
    )
  return files


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
    values = values.astype(str).str.strip().str.lower()
  encoded = pd.factorize(values)[0].astype(np.int32)
  n_classes = len(np.unique(encoded))
  if n_classes != 2:
    raise ValueError(f"Expected binary {label} values, found {n_classes} classes.")
  return encoded


# ---------------------------------------------------------------------------
# Tabular encoder (fit on train, apply to test)
# ---------------------------------------------------------------------------


def _normalize_categorical_values(values):
  normalized = values.where(values.notna(), '__missing__')
  return normalized.astype(str).str.strip().replace('', '__missing__')


def _fit_tabular_transformer(
    features,
    categorical_columns,
    frequency_encode_columns=None,
    rare_category_min_count=1,
    return_dataframe=False,
):
  categorical = [c for c in categorical_columns if c in features.columns]
  numeric = [c for c in features.columns if c not in categorical]
  frequency_encode_columns = [
      c for c in (frequency_encode_columns or []) if c in categorical
  ]
  one_hot_columns = [c for c in categorical if c not in set(frequency_encode_columns)]
  min_count = int(max(1, rare_category_min_count))

  numeric_medians, numeric_means, numeric_stds = {}, {}, {}
  numeric_frame = pd.DataFrame(index=features.index)
  for column in numeric:
    values = pd.to_numeric(features[column], errors='coerce')
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
    values = _normalize_categorical_values(features[column])
    dummies = pd.get_dummies(values, prefix=column, dtype=np.float32)
    categorical_dummy_columns[column] = list(dummies.columns)
    categorical_frames.append(dummies)

  for column in frequency_encode_columns:
    values = _normalize_categorical_values(features[column])
    if min_count > 1:
      counts = values.value_counts()
      rare_values = counts[counts < min_count].index
      if len(rare_values):
        values = values.where(~values.isin(rare_values), '__other__')
    frequencies = values.value_counts(normalize=True).astype(np.float32).to_dict()
    categorical_frequency_maps[column] = frequencies
    encoded = values.map(frequencies).fillna(0.0).astype(np.float32)
    categorical_frames.append(
        pd.DataFrame({f'{column}__freq': encoded}, index=features.index)
    )

  transformed = pd.concat([numeric_frame, *categorical_frames], axis=1)
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
  if return_dataframe:
    return transformed, transformer
  return transformed.to_numpy(dtype=np.float32), transformer


def _transform_tabular_features(features, transformer):
  """Apply a fitted tabular transformer to new feature data."""
  frequency_encoded = set(transformer.get('frequency_encoded_columns', []))
  frequency_maps = transformer.get('categorical_frequency_maps', {})

  numeric_frame = pd.DataFrame(index=features.index)
  for column in transformer['numeric_columns']:
    if column in features.columns:
      values = pd.to_numeric(features[column], errors='coerce')
    else:
      values = pd.Series(np.nan, index=features.index)
    values = values.fillna(transformer['numeric_medians'][column])
    numeric_frame[column] = (
        (values - transformer['numeric_means'][column])
        / transformer['numeric_stds'][column]
    )

  categorical_frames = []
  for column in transformer.get('categorical_columns', []):
    if column in features.columns:
      values = _normalize_categorical_values(features[column])
    else:
      values = pd.Series('__missing__', index=features.index, dtype='object')
    if column in frequency_encoded:
      frequencies = frequency_maps.get(column, {})
      if '__other__' in frequencies:
        values = values.where(values.isin(set(frequencies)), '__other__')
      encoded = values.map(frequencies).fillna(0.0).astype(np.float32)
      categorical_frames.append(
          pd.DataFrame({f'{column}__freq': encoded}, index=features.index)
      )
    else:
      dummies = pd.get_dummies(values, prefix=column, dtype=np.float32)
      categorical_frames.append(dummies.reindex(
          columns=transformer['categorical_dummy_columns'][column],
          fill_value=0.0,
      ))

  transformed = pd.concat([numeric_frame, *categorical_frames], axis=1)
  transformed = transformed.reindex(
      columns=transformer['feature_columns'], fill_value=0.0,
  )
  return transformed.to_numpy(dtype=np.float32)


def _encode_features(features, categorical_columns, transformer=None, **fit_kwargs):
  """Fit a transformer when none is given, otherwise apply it."""
  if transformer is None:
    return _fit_tabular_transformer(features, categorical_columns, **fit_kwargs)
  return _transform_tabular_features(features, transformer), transformer


# ---------------------------------------------------------------------------
# Adult
# ---------------------------------------------------------------------------

ADULT_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'gender',
    'capital gain', 'capital loss', 'hours per week', 'native-country',
    'income',
]
ADULT_CATEGORICAL = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country',
]
_ADULT_MARITAL_STATUS = {
    'Divorced': 'not married',
    'Married-AF-spouse': 'married',
    'Married-civ-spouse': 'married',
    'Married-spouse-absent': 'married',
    'Never-married': 'not married',
    'Separated': 'not married',
    'Widowed': 'not married',
}


def _adult_income_to_binary(series):
  normalized = (
      pd.Series(series).astype(str).str.strip()
      .str.replace('.', '', regex=False).str.lower()
  )
  if set(normalized.unique()).issubset({'<=50k', '>50k'}):
    return (normalized == '>50k').astype(np.int32).to_numpy()
  return _normalize_binary_series(normalized, 'target')


def _adult_gender_to_binary(series):
  """1 = male, 0 = female."""
  normalized = pd.Series(series).astype(str).str.strip().str.lower()
  if set(normalized.unique()).issubset({'male', 'female'}):
    return (normalized == 'male').astype(np.int32).to_numpy()
  return _normalize_binary_series(normalized, 'sensitive attribute')


def _read_adult_csv(path, file_name):
  # adult.test starts with a '|1x3 Cross validator' comment line.
  skiprows = 1 if file_name == 'adult.test' else 0
  with open(os.path.join(path, file_name), 'rb') as f:
    return pd.read_csv(f, names=ADULT_COLUMNS, skiprows=skiprows)


def _preprocess_adult_frame(df, transformer=None):
  frame = df.dropna().copy()
  frame['marital-status'] = frame['marital-status'].replace(_ADULT_MARITAL_STATUS)
  y = _adult_income_to_binary(frame['income'])
  a = _adult_gender_to_binary(frame['gender'])
  x, transformer = _encode_features(
      frame.drop(columns=['income']), ADULT_CATEGORICAL, transformer,
  )
  return x, y, a, transformer


def _apply_adult_scenario(df, scenario_name):
  if scenario_name and scenario_name != 'no_drift':
    print(f"Applying drift scenario: {scenario_name}")
    return get_scenario(scenario_name)(df)
  return df


def read_adult(drift, path='data/adult', drift_scenario=None):
  """Read Adult: adult.data is the training set, adult.test the stream.

  Args:
    drift: ``True`` applies the default ``abrupt_gender`` scenario, a scenario
      name applies that scenario, anything else applies none.
    path: directory holding adult.data / adult.test.
    drift_scenario: explicit scenario name; overrides ``drift``.

  Returns:
    x_train, x_test, y_train, y_test, a_train, a_test. The test arrays are
    empty lists when adult.test is missing.
  """
  x_train, y_train, a_train, transformer = _preprocess_adult_frame(
      _read_adult_csv(path, 'adult.data')
  )
  if not os.path.exists(os.path.join(path, 'adult.test')):
    return x_train, [], y_train, [], a_train, []

  scenario_name = drift_scenario
  if scenario_name is None and isinstance(drift, str) and drift in SCENARIOS:
    scenario_name = drift
  elif scenario_name is None and drift is True:
    scenario_name = 'abrupt_gender'

  test_df = _apply_adult_scenario(_read_adult_csv(path, 'adult.test'), scenario_name)
  x_test, y_test, a_test, _ = _preprocess_adult_frame(test_df, transformer)
  return x_train, x_test, y_train, y_test, a_train, a_test


def load_drifted_test_set(scenario_name, path='data/adult'):
  """Adult test stream with ``scenario_name`` applied, encoded with the
  transformer fitted on adult.data. Returns x_test, y_test, a_test."""
  _, _, _, transformer = _preprocess_adult_frame(_read_adult_csv(path, 'adult.data'))
  test_df = _apply_adult_scenario(_read_adult_csv(path, 'adult.test'), scenario_name)
  x_test, y_test, a_test, _ = _preprocess_adult_frame(test_df, transformer)
  return x_test, y_test, a_test


# ---------------------------------------------------------------------------
# COMPAS
# ---------------------------------------------------------------------------

_COMPAS_TARGET_CANDIDATES = [
    'two_year_recid', 'is_recid', 'recid', 'label', 'target', 'y',
]
_COMPAS_SENSITIVE_CANDIDATES = [
    'race', 'ethnicity', 'sensitive', 'sensitive_attribute', 'group', 'a',
]
# Header for headerless legacy CSVs.
_COMPAS_LEGACY_COLUMNS = [
    'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
    'age', 'c_charge_degree', 'c_charge_desc', 'age_cat', 'sex', 'race',
    'is_recid',
]
_COMPAS_FEATURE_COLUMNS = _COMPAS_LEGACY_COLUMNS[:-1]


def _read_compas_work_df(path):
  """Read raw COMPAS CSV(s) into an un-encoded frame.

  Categorical columns stay raw strings so drift scenarios in
  ``src/drift/compas_scenarios.py`` can edit them before encoding.

  Returns:
    work_df: features + target + sensitive, rows missing target/sensitive
      dropped.
    feature_columns, target_col, sensitive_col.
  """
  def _has_any_column(frame, candidates):
    columns = {str(column).strip().lower() for column in frame.columns}
    return any(candidate in columns for candidate in candidates)

  frames = []
  for file_path in _resolve_csv_files(path, 'COMPAS'):
    frame = pd.read_csv(file_path)
    if not (_has_any_column(frame, _COMPAS_TARGET_CANDIDATES)
            and _has_any_column(frame, _COMPAS_SENSITIVE_CANDIDATES)):
      frame = pd.read_csv(file_path, names=_COMPAS_LEGACY_COLUMNS, header=None)
    frames.append(frame)
  df = pd.concat(frames, ignore_index=True)

  target_col = _resolve_column_name(df, _COMPAS_TARGET_CANDIDATES, 'target')
  sensitive_col = _resolve_column_name(
      df, _COMPAS_SENSITIVE_CANDIDATES, 'sensitive attribute'
  )

  feature_columns = [c for c in _COMPAS_FEATURE_COLUMNS if c in df.columns]
  if len(feature_columns) < 3:
    feature_columns = [
        c for c in df.columns
        if c != target_col
        and str(c).strip().lower() not in set(_COMPAS_TARGET_CANDIDATES)
    ]
  required_columns = list(dict.fromkeys(feature_columns + [target_col, sensitive_col]))
  work_df = df[required_columns].dropna(subset=[target_col, sensitive_col]).copy()
  if work_df.empty:
    raise ValueError(
        "COMPAS data has no usable rows after filtering missing target/sensitive values."
    )
  return work_df, feature_columns, target_col, sensitive_col


def _compas_target_to_binary(values):
  """COMPAS target -> int32 0/1 with a fixed, order-independent mapping.

  Unlike ``_normalize_binary_series`` this gives the same encoding on any
  slice, so ``y_test`` can be re-encoded after a scenario flips labels and
  still agree with ``y_train``.
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
  return np.asarray(_normalize_binary_series(values, 'target'), dtype=np.int32)


def _compas_sensitive_to_binary(values):
  """1 = White/Caucasian, 0 = any other race."""
  values = pd.Series(values)
  if values.dtype.kind in {'O', 'U', 'S'}:
    normalized = values.astype(str).str.strip().str.lower()
    known_groups = {
        'white', 'caucasian', 'black', 'african-american',
        'other', 'asian', 'hispanic', 'native american',
    }
    if normalized.isin(known_groups).any():
      return normalized.isin({'white', 'caucasian'}).astype(np.int32).to_numpy()
    values = normalized
  return _normalize_binary_series(values, 'sensitive attribute')


def _compas_sex_to_binary(values):
  """1 = Female, 0 = Male (the reference subgroup in the COMPAS literature)."""
  series = pd.Series(values)
  if series.dtype.kind in {'O', 'U', 'S'}:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({'female', 'f', 'woman'}).astype(np.int32).to_numpy()
  return _normalize_binary_series(series, 'sex attribute')


def _encode_compas_features(work_df, target_col, transformer=None,
                            return_dataframe=False):
  """Encode COMPAS features. ``c_charge_desc`` is frequency-encoded (its
  one-hot would be ~400 sparse columns); categories seen < 10 times pool."""
  features = work_df.drop(columns=[target_col])
  if features.empty:
    raise ValueError("COMPAS data has no feature columns after dropping target column.")
  categorical = [
      c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])
  ]
  x, transformer = _encode_features(
      features, categorical, transformer,
      frequency_encode_columns=['c_charge_desc'],
      rare_category_min_count=10,
  )
  x = np.asarray(x, dtype=np.float32)
  if return_dataframe:
    x = pd.DataFrame(x, columns=transformer['feature_columns'], index=work_df.index)
  return x, transformer


def read_compas(path='data/compas/*', return_dataframe=False):
  """Legacy loader: encode the whole dataset, return (x, y, a).

  Not used by the pipeline -- use ``read_compas_train_test``, which splits
  before encoding so scenarios can be applied to the test slice.
  """
  work_df, feature_columns, target_col, sensitive_col = _read_compas_work_df(path)
  print(f"Using {len(feature_columns)} features: {feature_columns}")
  y = np.asarray(_normalize_binary_series(work_df[target_col], 'target'), dtype=np.int32)
  a = np.asarray(_compas_sensitive_to_binary(work_df[sensitive_col]), dtype=np.int32)
  x, _ = _encode_compas_features(work_df, target_col, return_dataframe=return_dataframe)
  return x, y, a


def _stratified_split_indices(n, stratify_options, test_size, seed):
  """Seeded split, stratified on the first valid option (each class >= 2)."""
  stratify = None
  for option in stratify_options:
    _, counts = np.unique(option, return_counts=True)
    if len(counts) >= 2 and np.all(counts >= 2):
      stratify = option
      break
  indices = np.arange(n)
  try:
    return train_test_split(indices, test_size=float(test_size),
                            random_state=int(seed), shuffle=True, stratify=stratify)
  except ValueError:
    return train_test_split(indices, test_size=float(test_size),
                            random_state=int(seed), shuffle=True, stratify=None)


def read_compas_train_test(
    scenario_name=None,
    path='data/compas/*',
    seed=42,
    test_size=0.3,
    intersectional=False,
):
  """COMPAS split by seed, drift scenario applied to the test slice only.

  Raw rows are split first, the scenario edits the raw test strings, then the
  encoder is fitted on train and reused on test -- no leakage, and edits to
  ``race`` / ``age_cat`` / ``c_charge_degree`` propagate through the one-hot
  columns. ``test_size=0.3`` (~2.1k rows) keeps each of the three phases in
  ``compas_scenarios.py`` above ADWIN's reliability floor.

  Stratified on joint (y, a), falling back to y, then unstratified; the same
  ``seed`` gives identical partitions across scenarios, so seeded
  comparisons stay paired.

  Returns:
    x_train, x_test, y_train, y_test, a_train, a_test, and with
    ``intersectional=True`` the composite ``a = 2*race + sex`` plus an
    ``a_marginals`` dict of the per-attribute arrays.
  """
  work_df, feature_columns, target_col, sensitive_col = _read_compas_work_df(path)
  print(f"Using {len(feature_columns)} features: {feature_columns}")

  y_full = _compas_target_to_binary(work_df[target_col])
  a_full = _compas_sensitive_to_binary(work_df[sensitive_col])
  joint = np.asarray([f'{int(y)}_{int(a)}' for y, a in zip(y_full, a_full)])
  train_idx, test_idx = _stratified_split_indices(
      len(work_df), [joint, y_full], test_size, seed,
  )
  train_df = work_df.iloc[train_idx].reset_index(drop=True)
  test_df = work_df.iloc[test_idx].reset_index(drop=True)

  scenario = (scenario_name or '').strip()
  if scenario and scenario != 'no_drift' and scenario in COMPAS_SCENARIOS:
    print(f"Applying COMPAS drift scenario: {scenario}")
    test_df = get_compas_scenario(scenario)(test_df, target_col=target_col)
  else:
    print("COMPAS: no drift applied (baseline)")

  # Train labels come from the pre-split encoding; test labels and groups are
  # re-encoded from the edited test rows so scenario label flips and race
  # swaps are honoured. Both encoders are order-independent, so they agree.
  y_train = np.asarray(y_full[train_idx], dtype=np.int32)
  y_test = _compas_target_to_binary(test_df[target_col])
  a_train = np.asarray(a_full[train_idx], dtype=np.int32)
  a_test = np.asarray(_compas_sensitive_to_binary(test_df[sensitive_col]), dtype=np.int32)

  x_train, transformer = _encode_compas_features(train_df, target_col)
  x_test, _ = _encode_compas_features(test_df, target_col, transformer)

  if not intersectional:
    return x_train, x_test, y_train, y_test, a_train, a_test

  if 'sex' not in train_df.columns:
    raise ValueError(
        "COMPAS intersectional mode requires a 'sex' column; got "
        f"{list(train_df.columns)}"
    )
  sex_train = _compas_sex_to_binary(train_df['sex'])
  sex_test = _compas_sex_to_binary(test_df['sex'])
  a_marginals = {
      'attr_names': ('race', 'sex'),
      'train': np.stack([a_train, sex_train], axis=1).astype(np.int32),
      'test': np.stack([a_test, sex_test], axis=1).astype(np.int32),
  }
  return (
      x_train, x_test, y_train, y_test,
      (2 * a_train + sex_train).astype(np.int32),
      (2 * a_test + sex_test).astype(np.int32),
      a_marginals,
  )


# ---------------------------------------------------------------------------
# Folktables (ACS Income)
# ---------------------------------------------------------------------------

_ACS_FEATURES = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
_ACS_CATEGORICAL = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']


def _adult_income_filter(df):
  """The ACSIncome filter that mimics the original Adult extraction."""
  return df[(df['AGEP'] > 16) & (df['PINCP'] > 100)
            & (df['WKHP'] > 0) & (df['PWGTP'] >= 1)]


def _normalize_sensitive_attribute(values, sensitive_attribute):
  values = np.asarray(values)
  if str(sensitive_attribute).lower() == 'race':
    return (values == 1).astype(np.int32)  # white (1) vs non-white (0)
  return (values - 1).astype(np.int32)     # SEX {1, 2} -> {0, 1}


def _preprocess_folktables_frame(frame, sensitive_attribute, transformer=None,
                                 intersectional=False):
  sensitive_column = 'RAC1P' if str(sensitive_attribute).lower() == 'race' else 'SEX'
  work = _adult_income_filter(frame).dropna(subset=_ACS_FEATURES + ['PINCP']).copy()

  y = (pd.to_numeric(work['PINCP'], errors='coerce') > 50000).astype(np.int32).to_numpy()
  a = _normalize_sensitive_attribute(
      pd.to_numeric(work[sensitive_column], errors='coerce').to_numpy(),
      sensitive_attribute,
  )
  x, transformer = _encode_features(work[_ACS_FEATURES], _ACS_CATEGORICAL, transformer)
  if not intersectional:
    return x, y, a, transformer, None

  # Columns [sex, race-binarised], for per-attribute diagnostic DP/EO.
  marginals = np.stack([
      _normalize_sensitive_attribute(pd.to_numeric(work['SEX'], errors='coerce').to_numpy(), 'sex'),
      _normalize_sensitive_attribute(pd.to_numeric(work['RAC1P'], errors='coerce').to_numpy(), 'race'),
  ], axis=1).astype(np.int32)
  return x, y, a, transformer, marginals


def read_folktables(path='data/acs-folktables', train_year=2015, test_years=(2017, 2018),
                    state='CA', horizon='1-Year', sensitive_attribute='sex',
                    download=True, intersectional=False):

  states = state if isinstance(state, (list, tuple)) else [state]
  states = [str(s).strip() for s in states if str(s).strip()] or ['CA']

  def _load(year):
    source = ACSDataSource(survey_year=int(year), horizon=horizon,
                           survey='person', root_dir=path)
    return source.get_data(states=states, download=download)

  x_train, y_train, a_train, transformer, m_train = _preprocess_folktables_frame(
      _load(train_year), sensitive_attribute, intersectional=intersectional,
  )

  test_years = tuple(int(year) for year in test_years)
  split_size = len(x_train) // max(len(test_years), 1)
  parts = []
  for year in test_years:
    x_t, y_t, a_t, _, m_t = _preprocess_folktables_frame(
        _load(year), sensitive_attribute, transformer, intersectional=intersectional,
    )
    parts.append((x_t[:split_size], y_t[:split_size], a_t[:split_size],
                  None if m_t is None else m_t[:split_size]))

  def _concat(i, dtype, empty_shape=(0,)):
    arrays = [p[i] for p in parts]
    return (np.concatenate(arrays, axis=0) if arrays
            else np.zeros(empty_shape, dtype=dtype)).astype(dtype)

  x_test = _concat(0, np.float32)
  y_test = _concat(1, np.int32)
  a_test = _concat(2, np.int32)
  x_train = np.asarray(x_train, dtype=np.float32)
  y_train = np.asarray(y_train, dtype=np.int32)

  if not intersectional:
    a_train = np.asarray(a_train, dtype=np.int32)
    return (x_train, x_test, y_train, y_test, a_train, a_test,
            len(set(a_train.tolist())))

  m_test = _concat(3, np.int32, empty_shape=(0, 2))
  a_train = (2 * m_train[:, 0] + m_train[:, 1]).astype(np.int32)
  a_test = (2 * m_test[:, 0] + m_test[:, 1]).astype(np.int32)
  a_marginals = {
      'attr_names': ('sex', 'race'),
      'train': m_train.astype(np.int32),
      'test': m_test,
  }
  return (x_train, x_test, y_train, y_test, a_train, a_test,
          len(set(a_train.tolist())), a_marginals)
