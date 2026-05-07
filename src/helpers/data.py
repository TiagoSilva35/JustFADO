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

from sklearn import preprocessing
from src.drift.create_drifted_ds import generate_drifted_dataset
from src.drift.scenarios import get_scenario, SCENARIOS


# %%


IM_WIDTH = IM_HEIGHT = 160


# %%


def preprocess_adult(df):
  """Pre-process the Adult dataset.

  Args:
    df: pandas data frame.

  Returns:
  """
  df = df.dropna()

  # Here we apply discretisation on column marital_status
  df.replace(
      [
          'Divorced',
          'Married-AF-spouse',
          'Married-civ-spouse',
          'Married-spouse-absent',
          'Never-married',
          'Separated',
          'Widowed',
      ],
      [
          'not married',
          'married',
          'married',
          'married',
          'not married',
          'not married',
          'not married',
      ],
      inplace=True,
  )

  label_encoder = preprocessing.LabelEncoder()

  # Perform one-hot encoding on categorical features
  categorical_features = [
      'workclass',
      'education',
      'marital-status',
      'occupation',
      'relationship',
      'race',
      'gender',
      'native-country',
      'income',
  ]

  for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

  # Split the dataset into features and target variable
  data = df.drop('income', axis=1)

  for n in data.columns:
    data[n] = (data[n] - data[n].mean()) / data[n].std()

  x = np.array(data.values, dtype=np.float32)

  y = np.array(df['income'], dtype=np.int32)
  a = np.array(df['gender'], dtype=np.int32)

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


  x_train, y_train, a_train = preprocess_adult(train_df)

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

    x_test, y_test, a_test = preprocess_adult(test_df)
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
  with open(test_file_path, 'rb') as f:
    df = pd.read_csv(f, names=columns, skiprows=1)

  if scenario_name and scenario_name != 'no_drift':
    scenario_fn = get_scenario(scenario_name)
    print(f"Applying drift scenario: {scenario_name}")
    df = scenario_fn(df)
  else:
    print("No drift applied (baseline)")

  x_test, y_test, a_test = preprocess_adult(df)
  return x_test, y_test, a_test


# %%


def preprocess_census(df):
  """Pre-process the Census dataset.

  Args:
    df:

  Returns:
  """
  df.dropna(inplace=True)

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
      'income_50k',
  ]
  for feature in categorical_features:
    label_encoder = preprocessing.LabelEncoder()
    df[feature] = label_encoder.fit_transform(df[feature])

  y = np.array(df['income_50k'], dtype=np.int32)
  a = np.array(df['sex'], dtype=np.int32)
  df.drop(columns=['income_50k'], inplace=True)
  df.drop(columns=['unk'], inplace=True)

  for n in df.columns:
    df[n] = (df[n] - df[n].mean()) / df[n].std()

  x = np.array(df.values, dtype=np.float32)
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


def read_compas(path="data/compas/*"):
  target_candidates = [
      'two_year_recid', 'is_recid', 'recid', 'label', 'target', 'y',
  ]
  sensitive_candidates = [
      'race', 'ethnicity', 'sensitive', 'sensitive_attribute', 'group', 'a',
  ]
  legacy_feature_names = [
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
  preferred_feature_columns = [
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

  def _has_any_column(frame, candidates):
    columns = {str(column).strip().lower() for column in frame.columns}
    return any(candidate in columns for candidate in candidates)

  files = _resolve_compas_files(path)
  frames = []
  for file_path in files:
    frame = pd.read_csv(file_path)
    if not (_has_any_column(frame, target_candidates) and _has_any_column(frame, sensitive_candidates)):
      frame = pd.read_csv(file_path, names=legacy_feature_names, header=None)
    frames.append(frame)

  df = pd.concat(frames, ignore_index=True).copy()

  target_col = _resolve_column_name(df, target_candidates, 'target')
  sensitive_col = _resolve_column_name(df, sensitive_candidates, 'sensitive attribute')

  present_preferred = [column for column in preferred_feature_columns if column in df.columns]
  if len(present_preferred) >= 3:
    feature_columns = list(dict.fromkeys(present_preferred))
  else:
    feature_columns = [
        column for column in df.columns
        if column != target_col and str(column).strip().lower() not in set(target_candidates)
    ]

  required_columns = list(dict.fromkeys(feature_columns + [target_col, sensitive_col]))
  work_df = df[required_columns].copy()
  work_df = work_df.dropna(subset=[target_col, sensitive_col]).copy()
  if work_df.empty:
    raise ValueError(
        "COMPAS data has no usable rows after filtering missing target/sensitive values."
    )

  target_values = work_df[target_col]
  y = _normalize_binary_series(target_values, 'target')

  sensitive_values = work_df[sensitive_col]
  if sensitive_values.dtype.kind in {'O', 'U', 'S'}:
    normalized_sensitive = sensitive_values.astype(str).str.strip().str.lower()
    known_groups = {
        'white', 'caucasian', 'black', 'african-american',
        'other', 'asian', 'hispanic', 'native american',
    }
    if normalized_sensitive.isin(known_groups).any():
      a = normalized_sensitive.isin({'white', 'caucasian'}).astype(np.int32).to_numpy()
    else:
      a = _normalize_binary_series(normalized_sensitive, 'sensitive attribute')
  else:
    a = _normalize_binary_series(sensitive_values, 'sensitive attribute')

  features = work_df.drop(columns=[target_col]).copy()
  if features.empty:
    raise ValueError("COMPAS data has no feature columns after dropping target column.")

  for column in features.columns:
    if not pd.api.types.is_numeric_dtype(features[column]):
      encoder = preprocessing.LabelEncoder()
      features[column] = encoder.fit_transform(features[column].astype(str))
    features[column] = pd.to_numeric(features[column], errors='coerce')

  features = features.fillna(features.median(numeric_only=True))
  features = features.fillna(0.0)
  for column in features.columns:
    std = features[column].std()
    if std and std > 0:
      features[column] = (features[column] - features[column].mean()) / std

  x = np.asarray(features.values, dtype=np.float32)
  return x, np.asarray(y, dtype=np.int32), np.asarray(a, dtype=np.int32)


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


def _preprocess_diabetes_frame(df):
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

  for column in features.columns:
    if not pd.api.types.is_numeric_dtype(features[column]):
      encoder = preprocessing.LabelEncoder()
      features[column] = encoder.fit_transform(features[column].astype(str))
    features[column] = pd.to_numeric(features[column], errors='coerce')

  features = features.fillna(features.median(numeric_only=True))
  features = features.fillna(0.0)
  for column in features.columns:
    std = features[column].std()
    if std and std > 0:
      features[column] = (features[column] - features[column].mean()) / std

  x = np.asarray(features.values, dtype=np.float32)
  return x, y, a


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

  x_train, y_train, a_train = _preprocess_diabetes_frame(train_df)
  x_test, y_test, a_test = _preprocess_diabetes_frame(test_df)
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

  income_problem = _acs_income_problem(sensitive_attribute=sensitive_attribute)

  train_source = ACSDataSource(
      survey_year=train_year, horizon=horizon, survey='person', root_dir=path
  )
  train_df = train_source.get_data(states=states, download=download)
  x_train, y_train, a_train = income_problem.df_to_numpy(train_df)
  a_train = _normalize_sensitive_attribute(a_train, sensitive_attribute)
  x_tests, y_tests, a_tests = [], [], []
  for year in test_years:
    test_source = ACSDataSource(
        survey_year=year, horizon=horizon, survey='person', root_dir=path
    )
    test_df = test_source.get_data(states=states, download=download)
    x_t, y_t, a_t = income_problem.df_to_numpy(test_df)
    a_t = _normalize_sensitive_attribute(a_t, sensitive_attribute)
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
