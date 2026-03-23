#!/usr/bin/env python
# coding: utf-8
# %%

# %%


"""Dataloaders."""


# %%


import os


# %%

import pickle
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSPublicCoverage

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


def read_compas(path="../data/compas"):

  feature_names = [
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
          "is_recid"]
  categorical_features = ["c_charge_degree",
          "c_charge_desc",
          "age_cat",
          "sex",
          "race",
          "is_recid"]
  train_df = pd.read_csv(os.path.join(path, "train.csv"),
                         names=feature_names, header=None)
  test_df = pd.read_csv(os.path.join(path, "test.csv"),
                        names=feature_names, header=None)


  df = pd.concat([train_df, test_df])
  df = df.dropna()

  mapping = {'White': 1, 'Black': 0, 'Other': 0}
  df['race'] = df['race'].map(mapping)

  label_encoder = preprocessing.LabelEncoder()
  for feature in categorical_features:
      df[feature] = label_encoder.fit_transform(df[feature])

  Y = np.array(df['is_recid'], dtype=np.float32)
  A = np.array(df['race'], dtype=np.float32)

  df = df.drop('is_recid', axis=1)
  X = np.array(df.values, dtype=np.float32)

  return X, Y, A


# CelebA

def load_dump(file_path):
  with open(file_path, "rb") as f:
    data = pickle.load(f)
  return data


def read_celeba(path="../data/celeba/"):
  X, Y, A = load_dump(os.path.join(path, "clip_data.pkl"))
  x_train, y_train, a_train = X, Y[: len(X)], A[: len(X)]
  return x_train, y_train, a_train


def read_folktables(path='data', train_year=2014, test_years=(2015, 2016, 2017, 2018),
                    state='CA', horizon='1-Year'):
  """Read Folktables ACS Public Coverage data from local cache."""

  train_source = ACSDataSource(
      survey_year=train_year, horizon=horizon, survey='person', root_dir=path
  )
  train_df = train_source.get_data(states=[state], download=False)
  x_train, y_train, a_train = ACSPublicCoverage.df_to_numpy(train_df)

  x_tests, y_tests, a_tests = [], [], []
  for year in test_years:
    test_source = ACSDataSource(
        survey_year=year, horizon=horizon, survey='person', root_dir=path
    )
    test_df = test_source.get_data(states=[state], download=False)
    x_t, y_t, a_t = ACSPublicCoverage.df_to_numpy(test_df)
    x_tests.append(x_t)
    y_tests.append(y_t)
    a_tests.append(a_t)

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
  )
