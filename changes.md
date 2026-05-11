# Preprocessing and model-input fixes

## Why this was needed
- The tabular preprocessing was fitting encoders/scalers independently per split.
- Categorical columns were ordinal label-encoded, which injected artificial ordering.
- The forest training code used hardcoded `data_dim` values, which break when feature dimensions change after safer encoding.

## What changed
1. Added shared tabular preprocessing helpers in `src/helpers/data.py`:
   - `_fit_tabular_transformer(...)`
   - `_transform_tabular_features(...)`
   - These now:
     - impute numeric values with train medians
     - standardize numeric features with train mean/std
     - one-hot encode categorical values
     - align test/inference features to train-time columns

2. Adult dataset preprocessing was rewritten to be split-consistent:
   - Added `_preprocess_adult_frame(...)` with a reusable fitted transformer.
   - `read_adult(...)` now fits on train and applies the same transformer to test.
   - `load_drifted_test_set(...)` now fits from `adult.data` and applies that transformer to the drifted test.
   - Added robust binary parsing for Adult target/sensitive fields:
     - `_adult_income_to_binary(...)`
     - `_adult_gender_to_binary(...)`

3. Census preprocessing was updated:
   - Added `_preprocess_census_frame(...)` using the same fit/transform strategy.
   - `preprocess_census(...)` now uses one-hot + train-style numeric standardization.

4. COMPAS preprocessing was updated:
   - Replaced per-column `LabelEncoder` with one-hot + numeric standardization through shared helpers.

5. Diabetes preprocessing was updated:
   - `_preprocess_diabetes_frame(...)` now accepts a fitted transformer.
   - `read_diabetes(...)` now fits on train and transforms test with the same feature mapping.

6. Forest training input dimensions were made dynamic in `src/models/forest/train.py`:
   - Replaced hardcoded `data_dim` assignments with `x_train.shape[1]` across dataset branches.
   - This keeps model construction consistent with new encoded feature widths.

## Result
- Tabular pipelines now avoid train/test preprocessing drift.
- Categorical handling is no longer ordinal-by-accident.
- The forest model input dimension now matches real feature outputs automatically.
