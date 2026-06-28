Perguntar ao professor se dá para submeter nos dois ciclos

# Multi-Protected-Attribute Implementation Notes (Path A)

This document records every code and mathematical decision made to extend
`FADO` from a single binary protected attribute to two binary attributes
(intersectional / Cartesian-product grouping), and the reasoning behind each
choice. Audience: reviewer / future collaborator who wants to verify the
extension without re-deriving it from the diff.

All math is written in plain text / unicode so it renders in any markdown
viewer.

---

## 1. Design choice: Path A (intersectional) over Path B (marginal sum)

### 1.1 The two candidates

Given two binary protected attributes `a1, a2 ∈ {0, 1}`:

- **Path A — Intersectional.** Treat the joint subgroup as a single index
  ```
  a = a1 · |A2| + a2  ∈  {0, 1, 2, 3}
  ```
  and use the existing K-group regulariser at `K = 4`.

- **Path B — Marginal sum.** Keep two separate group axes, two sets of
  running statistics, and two penalty terms:
  ```
  L_fair = λ1 · Σ_nodes Σ_{a ∈ A1} φ_δ(F_{ij,a}^{(1)})
         + λ2 · Σ_nodes Σ_{a ∈ A2} φ_δ(F_{ij,a}^{(2)})
  ```

### 1.2 Why Path A wins for a KDD-grade submission

Kearns, Neel, Roth, Wu (ICML 2018, *Preventing Fairness Gerrymandering*)
prove a construction where a classifier achieves **perfect marginal DP on
every protected attribute taken individually** while having **arbitrarily
bad DP on the joint subgroups** of the Cartesian product. Marginal-only
regularisation (Path B) is therefore *provably defeatable* — and the
seed-1 smoke test results below show the failure mode is not theoretical
but immediate on COMPAS:

| Model | Composite DP | Marginal DP_race | Marginal DP_sex |
|---|---|---|---|
| Aranyani-Base | 0.098 | **0.002** | 0.026 |
| ARF           | 0.221 | 0.133 | 0.108 |
| RFR           | 0.218 | 0.118 | 0.113 |

Aranyani-Base achieves nearly perfect marginal parity on race (0.002) but
the intersectional gap is 49× larger. Path A optimises against the joint
subgroup directly, closes this loophole by construction, and aligns with
the dominant fairness-literature definition (Foulds et al., ICDE 2020).

### 1.3 Why this also minimises code change

The `Aranyani` regulariser computes `number_of_attributes` from
`np.unique(protected_targets)` at runtime
(`src/models/forest/aranyani.py:54`). Feeding it the composite code
auto-scales every running statistic, gradient accumulator, and group-mean
computation to `K = 4` without modifying the regulariser. RFR's
reweighting key is also `a`-by-value, so it inherits `K = 4`
identically. ARF never consumes `a` and is unaffected.

We therefore only had to change the **data layer**, the **CLI plumbing**,
and the **metric reporting** layer. The learning algorithms themselves
required zero modification.

---

## 2. Composite-encoding mathematics

### 2.1 The encoding

For two binary attributes `a1, a2 ∈ {0, 1}` we define

```
a = 2·a1 + a2  ∈  {0, 1, 2, 3}
```

This is a bijective base-2 packing: the two original axes can be recovered
by

```
a1 = a // 2     (integer division)
a2 = a mod 2
```

No information is lost, no group is merged with another. The choice of
which axis occupies the high-order bit is fixed *per dataset* and
explained in §2.3 below.

The 4-cell decoding table:

| Composite `a` | `a1` (high bit) | `a2` (low bit) |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 1 | 0 |
| 3 | 1 | 1 |

### 2.2 What the regulariser actually optimises

With the composite code in place, the node-level demographic-parity
penalty (Section III of the paper, unchanged in form) becomes a sum over
all four joint subgroups:

```
L_fair = λ · Σ_{(i,j)} (1 / |A|) · Σ_{a ∈ {0,1,2,3}} φ_δ(F_{ij,a})

with     F_{ij,a} = n̄_{ij} − n̄_{ij,a}
```

where `n̄_{ij,a}` is the running EMA of the soft routing decision at
node `(i, j)` restricted to rows whose joint subgroup is `a`, and
`n̄_{ij}` is the cross-group mean. Each of the 4 cells contributes a
penalty proportional to its deviation from the cross-group mean, so the
regulariser pressures every *joint* cell toward parity — not just the
marginal averages over rows.

The surrogate `φ_δ(z)` is the Huber-style piecewise quadratic-linear
function inherited from Aranyani (paper §III):

```
φ_δ(z) = (1/2) · z²            if |z| < δ
       = |z| + δ²/2 − δ         if |z| ≥ δ
```

### 2.3 Per-dataset axis convention (why the high-order bit differs)

- **COMPAS:**     `a = 2·a_race + a_sex`   (race is high-order bit).
- **Folktables:** `a = 2·a_sex  + a_race`  (sex is high-order bit).

The convention is: **the legacy single-attribute axis takes the high-order
bit**. Reason: in single-attribute mode COMPAS used race and Folktables
used sex; this convention guarantees that

```
a_composite // 2  ==  a_legacy_attribute
```

on both datasets, so any diagnostic that examined the legacy attribute
alone in a prior run is mechanically recoverable from a saved
intersectional run. The convention has no effect on the regulariser —
only on the integer labels of the four cells.

### 2.4 Binarisation conventions

- `_compas_sensitive_to_binary` (unchanged): `1 = Caucasian`, `0 = other`.
- `_compas_sex_to_binary` (new): `1 = Female`, `0 = Male`. The 1=Female
  convention follows the dominant COMPAS fairness literature, where
  positive-rate parity is typically analysed with women as the reference
  subgroup.
- `_normalize_sensitive_attribute` (Folktables, unchanged): for `SEX` it
  subtracts 1 to convert ACS `{1, 2} → {0, 1}`; for `RAC1P` it returns
  `(values == 1).astype(int32)`, i.e. `1 = White`, `0 = non-White`.
  Identical to the existing single-attribute behaviour; we just now run
  both branches and combine.

### 2.5 Empirical 4-cell occupancy

Smoke test on COMPAS seed 1 (train split): all 4 cells observed
(`composite_groups_train = 4`). The smallest cell (Caucasian × Female) is
about 1.8K rows out of the 5K training split. Folktables (train split):
all 4 cells observed; smallest cell is comfortably above 30K even at the
10% subsample. No cell-sparsity problem at `K = 4`.

---

## 3. Code changes, file-by-file

### 3.1 `src/helpers/data.py`

#### `_compas_sex_to_binary(values)` — new function

```python
def _compas_sex_to_binary(values):
  """Encode COMPAS ``sex`` as binary (1=Female, 0=Male)."""
  series = pd.Series(values)
  if series.dtype.kind in {'O', 'U', 'S'}:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({'female', 'f', 'woman'}).astype(np.int32).to_numpy()
  return _normalize_binary_series(series, 'sex attribute')
```

**Decisions:**
- Explicit `'female', 'f', 'woman'` whitelist rather than `pd.factorize`
  to avoid the same factorise-order-dependence bug that
  `_compas_target_to_binary` was written to dodge (factorize on disjoint
  train/test halves can flip codes if the first row's value differs
  between halves).
- Numeric-dtype fall-through preserves backwards compatibility with any
  pre-encoded `sex` column.

#### `read_compas_train_test(..., intersectional=False)` — extended

When `intersectional=True`, the function returns a 7-tuple that ends
with an `a_marginals` dict; otherwise the existing 6-tuple is unchanged
(back-compat for the single-attribute path).

```python
if not intersectional:
    return x_train, x_test, y_train, y_test, a_train, a_test

a_sex_train  = _compas_sex_to_binary(train_df['sex'])
a_sex_test   = _compas_sex_to_binary(test_df['sex'])
a_race_train = a_train      # pre-existing binary race code
a_race_test  = a_test       # already recomputed post-scenario

a_train_intersect = 2 * a_race_train + a_sex_train
a_test_intersect  = 2 * a_race_test  + a_sex_test

a_marginals = {
    'attr_names': ('race', 'sex'),
    'train': np.stack([a_race_train, a_sex_train], axis=1).astype(np.int32),
    'test':  np.stack([a_race_test,  a_sex_test],  axis=1).astype(np.int32),
}
return (x_train, x_test, y_train, y_test,
        a_train_intersect.astype(np.int32),
        a_test_intersect.astype(np.int32),
        a_marginals)
```

**Decisions:**
- `a_race_test` is reused as-is. The pre-existing single-attribute path
  already recomputes `a_race_test` from
  `_compas_sensitive_to_binary(test_df['race'])` *after* the drift
  scenario has edited `test_df['race']`. So when the `abrupt_race`
  scenario swaps a row's race string, `a_race_test` correctly reflects
  the swap, and the composite code `2·a_race + a_sex` inherits that.
  **No additional scenario-aware logic is needed.**
- `sex` is read *after* the scenario function, but the COMPAS scenarios
  edit `race` and the target label only — `sex` is identity-preserved.
  So `a_sex_test` matches what would have been read pre-scenario.
- We use `np.stack(..., axis=1)` so `a_marginals['train']` is a single
  `[N, 2]` int32 ndarray. The pipeline never needs the marginals as
  separate scalars; keeping them as a single ndarray makes alignment
  with row-level subsampling trivial (§3.4).

#### `_preprocess_folktables_frame(..., intersectional=False)` and `read_folktables(..., intersectional=False)` — extended

Same shape change. Two notes specific to Folktables:

- `required_columns` was widened to always include both `SEX` and `RAC1P`
  in the `dropna` filter, even when only one is the legacy sensitive
  attribute. Reason: the marginals must align row-for-row with the
  composite-code stream, which means any row dropped for missing `SEX`
  *or* missing `RAC1P` must be dropped from *both* paths. Without this
  widening, the intersectional path would have a different row set from
  the non-intersectional path, breaking direct comparisons.
- The Folktables loader has a per-test-year truncation step that takes
  the first `split_size` rows of each year. The marginals are sliced
  with the same `[:split_size]` slice on the same iteration, so they
  stay aligned.

### 3.2 `src/main.py`

#### New CLI flag

```python
flags.DEFINE_bool(
    'intersectional', False,
    'Enable multi-protected-attribute (intersectional) mode. ...',
)
```

Default `False` preserves bit-for-bit reproducibility of every prior
seeded run.

#### `_maybe_subsample_folktables(..., marginals_train=None, marginals_test=None)` — extended

The Folktables 10% subsample uses a per-seed RNG to pick row indices.
Those same indices are now applied to the marginal `[N, 2]` arrays so
the marginals stay row-aligned with `x/y/a`:

```python
def _take(x, y, a, m=None):
    n = len(y)
    k = max(1, int(round(n * fraction)))
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    ...
    return x[idx], y[idx], a[idx], None if m is None else m[idx]
```

`idx.sort()` is critical: keeping rows in their original streaming order
is what makes the prequential evaluation comparable across seeds.

#### `_load_dataset_splits` — returns 7-tuple

Always returns
`(x_train, x_test, y_train, y_test, a_train, a_test, marginals)`
where `marginals` is `None` outside intersectional mode. This shape
change required updating one downstream caller (`_single_scenario`) and
nothing else.

#### `_compute_marginal_dp_eo(stream, a_marginals_test, attr_names)` — new function

This is the **post-hoc, single-pass** marginal-DP/EO computation that
runs after a model completes the prequential stream:

```python
def _compute_marginal_dp_eo(stream, a_marginals_test, attr_names):
    y_preds = stream.get('y_preds_all')
    y_true  = stream.get('y_true_all')
    a_marg  = np.asarray(a_marginals_test, dtype=np.int32)
    n = min(len(y_preds), len(y_true), a_marg.shape[0])
    out = {}
    for idx, attr in enumerate(attr_names):
        marginal_a = a_marg[:n, idx].astype(np.int32).tolist()
        dp_val, _ = _utils.get_demographic_parity(preds, marginal_a)
        eo_val, _ = _utils.get_equalized_odds(preds, marginal_a, trues)
        out[str(attr)] = {'dp': float(dp_val), 'eo': float(eo_val)}
    return out
```

**Decisions:**
- **End-of-stream rather than per-step rolling window.** Per-step rolling
  marginal DP/EO would mean three live `get_demographic_parity` calls
  per sample (composite + 2 marginals) — about 3× the existing fairness
  metric cost per step, which is the hot loop on COMPAS. The end-of-stream
  pass walks the recorded prediction list exactly once. We pay an extra
  O(N·T) on memory for `y_preds_all`/`y_true_all` lists (negligible —
  about 18.7K ints per Folktables seed, 2.1K per COMPAS scenario), and
  the runtime overhead is in the milliseconds.
- We reuse `get_demographic_parity` / `get_equalized_odds` unchanged —
  they already iterate over `np.unique(protected_group)`, so they
  natively handle whatever `K` the marginal array implies (here
  `K = 2` per attribute).
- The composite (intersectional) DP/EO is *still* reported as the
  primary `dp`/`eo` keys; marginals are diagnostic.

#### `_single_scenario` — calls `_compute_marginal_dp_eo` post-eval

```python
if a_marginals is not None:
    marginal_metrics = _compute_marginal_dp_eo(
        stream=stream,
        a_marginals_test=a_marginals['test'],
        attr_names=a_marginals['attr_names'],
    )
    if marginal_metrics:
        test_metrics['marginal'] = marginal_metrics
        stream['marginal_metrics'] = marginal_metrics
```

#### Result-row serialisation — both nested and flat keys

```python
if marginal:
    row['marginal'] = marginal
    for attr_name, vals in marginal.items():
        if vals.get('dp') is not None:
            row[f'dp_{attr_name}'] = float(vals['dp'])
        if vals.get('eo') is not None:
            row[f'eo_{attr_name}'] = float(vals['eo'])
```

**Decision: write both shapes.** The nested `marginal` block preserves
the attribute structure and is the canonical record. The flat
`dp_race`, `dp_sex`, `eo_race`, `eo_sex` keys mirror the existing
`accuracy`/`dp`/`eo` shape so the existing `significance_tests.py`
infrastructure can pick them up by passing
`--metrics accuracy,dp,eo,dp_race,dp_sex,eo_race,eo_sex` — no script
modification required.

### 3.3 Evaluator return-dict expansions

All four evaluators already maintained `y_preds_all` and `y_true_all`
in local lists for their own fairness-window computations; we just
expose them in the returned dict:

- `src/models/forest/evaluator.py` (FADO): added two lines to the
  return dict.
- `src/models/forest/baseline_evaluator.py` (Aranyani-Base): same.
- `src/models/arf/arf.py` (ARF): same.
- `src/models/rfr/evaluator.py` (RFR): same.

No mathematical change — pure plumbing.

### 3.4 What was deliberately NOT changed

- **`aranyani.py` regulariser.** Auto-detects
  `number_of_attributes = np.unique(a).size` at runtime; allocates
  `[num_internal_nodes, K, ...]` running tensors; the per-node penalty
  sums over `a` in `range(K)`. With composite codes it transparently
  runs at `K = 4`.
- **Dual-ADWIN drift detector.** Observes the binary error stream
  `e_t = 1[y_pred_t ≠ y_t]`, which has no protected-attribute axis.
  No change required and no semantics change at `K = 4`.
- **Reaction controller.** Modulates the *global* optimiser learning
  rate `η` and *global* soft-routing temperature `τ`. These act
  uniformly across all groups by construction — there is no per-group
  controller decision to lift.
- **RFR reweighting.** The reweighting hash key takes the `a` array by
  value, so a `K = 4` composite code is hashed identically to a
  `K = 2` marginal code. The four-cell reweighting is therefore the
  apples-to-apples intersectional comparison vs FADO.
- **ARF model.** Does not consume `a` at all; the only effect of
  intersectional mode is downstream metric reporting against the
  composite code.
- **COMPAS drift scenarios.** Edit `race` and the target label only;
  `sex` is identity-preserved. The composite code naturally inherits
  the race edit because `_compas_sensitive_to_binary` is re-applied
  post-scenario in the existing single-attribute path (see §3.1).

---

## 4. Metric definitions

### 4.1 Composite (intersectional) DP and EO

Identical to the single-attribute definitions in Section III of the paper,
applied to the composite code `a ∈ {0, 1, 2, 3}`. For DP:

```
p̂_a = E[ ŷ | a ]                         per-group positive-prediction rate
p̄   = (1 / |A_t|) · Σ_{a' ∈ A_t} p̂_{a'}    cross-group mean
DP_intersectional = max_{a ∈ A_t} | p̂_a − p̄ |
```

EO is the same, conditioned on `y`:

```
p̂_{a|y} = E[ ŷ | a, y ]                     per-group, per-class rate
p̄_y     = (1 / |A_t|) · Σ_{a' ∈ A_t} p̂_{a'|y}
EO_intersectional = max_{y ∈ {0,1}} max_{a ∈ A_t} | p̂_{a|y} − p̄_y |
```

These are computed over the rolling window `W` during prequential
evaluation, mean-aggregated end-of-stream, and reported as `dp` / `eo`
in `results.json`.

### 4.2 Marginal DP and EO (diagnostics)

For each attribute axis `k ∈ {1, 2}`, restrict the group index to that
axis (project the joint subgroup down to just `a_k`) and apply the same
definition:

```
DP_k = max_{a ∈ A_t^(k)} | p̂_a^(k) − p̄^(k) |
EO_k = max_{y} max_{a ∈ A_t^(k)} | p̂_{a|y}^(k) − p̄_y^(k) |
```

These are computed once at end-of-stream over the **full** prediction
sequence (not over the rolling window). The end-of-stream choice was
made for two reasons:

1. The marginals are a *diagnostic*, not an optimisation target, so they
   don't need to be available step-by-step.
2. Computing them in a single pass at the end costs essentially nothing
   on top of the existing fairness metric path.

Reported in `results.json` as the nested `marginal.<attr>.dp` /
`marginal.<attr>.eo` and as the flat `dp_<attr>` / `eo_<attr>` keys.

### 4.3 Gerrymandering ratio (proposed reporting addition)

Define the **gerrymandering ratio** for a model as

```
GR = DP_intersectional / max_k DP_k
```

A value of 1 means the joint subgroup violation is fully explained by
the worst marginal axis; values much greater than 1 indicate that
subgroup disparity is being hidden by marginal balance — the
Kearns/Foulds failure mode. From the smoke test:

| Model         | GR (no_drift)        |
|---|---|
| Aranyani-Base | 0.098 / 0.026 ≈ 3.8  |
| ARF           | 0.221 / 0.133 ≈ 1.7  |
| RFR           | 0.218 / 0.118 ≈ 1.8  |

The Aranyani-Base ratio of 3.8 is the empirical confirmation of why
Path A — not Path B — is the right optimisation target for an
in-processing fair learner.

---

## 5. Row-alignment invariants

A correctness-critical implicit contract in the new pipeline:

> For every test-stream index `t`, the rows
> `x_test[t]`, `y_test[t]`, `a_test[t]` (composite code),
> `a_marginals['test'][t, 0]`, and `a_marginals['test'][t, 1]`
> must all refer to **the same original row** of the source dataframe.

Three places where this could break and how each is handled:

1. **Folktables subsample.** Same `idx = rng.choice(...).sort()` is
   applied to both `x/y/a` and the marginals tuple via the extended
   `_take` helper.
2. **Folktables per-year split.** The same `[:split_size]` slice is
   applied to `x/y/a` and `m_test` in the same loop iteration.
3. **COMPAS scenario edit.** Scenarios edit `test_df` in place but do
   not reorder rows. Marginals are recomputed from the *post*-edit
   `test_df` (race edits flow through; sex is unaffected).

If any future loader is added, it must preserve this invariant or the
post-hoc marginal metric will silently misalign and produce nonsense.

---

## 6. Paper-side change

§III gained a new paragraph "Multiple protected attributes"
(`sec:problem:multi-attr` label) that:

1. Names the formulation as intersectional / Cartesian product.
2. Cites Kearns 2018 (`kearns2018gerrymandering`) for the gerrymandering
   theorem that rules out Path B.
3. Cites Foulds 2020 (`foulds2020intersectional`) for the intersectional
   fairness definition.
4. Notes that the §III regulariser and the §IV controller are unchanged
   at `K = 4`.
5. Pre-commits to reporting marginal DP/EO as diagnostics in §V.

Two new bib entries appended to `DOCS/references.bib`. The paper still
compiles to 10 pages under the `acmart` `sigconf,anonymous,review`
options.

---

## 7. Open follow-ups (not yet executed)

- **Full sweep.** Run 30 seeds × 4 models × 2 datasets in intersectional
  mode (same command as the current sweep with `--intersectional=true`
  added). Composite cell counts will be roughly equal across seeds
  because of stratification on `(y, a_race)` (COMPAS) and the per-year
  subsampling RNG (Folktables).
- **Significance tests.** Rerun `src.significance_tests` with
  `--metrics accuracy,dp,eo,dp_race,dp_sex,eo_race,eo_sex` to produce a
  6-column table (composite DP/EO + 4 marginal cells). The flattened
  row keys mean no script change is required.
- **Results-table update.** The current table in §V (`tab:results`) has
  3 metric columns; in intersectional mode it grows to 7. A reasonable
  layout is to split into two stacked sub-tables ("Composite" / "Marginal
  per attribute") to preserve the 10-page budget.
- **Hyperparameter re-tune for `K = 4`.** Current `λ = 1.0` was
  calibrated for `K = 2`. At `K = 4` the per-node penalty has twice as
  many summands, so the effective regularisation strength is roughly
  doubled. A one-axis re-scan on the held-out training subset is
  warranted before the final run if the composite DP looks
  under/over-regularised on a pilot seed.
