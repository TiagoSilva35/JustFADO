# FADO: A Reaction Controller for Fairness-Aware Online Learning Under Concept Drift

This repository contains the implementation, configuration, and per-seed
result files for the paper *FADO: A Reaction Controller for Fairness-Aware
Online Learning Under Concept Drift*.

`FADO` wraps the `Aranyani` fair oblique decision forest
(<https://github.com/brcsomnath/Aranyani>, MIT-licensed) with a dual-threshold
ADWIN drift detector and a reaction controller that modulates the optimiser
learning rate and the soft-routing temperature on confirmed drift.

---

## Setup

```bash
git clone https://github.com/TiagoSilva35/Aranyani2.0.git
cd Aranyani2.0
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Datasets:

* **COMPAS** — `data/compas/*.csv` (included).
* **Folktables (ACS Income, CA)** — downloaded automatically by the
  `folktables` library on first run; cached locally.

---

## Reported results

Mean ± std over 30 random seeds, paired Wilcoxon Holm-corrected
significance vs every baseline on the **FADO** row
(`*` p<0.05, `**` p<0.01, `***` p<0.001).
**Bold** marks the best mean per metric within each dataset.

| Dataset    | Model        | Accuracy ↑          | DP ↓                  | EO ↓                  |
|------------|--------------|---------------------|-----------------------|-----------------------|
| Folktables | **FADO**     | 0.8033 ± 0.0053     | **0.0284 ± 0.0041**\*\* | 0.0281 ± 0.0027       |
| Folktables | Aranyani     | **0.8053 ± 0.0049** | 0.0299 ± 0.0032       | **0.0257 ± 0.0024**   |
| Folktables | ARF          | 0.7447 ± 0.0104     | 0.0722 ± 0.0108       | 0.0743 ± 0.0153       |
| Folktables | RFR          | 0.7780 ± 0.0047     | 0.0733 ± 0.0063       | 0.0586 ± 0.0065       |
| COMPAS     | **FADO**     | 0.6434 ± 0.0105     | **0.0611 ± 0.0094**\*\*\* | **0.0923 ± 0.0130**\*\* |
| COMPAS     | Aranyani     | **0.6451 ± 0.0100** | 0.0679 ± 0.0065       | 0.0979 ± 0.0104       |
| COMPAS     | ARF          | 0.6313 ± 0.0133     | 0.0808 ± 0.0183       | 0.1053 ± 0.0234       |
| COMPAS     | RFR          | 0.6404 ± 0.0154     | 0.0813 ± 0.0160       | 0.1067 ± 0.0226       |

(`Aranyani` here is the controller-free ablation: the bare `Aranyani` base
learner run prequentially under the same protocol, with the dual-ADWIN
detector and reaction controller disabled. It shares the seed-pinned data
ordering with `FADO`, so the FADO-vs-Aranyani gap is attributable to the
controller alone.)

---

## Reproducing the results

Each command below regenerates the per-seed `results.json` files under
`files/experiments/dataset_<name>/model_<name>/seed_<N>/results.json` for
all four models on the 30-seed sweep.

### COMPAS (30 seeds × 4 models × 2 scenarios)

```bash
python -m src.main \
    --dataset compas \
    --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
```

The dataset-specific hyperparameter overrides for COMPAS
(`adwin_delta_confirm = 0.2`, `fairness_window = 250`,
`lr_decay_steps = 600`, `min_samples_per_stream = 20`,
`lambda_const = 1.0`) are applied automatically by `_COMPAS_FADO_OVERRIDES`
in `src/main.py` when `--dataset compas` is selected. No CLI flag needed.

### Folktables (30 seeds × 4 models, 10% uniform subsample per seed)

```bash
python -m src.main \
    --dataset folktables \
    --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 \
    --folktables_subsample_fraction 0.10 \
    --lambda_const 1.0
```

`--folktables_subsample_fraction 0.10` slims each split from ~187k to
~18.7k samples per seed, drawn deterministically from the per-seed RNG so
all four models within a seed see identical rows (paired tests remain
valid). Removing the flag runs on the full splits.

---

## Reproducing the significance tests

Once the per-seed result files exist, the paired Wilcoxon + Welch tests
(both with Holm--Bonferroni correction across baseline comparisons within
each metric column) are computed by `src.significance_tests`. It reads the
per-seed `results.json` files directly:

```bash
# COMPAS — reproduces the COMPAS rows of the table above
python -m src.significance_tests \
    --inputs files/experiments/dataset_compas \
    --reference aranyani \
    --baselines aranyani_base,arf,rfr \
    --metrics accuracy,dp,eo

# Folktables — reproduces the Folktables rows of the table above
python -m src.significance_tests \
    --inputs files/experiments/dataset_folktables \
    --reference aranyani \
    --baselines aranyani_base,arf,rfr \
    --metrics accuracy,dp,eo
```

The script restricts to scenarios common to every model under comparison
(so the paired test is apples-to-apples across models with different
scenario coverage), prints both the paired Wilcoxon and the Welch t-test
side by side, and reports the Holm-corrected p-value and the conventional
`*` / `**` / `***` marker for each comparison.

---

## Summary aggregation

If you only want the mean ± std table (no significance tests),
`src.extract_seed_metrics` aggregates the per-seed JSON files:

```bash
python -m src.extract_seed_metrics \
    --inputs files/experiments/dataset_folktables \
    --format summary --include-average
python -m src.extract_seed_metrics \
    --inputs files/experiments/dataset_compas \
    --format summary --include-average
```

---

## Repository layout

* `src/main.py` — entry point; orchestrates per-seed sweeps across the
  four models (`aranyani` = FADO, `aranyani_base` = controller-free
  ablation, `arf`, `rfr`).
* `src/models/forest/evaluator.py` — FADO prequential evaluator
  (Aranyani + dual-ADWIN + reaction controller).
* `src/models/forest/baseline_evaluator.py` — controller-free Aranyani
  prequential evaluator.
* `src/models/arf/`, `src/models/rfr/` — streaming baselines.
* `src/drift/compas_scenarios.py` — the two active COMPAS drift scenarios.
* `src/significance_tests.py` — paired Wilcoxon + Welch with Holm
  correction, reads per-seed result JSONs.
* `src/extract_seed_metrics.py` — mean ± std aggregator.
* `files/experiments/dataset_<name>/` — per-seed result outputs from the
  reported sweeps.

---

## Acknowledgements

This project is based on code from
<https://github.com/brcsomnath/Aranyani>, licensed under the MIT License.
