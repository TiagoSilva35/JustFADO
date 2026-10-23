# Fairness-Aware Online Learning Under Concept Drift With an Adaptive Fairness Constraint

This repository contains the implementation, configuration files, seed
manifests, and per-seed result files for the paper *Fairness-Aware Online
Learning Under Concept Drift With an Adaptive Fairness Constraint*. It
reproduces Table III of the paper and lists, for each dataset, the exact
commands that produced every reported row: the per-seed sweep, the paired
Wilcoxon and Welch significance run, and the summary aggregation.

`FADO` (Fairness-Aware Drift-Aware Online Learning) builds on the `Aranyani`
fair oblique decision forest (<https://github.com/brcsomnath/Aranyani>,
MIT-licensed) and adds three mechanisms:

* an **adaptive fairness constraint**: the fixed fairness weight of Aranyani
  becomes the dual price `λ_t` of a demographic-parity bound `DP ≤ ε`, tracked
  online by projected stochastic dual ascent on an unbiased per-sample
  Horvitz–Thompson estimate of each group's disparity;
* **drift detection with forgetting**: a dual-threshold ADWIN detector on the
  error stream and a per-group fairness ADWIN on the disparity signal, both
  resetting the fairness penalty's running statistics on a detected change;
* a **reaction controller** that pre-warms or spikes the learning rate and
  sharpens a routing temperature (added by FADO; `τ = 1` recovers Aranyani)
  on drift, annealing both back during recovery.

---

## Setup

Python 3.11 or newer, and Java 17 or newer on the `PATH` (CapyMOA, used for
the injected drifts and its detectors; check with `java -version`).

```bash
git clone https://github.com/TiagoSilva35/JustFADO.git
cd JustFADO
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
wandb login                     # the reported sweeps log to W&B
```

Datasets:

* **Adult** — `data/adult/adult.data` and `adult.test` (included). The test
  split is the stream; one of four drifts is injected into it per run.
* **Folktables (ACS Income, CA)** — read from `data/acs-folktables/` if
  present, otherwise downloaded by the `folktables` library on the first run.
  Training is 2015; the stream is 2017 then 2018.

---

## Reported results (Table III)

Whole-stream prequential metrics, mean ± std over 30 seeds. Superscripts on
the FADO row mark a paired Wilcoxon test, Holm-corrected across the three
baselines, significant against **every** baseline in FADO's favor
(`*` p<0.05, `**` p<0.01, `***` p<0.001). **Bold** is the best mean per
metric within each block.

| Data / drift              | Model    | Accuracy ↑        | DP ↓                    | EO ↓                    |
|---------------------------|----------|-------------------|-------------------------|-------------------------|
| Folktables, natural       | FADO     | 0.805 ± 0.003     | **0.042 ± 0.003**\*\*\* | **0.025 ± 0.003**\*\*\* |
|                           | Aranyani | **0.808 ± 0.003** | 0.057 ± 0.004           | 0.036 ± 0.003           |
|                           | ARF      | 0.731 ± 0.004     | 0.072 ± 0.012           | 0.077 ± 0.016           |
|                           | RFR      | 0.779 ± 0.003     | 0.075 ± 0.006           | 0.059 ± 0.006           |
| Adult, x_permutations     | FADO     | **0.838 ± 0.003** | **0.061 ± 0.004**\*\*\* | **0.053 ± 0.009**\*\*\* |
|                           | Aranyani | 0.837 ± 0.004     | 0.083 ± 0.007           | 0.073 ± 0.010           |
|                           | ARF      | 0.826 ± 0.004     | 0.090 ± 0.003           | 0.110 ± 0.013           |
|                           | RFR      | 0.837 ± 0.002     | 0.100 ± 0.004           | 0.091 ± 0.010           |
| Adult, y_swaps            | all      | *running*         |                         |                         |
| Adult, y_prior_skip       | all      | *running*         |                         |                         |
| Adult, x_exceed_skip      | all      | *running*         |                         |                         |
| Adult, mean of the four   | all      | *pending*         |                         |                         |

`Aranyani` is the controller-free baseline: the same forest, loss,
initialization, rolling-window statistics, and seed-pinned data ordering as
FADO, with the fixed weight `λ = λ_base` and the adaptive constraint, the
detectors, and the reaction controller disabled. The FADO-vs-Aranyani gap is
therefore attributable to FADO's additions alone. ARF is the Adaptive Random
Forest; RFR is robust fairness regularization.

---

## Reproducing Table III

All four models (`fado`, `aranyani`, `arf`, `rfr`) are run under the same
batch-size-1 test-then-train protocol. Within a seed they share the data
split, the forest initialization, and the stream order: the NumPy, Python, and
TensorFlow generators are pinned per seed, TensorFlow op determinism is on,
and `PYTHONHASHSEED=0`.

**Reporting seeds** (30, disjoint from the tuning seeds `11,22,33,44,55`):

```text
66,77,88,99,101,111,122,133,144,155,166,177,188,199,202,213,224,235,246,257,268,279,291,303,314,325,336,347,358,369
```

**Hyperparameters** are the defaults in `src/main.py` (Table II of the paper).
The Folktables overrides (`drift_lr_spike_mult = 3`,
`temperature_on_drift = 0.5`, `lr_decay_steps = 1000`) are applied
automatically by `_FOLKTABLES_FADO_OVERRIDES` when the dataset is Folktables;
no flag is needed.

### 1. Per-seed sweep

The reported rows come from the W&B grid sweeps in `src/configs/sweeps/`,
one cell per (seed, drift scenario), each cell running all four models:

| Rows                  | Config                                     | Scenarios |
|-----------------------|--------------------------------------------|-----------|
| Folktables, natural   | `sweep_all_arms_folktables.yaml` (10% subsample per seed) | natural cross-year drift |
| Adult, all four drifts | `sweep_all_arms_adult.yaml`               | `sim_x_permutations`, `sim_y_swaps`, `sim_y_prior_skip`, `sim_x_exceed_skip` |

The Adult scenarios were split across several sweeps (and machines) for
wall-clock parallelism; edit the `drift_scenario` list of
`sweep_all_arms_adult.yaml` to choose which scenarios a sweep covers. Then
register the sweep and start one agent per free CPU core:

```bash
src/configs/sweeps/launch.sh all_arms_folktables all_arms_adult
wandb agent <entity>/fado-ablations/<sweep-id>
```

Each run writes its per-seed `results.json` files to
`files/experiments/wandb/<run id>/dataset_<name>/model_<name>/seed_<N>/`.
`src/configs/sweeps/README.md` covers the agents, costs, and the Windows
specifics (W&B's `${env}` does not work there; use a config that calls the
venv's `python.exe` directly, as in `sweep_all_arms_adult_win.yaml`).

A single cell can be run without W&B, e.g. to check the setup:

```bash
PYTHONHASHSEED=0 python -m src.main --pipeline_dataset=adult \
    --pipeline_model=fado,aranyani,arf,rfr --drift_scenario=sim_x_permutations --seeds=66
```

### 2. Summary aggregation

Download every finished run's per-seed metrics of a sweep (keeping only the
seeds on which every model finished):

```bash
python -m src.fetch_wandb_arm_metrics --sweep <sweep-id> --models fado,aranyani,arf,rfr
```

This writes `files/experiments/wandb_arm_metrics/<sweep-id>/raw.csv` (one row
per seed, model, and scenario) and `summary.csv` (mean, std, and number of
seeds per scenario and model, plus an `average` scenario: each seed averaged
over the scenarios first).

### 3. Paper tables and significance tests

```bash
python DOCS/paper_icde/scripts/make_tables.py --refresh
```

The script downloads the per-seed metrics of every sweep listed in its `SWEEPS`
dictionary (finished runs only; cached under `files/experiments/wandb_arm_metrics/`,
so later calls without `--refresh` are offline), computes F1 from the local
per-sample predictions, and writes:

* `DOCS/paper_icde/tables/results.tex` (Table III) and `tables/detection.tex`
  (Table IV), which `paper.tex` includes with `\input`;
* `DOCS/paper_icde/supplementary/significance.md`: FADO against each baseline,
  per block and metric, under the paired Wilcoxon signed-rank test and the
  Welch t-test, each Holm-corrected across the baselines, with the number of
  seeds on which FADO is better;
* `DOCS/paper_icde/supplementary/detection_split.md`: FADO minus Aranyani after
  the drift onset, split by whether FADO detected the drift.

F1 needs the per-sample predictions, which are in the result folders of the
machine that ran each sweep; copy them to `files/experiments/wandb/` first (see
`DOCS/RUNBOOK.md`).

---

## Repository layout

* `src/main.py` — entry point; runs the per-seed pipeline for the selected
  models (`fado`, `aranyani`, `arf`, `rfr`) and holds the default
  hyperparameters and the Folktables overrides.
* `src/models/forest/evaluator.py` — FADO prequential evaluator (Aranyani
  forest + adaptive fairness constraint + detectors + reaction controller).
* `src/models/forest/baseline_evaluator.py` — controller-free Aranyani
  prequential evaluator.
* `src/models/arf/`, `src/models/rfr/` — streaming baselines.
* `src/drift/` — drift injection for the Adult stream (`sim_*` scenarios).
* `src/configs/sweeps/` — W&B sweep configs, `launch.sh`, and their README.
* `src/fetch_wandb_arm_metrics.py` — per-seed metrics of a W&B sweep, mean ±
  std over seeds (CSV).
* `src/fetch_wandb_trajectories.py` — accuracy / DP / EO over time of a W&B
  sweep, averaged over seeds (CSV, plots).
* `DOCS/paper_icde/scripts/make_tables.py` — the paper's Tables III and IV and
  the supplementary significance and detection reports.
* `DOCS/RUNBOOK.md` — everyday commands: machines, sweeps, agents, syncing
  results, tables, figures.
* `files/experiments/wandb/<run id>/` — per-seed result outputs of each run.

---

## Acknowledgements

This project is based on code from
<https://github.com/brcsomnath/Aranyani>, licensed under the MIT License.
