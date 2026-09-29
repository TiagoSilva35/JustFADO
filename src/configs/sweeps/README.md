# Sweeps

Fifteen sweeps, each answering one question, plus the rules that keep them
defensible. All log to the W&B project `fado-ablations`.

**Datasets and drift.** Every sweep runs on injected drift or on natural drift,
never on the hand-built scenarios:
- **COMPAS**: all 8 injected-drift scenarios (`sim_*`: 4 drift types × abrupt / gradual, `src/drift/dataset_generator.py`).
- **Adult**: the 4 abrupt injected-drift scenarios. The gradual ones would double sweeps that already take days.
- **Folktables**: its natural drift (train on 2015, stream 2017 then 2018), 10% subsample (~18.7k + 18.7k).

The one exception is `no_drift` in the monitor sweep. It is the control where false
alarms are counted: the same stream before injection.

| file | question | runs × seeds |
|---|---|---|
| `sweep_all_arms_{compas,adult,folktables}` | every arm (FADO, FADO without λ, Base, ARF, RFR) on every drift: the main table | 8 / 4 / 1 × 5 |
| `sweep_component_{compas,adult,folktables}` | does each controller component contribute? 9 presets | 72 / 36 / 9 × 5 |
| `sweep_lambda_budget_{compas,adult,folktables}` | λ trade-off curves, same grid for FADO and Base (log 3.2) | 56 / 28 / 7 × 5 |
| `sweep_reference_{compas,adult,folktables}` | ARF / RFR points on those curves | 8 / 4 / 1 × 5 |
| `sweep_sensitivity.yaml` | basin or knife edge? controller parameters; drift type sampled too (COMPAS) | 120 × 5 |
| `sweep_monitor_ablation.yaml` | which signal and detector should watch fairness? (COMPAS) | 120 × 5 |
| `sweep_architecture.yaml` | does the gap survive other depths / tree counts? (COMPAS `sim_y_swaps`) | 15 × 5 |

Every run evaluates all its arms on each of its seeds, so its cost is
seeds × (pre-training + one pass of the stream per arm).

---

## How to run the ablations

### 0. Once per machine

```bash
source venv/bin/activate
wandb login                          # once; the sweeps log to project fado-ablations
python -m TESTS.check_sweeps         # every config must print "ok"
```

- Java 17+ must be on the PATH (capymoa: injected drift and the CapyMOA
  detectors). `java -version` to check.
- Folktables must be in `data/acs-folktables/` (2015, 2017 and 2018 are, on
  this machine). Elsewhere the first run downloads it from the ACS servers.

### 1. Smoke test (≈10 min)

One seed, one scenario, W&B off, before committing hours to a grid:

```bash
PYTHONHASHSEED=0 python -m src.main --pipeline_dataset=compas \
  --pipeline_model=fado,aranyani --drift_scenario=sim_y_swaps --seeds=11
```

### 2. Register the sweeps

```bash
src/configs/sweeps/launch.sh all_arms_compas all_arms_folktables
```

`launch.sh` validates, registers each sweep, and prints its
`wandb agent <entity>/fado-ablations/<id>` command (the "Run sweep agent with" line). Copy it whole: without the entity W&B answers "entityName required".

### 3. Start agents

Each agent runs one cell at a time on about one core (batch-size-1 updates).
Run **one agent per free CPU core**, in separate terminals:

```bash
wandb agent <entity>/fado-ablations/<sweep-id>   # e.g. tjcsilva04-universidade-de-coimbra/fado-ablations/iwoes46q
```

Several agents on the same sweep split its grid between them. Each run writes to
its own `files/experiments/wandb/<run id>/`, so they never overwrite each other.

### 4. Order, cheapest and most decisive first

Costs are **core-hours**: with N agents, divide by about N. Measured at
~30 ms per sample per arm (COMPAS, Folktables, Sep 2026), plus one
pre-training pass per seed.

| stage | sweeps | why | core-hours |
|---|---|---|---|
| A | `all_arms_compas`, `all_arms_folktables` | the main table: does FADO beat the baselines, and on which drift? | ~5 + ~5 |
| B | `component_compas`, `component_folktables` | which component does it: λ, reset, LR, temperature | ~27 + ~21 |
| C | `lambda_budget_*` + `reference_*` for COMPAS and Folktables | the 3.2 trade-off curves | ~24 + ~18 |
| D | `sensitivity`, `monitor_ablation`, `architecture` | robustness (COMPAS) | ~44 + ~44 + ~8 |
| E | the four `_adult` sweeps | the large tabular benchmark | ~19 + ~96 + ~75 + ~11 |

**Stop after stage A if FADO and `fado_no_lambda` don't separate.** Nothing
later is worth its cost until they do.

Per seed: a two-arm run takes ~4.5 min on COMPAS, ~30 min on Folktables and
~32 min on Adult; a five-arm run ~7.5 min, ~57 min and ~56 min.

### 5. Read the results

```bash
python -m src.plot_lambda_tradeoff                   # stage C: trade-off curves
python -m src.plot_lambda_trajectory \
  --results files/experiments/wandb/<run id>/dataset_compas/seed_pipeline_results.json --seed 11 \
  --models fado,fado_no_lambda,aranyani --out lambda_trajectory.png
python -m src.significance_tests --inputs files/experiments/wandb/<run id>/dataset_compas \
  --reference fado --baselines fado_no_lambda,aranyani,arf,rfr
```

LaTeX tables (one per metric, a column per scenario, mean ± std over seeds;
`--include-average` adds an Average column):

```bash
python -m src.extract_seed_metrics --format latex --include-average \
  --metrics accuracy,dp,eo,post_drift_dp --sweep <sweep id> \
  --inputs files/experiments/wandb/*/dataset_compas/seed_pipeline_results.json
```

Every scenario found is a column; `--scenarios a,b` and `--models fado,aranyani`
select explicitly. Always pass `--sweep`: without it the script pools every run
in `--inputs`, whatever sweep it came from.

In W&B, group runs by `drift_scenario` and compare `delta_dp`,
`delta_dp_post_drift` and `delta_accuracy`. Every run also logs per-arm values
as `<arm>/<metric>`.

### 6. Report

Sweeps run on the tuning seeds only. The numbers that go in the paper come from
one final run on the reporting seeds (decision log 3.1), with the settings fixed
beforehand:

```bash
SEEDS=66,77,88,99,101,111,122,133,144,155,166,177,188,199,202
for s in sim_x_permutations sim_x_permutations_gradual sim_y_swaps sim_y_swaps_gradual \
         sim_y_prior_skip sim_y_prior_skip_gradual sim_x_exceed_skip sim_x_exceed_skip_gradual; do
  PYTHONHASHSEED=0 python -m src.main --pipeline_dataset=compas --drift_scenario=$s --seeds=$SEEDS
done
PYTHONHASHSEED=0 python -m src.main --pipeline_dataset=folktables \
  --folktables_subsample_fraction=0.1 --seeds=$SEEDS
```

(`--run_all_scenarios` would also run the hand-built scenarios, which the sweeps no longer use.)

### Details

`launch.sh` runs `python -m TESTS.check_sweeps` first, which rejects unknown
flags, scenarios not registered for the sweep's dataset, and unselectable
models. Run it on its own after editing any config. Commands use
`-m src.main` (`python src/main.py` cannot import `src`) and set
`PYTHONHASHSEED=0`.

**Each run writes its own results folder**, `files/experiments/wandb/<run id>/`
(gitignored), with the run's configuration saved in
`seed_pipeline_results.json`. Parallel agents used to overwrite each other in
`files/experiments/dataset_<name>/`.

**Whole-stream and post-drift metrics (decision 2.4).** Every run reports both.
Each paired delta has a `_post_drift` twin (`delta_dp_post_drift`, ...), and
per-phase means are in the results as `phase_<phase>_<metric>`. The sweeps
still optimise the whole-stream `delta_dp`; the post-drift values sit next to
it in W&B.

**Explicit flags beat dataset defaults.** `_COMPAS_FADO_OVERRIDES` /
`_FOLKTABLES_FADO_OVERRIDES` in `main.py` now apply only to params whose flag
was not passed. Before this, a sweep over `--lambda_const` or the ADWIN deltas
on COMPAS ran every cell at the override value. The resolved values are logged
to the run config as `static_params`, so check that field when in doubt.

---

## Which search method, and why it differs per sweep

| sweep | method | why |
|---|---|---|
| component ablation | **grid** | It is a full factorial over a discrete factor (which component is on), not a search. Every cell is a condition you want to report. |
| sensitivity | **random** | The claim is about the *distribution* of outcomes over a stated prior, not about a best point. Random draws i.i.d. from that prior, so "X% of configurations beat the baseline, 95% CI [a,b]" is an estimate of a real quantity. A grid's fraction is an artifact of the levels you chose; a Bayesian argmax is the most overfitted point in the space and has no error bar. Random also spends its budget better in the dimensions that matter (Bergstra & Bengio, JMLR 2012). |
| monitor ablation | **grid** | Full factorial: signal design x detector backend. Both factors are discrete and every cell is a reported condition. |

**Report the distribution, never the argmax.** "We ran random search and the best
configuration achieves X" invites the obvious objection. "We sampled 120
configurations from the stated prior; Y% improved DP over the baseline, median
improvement Z" does not, and it is the stronger claim.

---

## The tune / report split

**Nothing reported in the paper may be tuned on the data it is reported on.**
Every sweep here runs on the tuning split; the winning configuration is then run
once on the reporting split with `--run_seed_pipeline`.

| split | seeds | scenarios |
|---|---|---|
| tuning | `11,22,33,44,55` | the sweep's injected-drift scenarios; Folktables' natural drift |
| reporting | `66,77,...,202` (15 seeds) | the same scenarios |

The split is by seed: every seed draws a different train/test partition (COMPAS),
subsample (Folktables) and injected-drift onset, drifted class and permutation.
The reported numbers are therefore on unseen streams, but not on unseen drift
*types*. Say so in the paper.

Aranyani-Base gets the same tuning budget for its one knob (`--lambda_const`),
otherwise the comparison is tuned-vs-untuned and a reviewer is right to say so.

### An online-setting caveat worth stating explicitly

Tuning on disjoint *streams* (different seeds and scenarios) is the convention
in the drift-detection literature and is what these sweeps do. It is not the
same as being *deployable*: a controller whose hyperparameters were chosen with
knowledge of complete streams is not something you could have configured at
t=0. If a reviewer presses on this, the stronger protocol is prefix tuning --
tune on the first k% of each stream, report on the remainder, never looking
forward. That changes the pipeline, so it is a decision to make deliberately
rather than a default. State whichever one you used.

---

## Why both arms in every run

Each run evaluates the treatment arm *and* `aranyani`, and the metric is
the paired difference (`delta_dp`, `delta_accuracy`). Both arms share one
pre-trained forest (`_pretrain_aranyani`), so the pairing is exact and the cost
is ~1.5x a single arm rather than 2x. Absolute DP moves with the scenario and
the seed; the paired difference does not.

---

## The two detectors are scored separately

There are two detectors in a run and every detection metric is prefixed with
which one it describes:

| prefix | detector | what it does |
|---|---|---|
| `accuracy_*` | the ADWIN pair on the prequential **error** stream | drives the LR / temperature reaction |
| `fairness_*` | the fairness monitor: a detector from `detectors.py` fed by a signal from `fairness_signal.py` | **observation only** -- records what it would have signalled; nothing reacts to it yet |

Metrics, computed for each detector independently, using the standard
drift-detection convention (a detection inside `[cp, cp + tolerance]` around a
change point is a true positive for it, everything else is a false alarm):

- `*_detections` raw alarms raised
- `*_true_positives` / `*_false_alarms`
- `*_mdr` Missed Detection Rate, change points never detected (lower better)
- `*_mtd` Mean Time to Detection, mean delay in samples (lower better)
- `*_mtfa` Mean Time between False Alarms, in samples; needs >= 2 false alarms,
  NaN otherwise (higher better)
- `*_far` False Alarm Rate, false alarms per sample; defined from one false
  alarm, so this is what a sweep should optimise (lower better)
- `*_mtr` Mean Time Ratio, `(MTFA / MTD) * (1 - MDR)` (higher better)

Change points come from the scenario's phase boundaries, *all* of them: COMPAS
`SPLITS = [0.30, 0.70, 1.0]` changes at 30% (drift onset) **and** at 70% (the
recovery edge, where the concept reverts). Scoring against a single onset would
count a detection at the recovery edge as a very late detection of the first
drift. `--detection_tolerance` defaults to one accuracy window, because a
detector cannot see a change before its own window has refilled.

---

## 1. `sweep_component_{compas,adult}.yaml` -- does each component contribute?

Grid over controller presets, everything else fixed. Answers "ablate the
effect of each component of the proposed method". `no_drift` is in the grid on
purpose: it is the cost of a component when there is nothing to react to.
Report `delta_dp` / `delta_accuracy` per preset with Wilcoxon against the full
controller.

## 2. `sweep_sensitivity.yaml` -- is the gain a tuning artifact?

120 sampled configurations. Report the fraction with `delta_dp > 0` and its CI,
plus the marginal of `delta_dp` against each hyperparameter.

## 3. `sweep_monitor_ablation.yaml` -- which monitor should watch fairness?

120 runs, signal design x detector backend x {`no_drift` (control) + the 4 abrupt injected-drift types}.

- `--fairness_signal_mode`: `raw_dp` (autocorrelated, the negative control),
  `subsampled_dp` (independent, one window of latency), `per_sample` (i.i.d.
  surrogate, no latency).
- `--fairness_detector`: `river:adwin` plus CapyMOA's `adwin`, `seed`, `stepd`,
  `hddm_a`, `page_hinkley`, `cusum`, `rddm`.

Primary metric `fairness_far` on `no_drift`, where there are no change points
so every alarm is a false alarm by construction. Report `fairness_mtd` and
`fairness_mdr` on the drifted scenarios as the cost side.

Expected headline: **signal design dominates detector choice.** On a stationary
50k stream, a rolling DP fed to ADWIN produces ~24 false alarms at delta=1e-5
(lag-1 autocorrelation 0.991) while the per-sample surrogate produces 0 at
delta=0.2.

## 4. `sweep_lambda_budget_{compas,adult}.yaml` -- equal budget for lambda

Grid over `--lambda_const` with **both** arms in every run, so each lambda
yields a paired Base point and FADO point. No lambda is selected: report each
arm's accuracy-vs-DP curve over the whole grid (decision 3.2a). The grid is
the budget, identical for the two arms.

Plot it with `python -m src.plot_lambda_tradeoff` once the lambda-budget and
reference sweeps have finished (it reads `files/experiments/wandb/*/`). Each
figure has a whole-stream panel and a post-drift panel (decision 2.4). If the
same folder also holds other sweeps, narrow it with `--config depth=4` etc.;
the script refuses to draw runs with different configurations on one curve.

**Budget rule.** Controller ablations (component, monitor, sensitivity) vary
only controller parameters. Anything that affects every arm -- lambda, depth,
tree count, windows -- is swept for every arm it reaches.

## 5. `sweep_architecture.yaml` -- forest size

depth x num_trees on COMPAS `sim_y_swaps`, both arms. Replaces
`files/wandb_tree_depth_sweep.yaml`, which ran one arm on accuracy only.
