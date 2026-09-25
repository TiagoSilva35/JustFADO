# Sweeps

Nine sweeps, each answering one question, plus the rules that keep them
defensible. All log to the W&B project `fado-ablations`.

| file | question | runs |
|---|---|---|
| `sweep_component_compas.yaml` / `_adult` | does each controller component contribute? | 18 / 27 |
| `sweep_monitor_ablation.yaml` | which signal and detector should watch fairness? | 48 |
| `sweep_sensitivity.yaml` | is the gain a basin or a knife edge? | 120 |
| `sweep_lambda_budget_compas.yaml` / `_adult` | equal lambda budget for Aranyani-Base (log 3.2) | 14 / 21 |
| `sweep_reference_compas.yaml` / `_adult` | ARF / RFR reference points for the lambda plot | 2 / 3 |
| `sweep_architecture.yaml` | does the gap survive other depths / tree counts? | 15 |

Each run is 5 tuning seeds, so a run's cost is 5 x (pre-train + both arms).

### Running

From the repo root, with the project venv active:

```bash
src/configs/sweeps/launch.sh all                 # or: launch.sh component_compas monitor_ablation
wandb agent fado-ablations/<sweep-id>            # the command launch.sh prints; one per machine/core
```

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

| split | seeds | Adult scenarios | COMPAS scenarios |
|---|---|---|---|
| tuning | `11,22,33,44,55` | `no_drift`, `abrupt_gender`, `gradual_gender` | `no_drift`, `abrupt_race` |
| reporting | `66,77,...,202` (15 seeds) | all five | `no_drift`, `abrupt_race` |

COMPAS registers only two scenarios, so its split is by seed alone: the
reported numbers are on unseen seeds (different train/test partitions and
different drifted rows), not unseen drift types. Say so in the paper.

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

Each run evaluates the treatment arm *and* `aranyani_base`, and the metric is
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

48 runs, signal design x detector backend x {`no_drift`, `abrupt_race`}.

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

depth x num_trees on COMPAS `abrupt_race`, both arms. Replaces
`files/wandb_tree_depth_sweep.yaml`, which ran one arm on accuracy only.
