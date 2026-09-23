# Sweeps

Three sweeps, each answering one question, plus the rules that keep them
defensible. Run from the repo root: `wandb sweep src/configs/sweeps/<file>`.

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
| tuning | `11,22,33,44,55` | `abrupt_*`, `gradual_*` |
| reporting | `66,77,...,202` (15 seeds) | `no_drift`, `*_decouple`, `*_swap` |

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

## 1. `sweep_component_ablation.yaml` -- does each component contribute?

24 runs. Grid over controller presets, everything else fixed. Answers "ablate
the effect of each component of the proposed method". Report `delta_dp` /
`delta_accuracy` per preset with Wilcoxon against the full controller.

## 2. `sweep_sensitivity.yaml` -- is the gain a tuning artifact?

120 sampled configurations. Report the fraction with `delta_dp > 0` and its CI,
plus the marginal of `delta_dp` against each hyperparameter.

## 3. `sweep_monitor_ablation.yaml` -- which monitor should watch fairness?

72 runs, signal design x detector backend.

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
