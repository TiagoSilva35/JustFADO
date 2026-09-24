# FADO — decisions and to-dos

Record a decision here when it is made, with one line of rationale. When a
to-do is done, move it to *Decisions* or delete it.

---

## Decisions

### Scope and protocol

| # | decision | rationale |
|---|---|---|
| scope | Exclusively online learning | Tiago, 2026-09-24 |
| scope | Dependency versioning and hardware re-timing are out of scope (old 1.1, 1.3, 2.5, 3.5 dropped) | Tiago, 2026-09-24 |
| 3.2 | Report each arm's accuracy-vs-DP trade-off curve across the shared λ grid; no λ is selected | Nothing to cherry-pick; the claim is that FADO's curve lies beyond Base's |
| budget | Controller ablations vary only controller parameters. Parameters that affect every arm (λ, depth, tree count, windows) are swept for every arm they reach | Keeps the comparison tuned-vs-tuned |
| 2.4 | Report both the whole-stream average and per-phase / post-drift DP and accuracy | The whole-stream mean dilutes the post-drift period, the only place FADO differs from Base |
| 2.2 | Deferred. Leaning towards one global adaptive λ plus a fixed node weighting (none / 1/2^l / reach-weighted) as an ablation; per-tree adaptive λ at most an extra ablation | Per-tree λ has no composition rule for ensemble DP, and trees can cancel each other's unfairness |
| search | Grid for factorial sweeps, random search for sensitivity; report distributions, never an argmax | |
| tune/report | Tuning seeds `11,22,33,44,55`; reporting seeds `66…202`. COMPAS splits by seed only | COMPAS registers only `no_drift` and `abrupt_race` |

### Verified or fixed

| # | decision | rationale |
|---|---|---|
| 1.2 | No stored result hit the `data_dim == 2^depth` collision | `TESTS/check_depth_collision.py`: adult 108, COMPAS 19, folktables 760 (diabetes was 376); depths 3–10 |
| 3.6 | Downstream scripts read per-row `accuracy`, which is now the rolling-curve mean | Do not mix pre-B1 and post-B1 numbers |
| data | `data.py` cleaned; 10-row COMPAS truncation removed; census/jigsaw/CelebA/diabetes loaders removed | Byte-identical outputs otherwise; COMPAS `--intersectional` works again |
| overrides | Explicitly passed flags win over `_COMPAS/_FOLKTABLES_FADO_OVERRIDES`; the resolved values are logged to W&B as `static_params` | The overrides silently nullified sweeps; pre-training and evaluation used different λ |
| windows | Every arm reads one resolved fairness/accuracy window | On COMPAS, FADO/Base used 250 while ARF/RFR used 1000 |
| sweep metric | `_log_sweep_summary` writes a dict via `wandb.summary.update` | `key in wandb.summary` raised KeyError, so the sweep metric was never written |
| sweeps | 7 configs + `launch.sh` + `TESTS/check_sweeps.py` | The old configs used unregistered scenarios and could not import `src` |
| A1 | ADWIN deltas: warn 0.02 / confirm 1e-5 | In river a larger delta is more sensitive; the warning detector must trip first |
| A2 | Label-noise guard on the confirmation branch | |
| A3 | Pre-drift reference lagged one window + `max_recovery_steps` timeout | |
| A4 | Drift temperature held for one full timestep | |
| A5 / leaf prob. | Level-wise vectorised leaf probabilities, `'auto'` by depth | Batch-aware; the per-leaf loop was 5–100× slower |
| B1 / B2 | Rolling accuracy on the fairness window; cumulative kept as `accuracy_cumulative` | Averaging a cumulative curve left the post-60% stream only 9.4% of the metric |
| B4 | Sweeps optimise paired `delta_dp` | |
| C2 | `ControllerConfig` + presets as pipeline model names | |
| D1 | Pre-train once, reload per arm | |
| D2 | Variables dispatched by identity, not shape | Fixes the 1.2 collision |
| D3 | One fairness window for pre-training and evaluation | |
| repro | torch seeded, ARF reseeded, op determinism, fixed default seeds | |
| detection | MDR / MTD / MTFA / FAR / MTR per detector, over all change points | |

---

## To-do

### Blocking

- [ ] **Re-run COMPAS.** Every stored COMPAS result used 10 test samples.
- [ ] **COMPAS default provenance.** λ=1.0, the ADWIN deltas and `lr_decay_steps=600` in `_COMPAS_FADO_OVERRIDES` were hand-set. Re-derive them on the tuning seeds, or document how they were chosen.

### Implement

- [ ] **2.4:** per-phase and post-drift DP/accuracy from `SPLITS` / `PHASE_LABELS`, in the result rows, the W&B summary and `significance_tests.py`, alongside the whole-stream average.
- [ ] **3.2:** trade-off plot of accuracy vs DP over the λ grid per arm (from `sweep_lambda_budget_*`), with ARF/RFR as reference points.
- [ ] **Budget rule:** add a window sweep (fairness/accuracy window) that runs all four arms.
- [ ] **3.3:** prequential McNemar test as a secondary significance test.
- [ ] **1.4:** time `FairDecisionTree.__call__` in `TESTS/rec_leaf_prob.py`, replace the "37.12 → 13.22 min" figure, and regenerate the depth-sweep figure.
- [ ] **Cleanup:** remove the dead `train()` in `src/models/forest/train.py` (it calls removed loaders); drop `age_race_decouple` from `extract_seed_metrics.py`; remove the hard-coded dataset and seed in `create_plots.py`.

### Run

- [ ] **Sweeps** (tuning seeds): `src/configs/sweeps/launch.sh all`.
- [ ] **3.1 final protocol:** `--seeds=<20 reporting seeds> --run_all_scenarios`, every model and dataset, all arms in one invocation, with `PYTHONHASHSEED=0`.

### Decide

- [ ] **2.1 λ controller.** Prerequisites before the reaction law:
  - add forgetting to the fairness statistics (`agg_y` / `gradient_w` / `gradient_b` are whole-stream running means, never reset);
  - pick the signal (per-sample HT surrogate recommended);
  - choose the law: the proportional law as proposed, or dual ascent on a DP ≤ ε constraint;
  - settle the stability argument (existing primal–dual bounds are convex-only);
  - claim to state: FADO recovers pre-drift DP faster than Base.
- [ ] **2.3 tuning protocol:** disjoint streams, or prefix tuning. For prefix tuning, drifts start at 15–30% of the stream, so the prefix would contain no drift to tune on.
- [ ] **3.4 fading-factor accuracy:** recommend dropping it; the rolling window already covers it.
- [ ] **ADWIN two-sidedness** (owned by Tiago): it also fires on accuracy improvements. Needs a one-sided test or a direction check before the controller reacts.
