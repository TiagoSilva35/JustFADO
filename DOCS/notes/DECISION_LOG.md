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

### 2.1 λ controller — current state (2026-09-25)

Built in the recommended setting. Unit-tested on synthetic streams and verified end to end on COMPAS over 3 seeds (finding below). No sweep has run yet.

| part | decision | where |
|---|---|---|
| law | Projected stochastic dual ascent on `+D_g ≤ ε` and `−D_g ≤ ε` for every group g (one multiplier μ ≥ 0 each): `μ ← clip(μ + η(±v_t[g] − ε), 0, λ_max)`, `λ_t = min(λ_base + Σμ, λ_max)` | `src/models/forest/lambda_controller.py` |
| signal | Per-sample Horvitz–Thompson estimate `v_t[g]`, `E[v_t[g]] = D_g`; D_g is exactly the quantity the pipeline reports as DP (`max_g |D_g|`) | `fairness_signal.py`, `per_sample` mode |
| target | ε on the pipeline's DP scale (for two groups, half the positive-rate gap). No pre-drift reference: the target is absolute | `--fairness_target` (0.05) |
| step / cap | η = `--lambda_dual_lr` (0.01), cap `--lambda_max` (10); floor is `--lambda_const`, the same λ Base runs with | `train.py` flags |
| runs when | Every step, not only after a detection. λ rises while DP > ε and drains back to λ_base once DP < ε | `evaluator.py` |
| forgetting | On every confirmed drift (accuracy or fairness), the fairness penalty's statistics (`agg_y`, `gradient_w`, `gradient_b`, counts) are reset and rebuilt from post-drift samples | `ControllerConfig.reset_fairness_stats` |
| FADO arm | `aranyani` / `fado_full` now includes the λ controller, the fairness monitor and the reset | `controller.py` |
| ablations | `fado_no_lambda` (= FADO before 2.1), `fado_lambda_only`, `fado_no_reset`, `fado_no_lr`, `fado_no_temp`, `fado_no_prewarm`, `fado_no_noise_guard`, `fado_detect_only`, `fado_monitor_only`. The old `fado_lr_only`, `fado_temp_only` and `fado_monitor` are gone | `sweep_component_*` |
| sensitivity | ε ~ U(0.01, 0.10) and η ~ log-U(0.001, 0.1) added to `sweep_sensitivity.yaml` (controller-only, per the budget rule) | |
| outputs | Per step `lambda`; rows `stream_final_lambda`, `post_drift_lambda`, `phase_<phase>_lambda`, `fairness_resets` | `main.py` |

#### Why λ is a dual variable — the derivation

Each step states what it gives and what justifies it.

**Step 1 — State the goal as a constraint, not a weighted sum.**
Aranyani-Base minimises `L(θ) + λ·P(θ)` with λ set by hand. What we want is
`min_θ L(θ)  s.t.  DP(θ) ≤ ε`.
*Justification:* ε is a requirement you can state and defend ("DP at most 0.05"). λ is a knob with no meaning outside one dataset. Under drift, the λ that meets a fixed ε changes, so a fixed λ cannot track it.

**Step 2 — Rewrite DP as one-sided linear constraints.**

*What D_g is.* The pipeline defines `DP = max_g |D_g|` with `D_g = r̄ − r_g`, where `r_g` is group g's positive-prediction rate and `r̄` is the mean of those rates. Example with two groups: `r_0 = 0.30`, `r_1 = 0.50`, so `r̄ = 0.40`, `D_0 = +0.10` (below the mean), `D_1 = −0.10` (above it) and `DP = 0.10`, half the 0.20 gap. With two groups the deviations are always mirror images: `D_0 = −D_1`.

*The rewrite.* `|x| ≤ ε` says x lies between −ε and +ε, which is the two one-sided rules `x ≤ ε` ("not too far above") and `−x ≤ ε` ("not too far below"). Applied to every group, `max_g |D_g| ≤ ε` becomes the 2A constraints `+D_g ≤ ε` and `−D_g ≤ ε`: the same requirement, with no absolute value and no max. In the example with ε = 0.05, `D_0 ≤ ε` is broken (0.10 > 0.05) and `−D_0 ≤ ε` holds, so the rewrite also says *which side* is violated.

*Why it matters: we only have noisy estimates.* D_g is never observed. Step 5 gives a per-sample estimate `v` that is right on average (`E[v] = D_g`) but very noisy, because one positive prediction is weighted by 1/(group share). Suppose the model is perfectly fair (D = 0) and v comes out +1 or −1 with equal chance:
- averaging v directly gives +1, −1, +1, … → **0**. Correct.
- taking the absolute value first gives |v| = 1 every time → **1**. The model looks extremely unfair.

That is `E|v| ≠ |E v|`: averaging and taking the size do not commute. A controller fed |v| (or a max of estimates) would read the noise as unfairness, and λ would climb forever on a fair model.

The one-sided rules use v and −v directly. On the fair model, the rule `D ≤ ε` gets steps `v − ε` (average −0.05) and the rule `−D ≤ ε` gets steps `−v − ε` (also average −0.05). Both prices drain towards zero: the noise makes each step jump, but it cancels over time instead of accumulating.

*Justification:* each constraint is linear in D_g. An unbiased estimate of D_g is therefore an unbiased estimate of every constraint's value, which is what stochastic dual ascent (Steps 4–5) requires. An absolute value or a max would turn estimation noise into apparent unfairness.

*Two-group redundancy.* Because `D_0 = −D_1`, the rule `+D_0 ≤ ε` is identical to `−D_1 ≤ ε`, and `−D_0 ≤ ε` to `+D_1 ≤ ε`. With A = 2 the four rules are two distinct rules, each written twice. The current implementation charges each price twice, which is equivalent to a step of 2η (see the implementation note under the finding, and the to-do on de-duplicating).

**Step 3 — Put a price on each constraint (Lagrangian).**
`𝓛(θ, μ) = L(θ) + Σ_g [ μ_{g,+}(D_g − ε) + μ_{g,−}(−D_g − ε) ]`, with every `μ ≥ 0`. The constrained problem is the saddle point `min_θ max_{μ≥0} 𝓛`.
*Justification:* standard Lagrangian duality. μ is the marginal price of unfairness. Complementary slackness at the optimum says each price is zero unless its constraint is tight. **So the right λ is not a hyperparameter; it is the price at which DP = ε, and it depends on the data.**

**Step 4 — Find the price by dual ascent.**
`∂𝓛/∂μ_{g,±} = ±D_g − ε` is just the violation. Ascent with projection gives
`μ_{g,±} ← clip(μ_{g,±} + η(±D_g − ε), 0, λ_max)`.
The model keeps its usual gradient step in θ; alternating the two is primal–dual gradient descent–ascent.
*Justification:* in control terms, μ integrates the violation. It keeps rising while DP > ε and stops only at DP ≤ ε, so there is no steady-state error. A proportional rule (λ as a function of current DP) would leave one; that is why it was dropped. The clip keeps prices non-negative (a requirement of duality) and bounded (protects accuracy).

**Step 5 — Estimate D_g from one sample at a time.**

*The problem.* Step 4 raises or lowers each price by the violation `±D_g − ε`, but online D_g is never observed. Each step sees one person: their group `a_t` and the model's prediction `ŷ_t` (1 = positive). One person belongs to one group, so a single sample cannot show a *rate* for every group. We need a number computed from that one sample that is right **on average**.

*The trick: weight by how rare the group is.* Let `π_g` be group g's share of the stream (estimated as `π̂_g`, the running share). From each sample, build one number per group:

`u_t[g] = ŷ_t / π̂_g` if the sample is from group g, and `0` otherwise.

Why this averages to the rate `r_g`: a sample is from group g with probability `π_g`, and it is positive with probability `r_g`, so `u_t[g]` equals `1/π_g` with probability `π_g · r_g` and 0 otherwise. On average: `π_g · r_g · (1/π_g) = r_g`. Dividing by the share cancels the chance of seeing that group at all. This is the Horvitz–Thompson estimator.

From the rates to the deviations: `v_t[g] = mean_h(u_t[h]) − u_t[g]` mirrors `D_g = r̄ − r_g`. Averages pass through sums and differences, so `E[v_t[g]] = D_g`.

*Worked example.* Shares `π_0 = 0.6`, `π_1 = 0.4`; true rates `r_0 = 0.30`, `r_1 = 0.50`, so `D_0 = +0.10` (as in Step 2). There are only three kinds of sample:

| sample | probability | u | v_t[0] |
|---|---|---|---|
| group 1, predicted positive | 0.4 × 0.5 = 0.20 | (0, 2.5) | +1.25 |
| group 0, predicted positive | 0.6 × 0.3 = 0.18 | (1.67, 0) | −0.83 |
| any negative prediction | 0.62 | (0, 0) | 0 |

Average of `v_t[0]`: `0.20 × 1.25 − 0.18 × 0.83 + 0 = 0.25 − 0.15 = 0.10 = D_0`. Correct on average, but no single sample says 0.10: each says +1.25, −0.83 or 0. This is the noise Step 2 is built to survive, and the reason it avoids absolute values.

*How the controller uses it.* Each step it plugs the sample's `±v_t[g]` into Step 4 in place of `±D_g`. A single step can move a price the wrong way, but the steps are right on average, so over many samples the prices follow the true violations.

*Justification:*
- **Unbiased.** Stochastic dual ascent needs an estimate of the violation that is correct on average (an unbiased stochastic gradient of the dual). `±v_t[g] − ε` is one, because every constraint from Step 2 is linear in D_g.
- **Independent from step to step.** Each `v_t` comes from one fresh sample. A rolling DP over the last W samples is almost the same number twice in a row: on a stationary 50k stream, rolling DP at W = 250 has lag-1 autocorrelation 0.991 and gave 24 false drift alarms, against 0 for this per-sample signal (`fairness_signal.py`). The fairness detector reads the same signal for the same reason.
- **No window, no lag.** It reacts from the first sample after a change instead of waiting for a window to refill.

*Limits:*
- `π̂_g` is estimated from the samples seen so far, so the estimate is a ratio estimator: consistent, and only approximately unbiased early in the stream. `π̂_g` is floored at 0.02 to keep `1/π̂_g` bounded when a group is rare.
- The predictions come from a model that keeps learning, so the samples are independent given the model, not identically distributed.
- The noise grows as a group gets rarer (1/π_g). A small η averages it out, at the cost of reacting more slowly: that is the η trade-off the sensitivity sweep explores.

**Step 6 — Map the prices onto Aranyani's penalty.**
`λ_t = min(λ_base + Σ μ, λ_max)` multiplies Aranyani's existing fairness penalty.
*Justification:* Aranyani's node-level penalty already pushes towards group parity, with a direction set by the sign of the disparity, so one scalar per step is enough. `λ_base` is the λ that Base runs with, so FADO never applies less fairness pressure than Base: it adds pressure only while DP > ε, and returns to Base's weight when fairness holds.

**Step 7 — Forget on drift.**
Every confirmed drift resets the penalty's running statistics.
*Justification:* the price (Steps 4–5) tracks the current DP, but the penalty's direction comes from statistics accumulated since they were last reset. Without a reset, after a drift a correctly raised price would push along the pre-drift direction.

**Predicted behaviour** (confirmed in the open-loop synthetic test below):
- Under a sustained violation Δ = DP − ε, λ rises by about η·Δ per sample.
- Once DP < ε it drains at η·(ε − DP) per sample. Slack is at most ε, so the drain is slow and a short stream can end with λ still raised.
- Near μ = 0 the clip only removes downward noise, so on a fair stream λ sits slightly above λ_base.

**What the derivation does not give** (see *What is not established* below):
- Primal–dual regret and constraint-violation bounds assume convexity; soft trees are not convex.
- Aranyani's penalty is a node-level proxy, not ∇D_g.
- π̂_g is estimated, so the estimate is a ratio estimator.

What the synthetic test showed (open loop: DP does not respond to λ):
- On a fair stream λ sits slightly above λ_base (~+0.15 at η = 0.01). The multiplier is a random walk held at zero, so noise lifts it a little.
- At DP = 0.20 with ε = 0.05, λ goes 0.3 → 1.2 after 250 samples → 3.0 after 1000 → hits the cap.
- After the unfairness stops, λ drains at η·(ε − DP) per sample, about 5·10⁻⁴ per step here. On a 2,165-sample COMPAS stream it may never return to λ_base. In closed loop, a higher λ should lower DP itself.

#### Finding: the λ controller holds DP down after the drift at no accuracy cost (COMPAS, 3 seeds)

Verification run: COMPAS `abrupt_race`, tuning seeds 11, 22, 33, λ_base = 0.3, ε = 0.05, η = 0.01, all four arms in one invocation (shared pre-training). Results in `files/experiments/wandb/jlspp7ly/` (gitignored); figure `DOCS/presentation/second_meating/assets/lambda_trajectory.png` (seed 11).

| arm (mean of 3 seeds) | accuracy | DP | post-drift DP | mean λ |
|---|---|---|---|---|
| FADO (`aranyani`, with 2.1) | 0.605 | **0.058** | **0.066** | 1.09 |
| FADO, no reset (`fado_no_reset`) | 0.606 | 0.081 | 0.099 | 1.53 |
| FADO, λ fixed (`fado_no_lambda`) | 0.605 | 0.094 | 0.117 | 0.30 |
| Aranyani-Base | 0.617 | 0.093 | 0.116 | 0.30 |

Per seed, DP drops against λ fixed by 54% (seed 11), 18% (seed 22) and 40% (seed 33). The direction is the same on every seed.

**How it happened** (seed 11 trajectory; the same pattern in the per-seed diagnostics):
1. **Nothing differs before the drift.** All three DP curves overlap until the drift starts at sample 649. λ creeps from 0.3 to about 0.8 because pre-drift DP sits near ε and noise lifts the price, but at that level it changes no visible prediction.
2. **The drift opens a gap, and the price answers it.** After the drift starts, λ fixed and Base climb to DP 0.15–0.20 and stay there for the whole drift phase. With the controller, DP peaks at 0.11 (sample ~830) and is then held at or below ε until the recovery phase. λ rises exactly while DP > ε: 0.8 → 2.0 by sample ~1100.
3. **The reset is part of it.** Without the reset, λ climbs higher (1.53 vs 1.09) yet DP falls less (0.081 vs 0.058). The price rises, but it pushes along the pre-drift direction. That is what Step 7 predicted.
4. **Why accuracy does not move.** About 12.5% of predictions change relative to λ fixed (261, 281 and 271 of 2,165). Of those, about half became correct and half became wrong (127/134, 141/140, 134/137). The controller moves borderline cases, and at ~61% accuracy COMPAS has many. The overall positive rate drops from ~0.40 to 0.37, so this is not a collapse to predicting "no" for everyone.

**Limits of the finding:**
- **Noise floor.** The reported DP is a 250-sample rolling window. A perfectly group-blind classifier on these streams still shows rolling DP ≈ 0.026 (simulated, sd ≈ 0.005). Measured above that floor, the excess DP is 0.032 for FADO vs 0.068 for λ fixed (−53%).
- **Scope.** 3 tuning seeds, one scenario, one λ_base, COMPAS only. No significance claim.
- **The controller lags after recovery.** λ drains during recovery and DP climbs back above ε near the end of the stream (seed 11: 0.14 at the last sample). λ only starts rising again at ~2070.
- **Accuracy gap to Base.** FADO is still ~1.1 points below Base, with or without λ. The gap comes from the LR/temperature reaction.
- **Binary groups count each constraint twice** (see the implementation note below).

**Implementation note — duplicate constraints for two groups.** With A = 2, `v_t[0] = −v_t[1]` exactly, so `+D_0 ≤ ε` and `−D_1 ≤ ε` are the same constraint and their multipliers get identical updates. λ = λ_base + Σμ therefore counts each price twice, which is equivalent to a step size of 2η (0.02 in this run). The duality still holds: duplicate constraints only make the individual prices non-unique, not their sum. But the reported η understates the effective step. The straight-line drain in the recovery phase (about 0.002 per sample = 4 active multipliers × ηε) is this effect plus noise lifting both sides.

What is not established:
- **Guarantees.** Primal–dual bounds hold for convex losses and constraints. Soft trees are non-convex, and λ scales Aranyani's node-level surrogate rather than ∇D_g, so the bounds are motivation only.
- **The estimate.** π_g is estimated, so it is a ratio estimator (approximately unbiased). Predictions come from a learning model, so the draws are independent given the model but not identically distributed.

### Verified or fixed

| # | decision | rationale |
|---|---|---|
| 1.2 | No stored result hit the `data_dim == 2^depth` collision | `TESTS/check_depth_collision.py`: adult 108, COMPAS 19, folktables 760 (diabetes was 376); depths 3–10 |
| 3.6 | Downstream scripts read per-row `accuracy`, which is now the rolling-curve mean | Do not mix pre-B1 and post-B1 numbers |
| data | `data.py` cleaned; 10-row COMPAS truncation removed; census/jigsaw/CelebA/diabetes loaders removed | Byte-identical outputs otherwise; COMPAS `--intersectional` works again |
| overrides | Explicitly passed flags win over `_COMPAS/_FOLKTABLES_FADO_OVERRIDES`; the resolved values are logged to W&B as `static_params` | The overrides silently nullified sweeps; pre-training and evaluation used different λ |
| windows | Every arm reads one resolved fairness/accuracy window | On COMPAS, FADO/Base used 250 while ARF/RFR used 1000 |
| sweep metric | `_log_sweep_summary` writes a dict via `wandb.summary.update` | `key in wandb.summary` raised KeyError, so the sweep metric was never written |
| sweeps | 9 configs + `launch.sh` + `TESTS/check_sweeps.py` | The old configs used unregistered scenarios and could not import `src` |
| 2.4 done | Rows carry `phase_<phase>_<metric>` and `post_drift_<metric>`; W&B gets `delta_*_post_drift`; `significance_tests.py` tests the post-drift metrics by default | Both are means of the same rolling curves, so the whole-stream value is the length-weighted mean of the phases |
| folktables phases | Folktables phases are its test years (2017, 2018); change point at the year boundary | Detection scoring used to apply Adult's `SPLITS` to Folktables |
| 3.2 done | `python -m src.plot_lambda_tradeoff`: whole-stream and post-drift panels, FADO/Base curves over λ, ARF/RFR from `sweep_reference_*` | Refuses to mix runs with different configurations |
| Holm NaN | `significance_tests._holm` keeps untestable comparisons (n < 2 paired seeds) as NaN, outside the family | `max(0.0, nan)` made them p = 0 and `***`; any table built from the old single-seed COMPAS file showed false significance |
| sweep outputs | Each W&B run writes to `files/experiments/wandb/<run id>/` and saves its config | Parallel agents overwrote each other in one shared folder |
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

- [ ] **2.1 metric for the claim:** samples above ε after each change point, and time to first return below ε, for every arm (using FADO's ε).
- [ ] **2.1 plot:** λ_t over the stream with the phase boundaries and resets marked, next to rolling DP.
- [ ] **Budget rule:** add a window sweep (fairness/accuracy window) that runs all four arms.
- [ ] **3.3:** prequential McNemar test as a secondary significance test.
- [ ] **1.4:** time `FairDecisionTree.__call__` in `TESTS/rec_leaf_prob.py`, replace the "37.12 → 13.22 min" figure, and regenerate the depth-sweep figure.
- [ ] **Cleanup:** remove the dead `train()` in `src/models/forest/train.py` (it calls removed loaders); drop `age_race_decouple` from `extract_seed_metrics.py`; remove the hard-coded dataset and seed in `create_plots.py`.

### Run

- [ ] **Sweeps** (tuning seeds): `src/configs/sweeps/launch.sh all`.
- [ ] **λ trade-off figures:** after `lambda_budget_*` and `reference_*` finish, `python -m src.plot_lambda_tradeoff`.
- [ ] **3.1 final protocol:** `--seeds=<20 reporting seeds> --run_all_scenarios`, every model and dataset, all arms in one invocation, with `PYTHONHASHSEED=0`.

### Decide

- [ ] **2.1 duplicate constraints (A = 2).** Either de-duplicate them (keep `±D_0` only when A = 2) and re-run the verification, or keep the current behaviour and report the effective step as 2η. Recommend de-duplicating, so η means what it says.
- [ ] **2.1 noise floor.** Report DP alongside the fair-classifier floor (≈ 0.026 at W = 250 on COMPAS), or add a floor-corrected column.
- [ ] **2.1 fairness-detector cooldown.** Resets fired at 703 and 704; decide whether the fairness side gets the same cooldown as the accuracy side.
- [ ] **2.1 accuracy gap to Base.** FADO is less accurate than Base with or without λ, so the gap comes from the LR/temperature reaction. `fado_lambda_only` in the component sweep will show whether λ alone keeps Base's accuracy.
- [ ] **2.1 ε per dataset.** Choose it on the tuning seeds, or state it as a requirement ("DP ≤ ε"). The default 0.05 is a placeholder.
- [ ] **2.1 stability argument.** Write it up for the non-convex case, or state the convex-only caveat in the paper.
- [ ] **2.1 claim.** The falsifiable claim changes with the law. Candidate: after drift onset, FADO spends fewer samples with DP > ε than Base, at equal λ_base.
- [ ] **2.3 tuning protocol:** disjoint streams, or prefix tuning. For prefix tuning, drifts start at 15–30% of the stream, so the prefix would contain no drift to tune on.
- [ ] **3.4 fading-factor accuracy:** recommend dropping it; the rolling window already covers it.
- [ ] **ADWIN two-sidedness** (owned by Tiago): it also fires on accuracy improvements. Needs a one-sided test or a direction check before the controller reacts.
