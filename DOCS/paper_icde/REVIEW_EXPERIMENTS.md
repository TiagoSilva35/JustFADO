# Experiments for the three review points

Status on 29 Oct 2026. Every sweep below reuses the arms that the paper's sweeps already ran on the
same 30 seeds and streams (FADO and Aranyani are bit-identical across machines), so no arm is run
twice. The sweeps also set `--pretrain_cache_dir`, so each machine pre-trains a seed once
(36 min on Adult, 21 min on Folktables on tesla) instead of once per run.

| Point | Already resolved | To run | CPU-hours (tesla) | Output |
|---|---|---|---|---|
| 1. ε = 0.05 vs DP ≈ 0.06 | explanation, verified on every local run (`dual_bound.md`) | sync run folders; optionally `dual_step_adult` | 0 (+ ~35) | `dual_bound.md`, figure `dp_lambda.pdf`, `dual_step.md` |
| 2. Ablation on Adult | arm presets, sweep, report | `ablation_adult` (+ `ablation_folktables`) | ~110 (+ ~45) | `ablation.md` |
| 3. Intersectional | pipeline support (Folktables, sex × race) | `intersectional_folktables` | ~32 | `intersectional.md` |

## 1. DP above ε: the dual-ascent bound, and the missing figure

**Explanation (no new runs needed).** Projected dual ascent telescopes into
`(1/T) Σ_t (v_t − ε) ≤ μ_T / (η T)`. The bound holds with equality while the multiplier never
touches zero. So on a finite stream the target holds on average only up to the final price
over (step × length).

`scripts/dual_bound.py` checks this run by run. It regenerates each stream's groups, replays the
λ controller from the stored predictions, and compares the replay with the stored λ_t: the
largest difference is 0.0 over 30 seeds. Before the onset on Adult (mean over 30 seeds):

- rolling DP 0.0597, of which about 0.002 is the window;
- DP of all predictions 0.0579;
- average disparity of the active constraint 0.0568, against a bound of 0.0570;
- λ at the onset 1.51, with T ≈ 10,000.

So the excess over ε is the dual ascent not having converged in about 10⁴ samples: the bound is
attained. On Folktables (T ≈ 18,700) the average disparity is 0.039, below ε, because the price
drains back to its floor.

**Figure.** `scripts/fig_dp_lambda.py` plots the rolling DP of FADO, Aranyani and the
leaked-λ Aranyani (with ε) and FADO's λ_t (mean and IQR) against the two constant weights. It has
one column for Folktables and one per Adult drift, aligned on the onset. It needs the run folders
that are not on the Mac:
- **Windows PC:** `uwhcwsdg` (x_permutations), `bi6frxi3` (y_swaps) and `wqab0alm` (leaked
  y_swaps).
- **tesla:** 23 of the 30 `pzbwmd4e` runs (leaked x_exceed_skip).

**Optional confirmation: `sweep_dual_step_adult`.** The bound predicts that a larger dual step η
shrinks the excess. This sweep reruns FADO with η = 0.03 and 0.1 (η = 0.01 is Table III) on
x_permutations and y_prior_skip. The stream before the onset is the same in every scenario, so
two scenarios suffice. Expect a trade-off: a larger η should bring DP closer to ε, but its λ_t
is noisier and its F1 cost may grow. `dual_step.md` reports both, and `dual_bound.md` adds the
bound for the new runs.

## 2. Component ablation on Adult

| Arm (preset) | Dual ascent | Detectors + reset | Reaction controller (lr, temperature) | Source |
|---|---|---|---|---|
| none: Aranyani | – | – | – | Table III runs |
| controller only: `fado_no_lambda` | – | – | ✓ | Folktables only (paper) |
| dual only: `fado_dual` (new preset) | ✓ | – | – | **new** |
| dual + reset: `fado_lambda_only` | ✓ | ✓ | – | **new** |
| dual + controller: `fado_no_reset` | ✓ | detectors only | ✓ | **new** |
| all: `fado` | ✓ | ✓ | ✓ | Table III runs |

"Drift-aware" is supported if **all** beats **dual only** after the onset, and the two middle arms
show which component carries the gain. `ablation_folktables` runs the same three arms on Folktables,
so the table covers both datasets with one set of arms. `ablation.md` gives mean ± sd per block, with
• or ◦ for each arm's paired test against full FADO (Holm across arms).

## 3. Intersectional fairness

The loader already builds four groups on Folktables (sex × race, white or not). FADO's constraint,
the per-sample estimate and the fairness monitor are defined for any number of groups; the smoke
test confirms four monitor channels. `sweep_intersectional_folktables` runs FADO, Aranyani and ARF.
The report gives DP and EO over the four groups, and DP and EO of each attribute alone, whose
logging was added to the W&B arm table. **RFR is left out:** its penalty
(`src/models/rfr/evaluator.py`) is written for two groups. Adult would need loader changes to build
the composite attribute, so it is not proposed.

## Order and commands

1. **No compute, do first:** copy the run folders listed above to the Mac, then run
   ```bash
   python DOCS/paper_icde/scripts/make_tables.py crawl
   PYTHONHASHSEED=0 python DOCS/paper_icde/scripts/dual_bound.py
   cd DOCS/paper_icde/figures && python ../scripts/fig_dp_lambda.py dp_lambda.pdf
   ```
2. **On each machine, once:**
   ```bash
   python -m TESTS.check_sweeps
   PYTHONHASHSEED=0 python -m TESTS.check_pretrain_cache
   ```
3. **Register and start agents, in priority order:**
   ```bash
   src/configs/sweeps/launch.sh ablation_adult intersectional_folktables dual_step_adult ablation_folktables
   ```
   Put each printed sweep id in `SWEEPS` (`ablation_adult`, `intersectional_folktables`,
   `dualstep_adult`, `ablation_folktables`) in `scripts/make_tables.py`.
4. **When they finish:** on each machine run `make_tables.py crawl` (F1), then
   `make_tables.py build --refresh` and `dual_bound.py`. The reports in `supplementary/` fill in, and
   their cells stay "pending" until each arm has all 30 seeds.

The total is about 220–280 CPU-hours: roughly 16–20 hours of wall-clock time on 14 parallel
runners (tesla 6, Windows 4, Mac 4).
