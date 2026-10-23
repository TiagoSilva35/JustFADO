# FADO runbook: everyday commands

The commands used most often in this project, in the order a typical day uses them. Every
command runs from the repository root, with the virtual environment active, unless the step
says otherwise.

- W&B entity/project: `tjcsilva04-universidade-de-coimbra/fado-ablations`
- Remote machine: `tsalazar@tesla.deec.uc.pt`, repository in `/home/tiago-silva/Aranyani2.0`.
  It needs the UC VPN.

---

## 1. Setup (once per machine)

```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```
```bash
wandb login
```
```bash
java -version          # 17+ is needed (CapyMOA drift injection)
```
```bash
python -m TESTS.check_sweeps          # every sweep config must print "ok"
```

On Windows, activate the environment with `venv\Scripts\activate` instead.

---

## 2. Machines

| machine | speed (ms / sample, one agent) | sensible agents | notes |
|---|---|---|---|
| MacBook (Apple M5) | ~26 | 3–4 (performance cores) | fastest; keep awake with `caffeinate` |
| `tesla` (i9-9900K, 8 cores / 16 threads) | ~62 | 6 (max 8) | shared Linux server, use tmux |
| Windows PC (i5-8500, 6 cores) | ~91 | 4 | |

All three give **bit-identical results** for the same seed and configuration (checked
Mac vs `tesla`, and Windows vs `tesla`), so any of them can run any sweep.

**Is anyone else using `tesla`, and how loaded is it?**
```bash
w                       # who is logged in; load average out of 16
```
```bash
ps -eo user,pid,pcpu,pmem,etime,cmd --sort=-pcpu | head -15
```
```bash
htop                    # live view; q to quit
```
```bash
lscpu; free -h          # CPU model / threads; free memory
```

**Keep the Mac awake while agents run.** Either start each agent through `caffeinate`, or
run `caffeinate -i` in its own terminal and stop it with Ctrl+C. Keep the lid open.
```bash
caffeinate -i wandb agent tjcsilva04-universidade-de-coimbra/fado-ablations/<sweep id>
```

---

## 3. tmux on `tesla` (sessions survive logout)

```bash
ssh tsalazar@tesla.deec.uc.pt
```
```bash
tmux new -s <name>            # new session
```
```bash
tmux attach -t <name>         # back to a running session
```
```bash
tmux ls                       # list sessions
```

Inside a session:

| keys | action |
|---|---|
| `Ctrl+b` then `c` | new window |
| `Ctrl+b` then `w` | list windows and pick one |
| `Ctrl+b` then `d` | detach; everything keeps running |
| `Ctrl+b` then `[` | scroll back; `q` to leave |

---

## 4. Running experiments (W&B sweeps)

**Smoke test** (a few minutes, no W&B):
```bash
PYTHONHASHSEED=0 python -m src.main --pipeline_dataset=compas --pipeline_model=fado,aranyani --drift_scenario=sim_y_swaps --seeds=11
```

**Register a sweep.** Pass the file name without the `sweep_` prefix and `.yaml`. It
validates the config and prints the `wandb agent …` command:
```bash
src/configs/sweeps/launch.sh aranyani_stream_lambda_adult
```
Never execute a `.yaml` file directly; it is a config, not a script.

**Start agents.** Use one per free core. Several agents, on any mix of machines, share the
same sweep queue.
```bash
wandb agent tjcsilva04-universidade-de-coimbra/fado-ablations/<sweep id>
```
To start six agents at once in tmux windows on `tesla`:
```bash
for i in 1 2 3 4 5 6; do tmux new-window "cd /home/tiago-silva/Aranyani2.0 && source venv/bin/activate && wandb agent tjcsilva04-universidade-de-coimbra/fado-ablations/<sweep id>; read"; done
```
To make an agent stop after N runs:
```bash
wandb agent --count 3 tjcsilva04-universidade-de-coimbra/fado-ablations/<sweep id>
```

**Stop gracefully.**
- **One agent:** press **Ctrl+C once**. The current run finishes and is uploaded, then the
  agent exits. A second Ctrl+C kills the current run.
- **A whole sweep:**
  ```bash
  wandb sweep --pause tjcsilva04-universidade-de-coimbra/fado-ablations/<sweep id>
  ```
  Restart it later with `--resume`.

**Re-run a failed seed by hand.** A grid sweep does not retry failed runs (for example,
after a W&B connection error). Use the sweep's own flags with that one seed:
```bash
PYTHONHASHSEED=0 python -m src.main --pipeline_dataset=adult --pipeline_model=aranyani --lambda_const=1.5 --pretrain_lambda_const=0.1 --drift_scenario=<scenario> --seeds=<seed> --deterministic=True --wandb_log=True --wandb_project=fado-ablations
```

**Get the code to another machine.** Commit and push on the Mac, then pull on the other
machine:
```bash
git pull
```

---

## 5. Bringing results back

W&B holds the summary metrics. The per-sample data (predictions, curves, λ_t, the drift
onset) exists only in the result folders of the machine that ran each sweep. It is needed
for F1 and the figures. Copy it to the Mac:
```bash
rsync -a tsalazar@tesla.deec.uc.pt:/home/tiago-silva/Aranyani2.0/files/experiments/wandb/ files/experiments/wandb/
```
From the Windows PC, copy `files\experiments\wandb\` over the same way (a USB stick, a cloud
folder, or `scp -r`). Run ids are unique, so nothing is overwritten.

---

## 6. Results, tables and figures

| what | command | output |
|---|---|---|
| **Paper tables (III, IV), significance report (Wilcoxon and Welch), detection split** | `python DOCS/paper_icde/scripts/make_tables.py` (add `--refresh` to download from W&B again) | `DOCS/paper_icde/tables/*.tex`, `DOCS/paper_icde/supplementary/` |
| Per-seed metrics of one or more sweeps | `python -m src.fetch_wandb_arm_metrics --sweep <id>[,<id>...]` | `files/experiments/wandb_arm_metrics/<ids>/raw.csv`, `summary.csv` |
| Curves over time, mean over seeds | `python -m src.fetch_wandb_trajectories --sweep <id> --plot` | `files/experiments/wandb_trajectories/<id>/` |
| Whole-stream F1 from local predictions (machine with the folders) | `python DOCS/paper_icde/scripts/stream_f1.py adult sim_x_permutations:uwhcwsdg sim_y_swaps:bi6frxi3` | appends to `files/experiments/extra_metrics/per_seed.csv` |
| Onset-aligned figure (Fig. 3) | `cd DOCS/paper_icde/figures && python ../scripts/fig_adult_onset_grid.py adult_onset_grid.pdf` | `adult_onset_grid.pdf/.png` |

The sweeps that feed the paper tables are listed in `SWEEPS` at the top of
`make_tables.py`. Edit that dictionary when a sweep is re-run. A table cell is filled only
when all 30 seeds of that method are finished, so re-running the script with `--refresh`
fills the pending rows as sweeps complete.

**Build the paper:**
```bash
cd DOCS/paper_icde && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```
If `latexmk` reports a stale-file error, delete `paper.aux` and build again.

---

## 7. Sweeps behind the paper

| sweep | what | machine |
|---|---|---|
| `jo5dqqhy` | Folktables: FADO, FADO without λ, Aranyani, ARF, RFR (30 seeds) | `tesla` |
| `uwhcwsdg` | Adult x_permutations: FADO, Aranyani, ARF, RFR | Windows |
| `bi6frxi3` | Adult y_swaps: same arms | Windows |
| `l8c4i60o` | Adult y_prior_skip and x_exceed_skip: same arms | `tesla` |
| `9ulal3qn` | Folktables, Aranyani tuned λ = 0.5, pre-train 0.1 | Mac |
| `rm2glvmh` | Adult x_permutations, Aranyani tuned λ = 1.5, pre-train 0.1 | Mac |
| `wqab0alm` | Adult y_swaps, Aranyani tuned λ = 1.5, pre-train 0.1 | Windows |
| `pzbwmd4e` | Adult y_prior_skip and x_exceed_skip, Aranyani tuned λ = 1.5, pre-train 0.1 | `tesla` |
| `6mauborl` | Folktables, Aranyani λ = 0.5 pre-trained at 0.5 (not in the paper) | `tesla` |

Unused duplicates in W&B that can be deleted: `vssf48sf` and `idng43fh` (registered, never
run) and `abc9p6h4` (its runs were deleted).

---

## 8. Unused or legacy files (found 2026-10-23)

Found with an import scan from every entry point (`src/main.py`, the fetch and plot
scripts, `TESTS/check_sweeps.py`, the paper scripts), plus a review of untracked junk.
Nothing has been deleted; decide per group. Everything tracked can be recovered from git.

**A. Junk tracked by git:**
- `src/helpers/__pycache__/`, `src/drift/__pycache__/`: 28 compiled `.pyc` files, including
  ones for a module that no longer exists (`nsga2_tuner`);
- `.DS_Store` files (7);
- `sweep_all_arms_adult_win.yaml~`: an editor backup;
- `sweep_all_arms_adult_win.yaml` at the repository root: a Windows copy of a sweep config;
- `src/out.txt`, `src/progress.txt`: old console logs;
- `leaf_probability_update_performance.pdf` at the root: a one-off benchmark plot.

Add `__pycache__/` and `.DS_Store` to `.gitignore`, then
`git rm -r --cached src/helpers/__pycache__ src/drift/__pycache__`.

**B. Old results and C. superseded scripts: removed on 2026-10-23.** Deleted with `git rm`,
so they can be recovered from history:
- **old results:** `files/experiments/dataset_*`, `model_*`, `log.txt`, `stats_output.txt` and
  `models/*.pkl`;
- **superseded scripts:** `TESTS/csv_to_table.py`, `TESTS/return_table_wandb.py`,
  `TESTS/artifacts/`, `DOCS/paper_icde/scripts/fig_adult_onset.py` and `paper_stats.py`, and
  `src/create_plots.py`, `significance_tests.py` and `plot_lambda_tradeoff.py`.

The Wilcoxon and Welch statistics that `paper_stats.py` printed are now produced by
`make_tables.py`, in `supplementary/significance.md`. To recover `plot_lambda_tradeoff.py`
for a λ trade-off figure:
`git checkout <commit before the deletion> -- src/plot_lambda_tradeoff.py`.

**D. One-off checks and benchmarks** (fine to keep, perhaps in a `TESTS/archive/` folder):
`TESTS/check_depth_collision.py`, `src/check_stack_equivalence.py`, `TESTS/rec_leaf_prob.py`,
`TESTS/rol_to_deque.py`, `TESTS/drift_dataset.py`.

**E. Library code nothing imports:**
- `src/drift/pandas_adapter.py`;
- `src/models/forest/fdt_test.py`: a unit test nobody runs;
- `src/models/rfr/main_dnn*.py`, `main_rfr.py`, `shell/*.sh`, `utils/fair_metric.py`: the
  original RFR repository's training scripts. The pipeline only uses `rfr.py`, `models.py`
  and `evaluator.py`.

**F. Stale documentation:**
- `src/configs/sweeps/README.md` describes 15 sweeps; only 4 configs remain
  (`all_arms_adult`, `all_arms_folktables`, `aranyani_stream_lambda_adult`,
  `aranyani_stream_lambda_folktables`).
- `src/main.py`: the docstring of `_detection_metrics` still calls the fairness monitor
  "observation only", but it now resets the penalty statistics.

**G. Code paths the paper no longer uses** (keep unless you decide otherwise; `main.py`
still offers them): the hand-built Adult/COMPAS scenarios (`src/drift/scenarios.py`,
`compas_scenarios.py`, `create_drifted_ds.py`), the group-conditional `sim_*_group`
scenarios, and the COMPAS dataset itself.
