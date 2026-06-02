"""Statistical analysis of FADO vs. baselines on the prequential experiments.

Walks the files/experiments/ tree, loads per-seed results.json, and runs:

  * paired Wilcoxon signed-rank test of FADO vs. each baseline
    (paired by seed, on every (dataset, scenario, metric) cell);
  * Cliff's delta as a non-parametric effect size;
  * Holm-Bonferroni correction within each (dataset, scenario, metric)
    across the baselines;
  * Brown-Forsythe test for equality of variances on the Folktables DP
    column (supporting the variance claim in V.B).

Emits:
  * a printed long-form summary;
  * LaTeX-ready table rows for the Adult and Folktables tables, with
    significance asterisks attached to the FADO entries (and a dagger on
    any baseline that significantly beats FADO).

Convention used for the asterisks (best-baseline-corrected):
    *** p < 0.001    ** p < 0.01    * p < 0.05

Run from anywhere with:
    python3 files/experiments/run_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_ROOT
ALPHA = 0.05

# Map raw model names (in JSON / folder names) -> display names.
DISPLAY_MODEL = {
    "aranyani": "FADO",  # the pipeline ran FADO under the 'aranyani' key
    "arf": "ARF",
    "rfr": "RFR",
}

# Map raw scenario names -> display names.
DISPLAY_SCENARIO = {
    "no_drift": "No Drift",
    "abrupt_gender": "Abrupt",
    "gradual_gender": "Gradual",
    "gender_relationship_decouple": "Decouple",
    "occupation_gender_reversal": "Reversal",
    "folktables": "Folktables",
}

# Metric direction. True = higher is better, False = lower is better.
HIGHER_IS_BETTER = {"accuracy": True, "dp": False, "eo": False}

FADO_KEY = "aranyani"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_long_dataframe(exp_root: Path) -> pd.DataFrame:
    """Walk the experiment tree and build a long-format DataFrame."""
    rows: List[Dict] = []
    for ds_dir in sorted(exp_root.glob("dataset_*")):
        dataset = ds_dir.name.removeprefix("dataset_")
        for mdl_dir in sorted(ds_dir.glob("model_*")):
            model_folder = mdl_dir.name.removeprefix("model_")
            for seed_dir in sorted(mdl_dir.glob("seed_*")):
                seed = int(seed_dir.name.removeprefix("seed_"))
                results_path = seed_dir / "results.json"
                if not results_path.exists():
                    continue
                with open(results_path) as f:
                    entries = json.load(f)
                for entry in entries:
                    if entry.get("error"):
                        continue
                    rows.append({
                        "dataset": dataset,
                        "model_raw": entry.get("model", model_folder),
                        "scenario": entry.get("scenario", "default"),
                        "seed": seed,
                        "accuracy": float(entry["accuracy"]),
                        "dp": float(entry["dp"]),
                        "eo": float(entry["eo"]),
                    })
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def cliffs_delta_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta on the sign of paired differences.

    Returns a value in [-1, +1]. Positive means x tends to exceed y.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    if n == 0:
        return float("nan")
    return float((np.sum(d > 0) - np.sum(d < 0)) / n)


def cliffs_delta_label(delta: float) -> str:
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Two-sided paired Wilcoxon signed-rank test.

    Returns (statistic, p-value). If all differences are zero the test is
    undefined; we return (0, 1.0) in that case so downstream code is safe.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    if not np.any(d != 0):
        return 0.0, 1.0
    res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def holm_correct(pvalues: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction.

    Returns the adjusted p-values in the original input order.
    """
    p = np.asarray(pvalues, float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    p_sorted = p[order]
    adjusted_sorted = np.empty_like(p_sorted)
    running_max = 0.0
    for i, raw in enumerate(p_sorted):
        bonf = raw * (n - i)
        running_max = max(running_max, bonf)
        adjusted_sorted[i] = min(1.0, running_max)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def per_seed_pivot(df: pd.DataFrame, dataset: str, scenario: str, metric: str) -> pd.DataFrame:
    sub = df[(df["dataset"] == dataset) & (df["scenario"] == scenario)]
    wide = sub.pivot_table(index="seed", columns="model_raw", values=metric, aggfunc="first")
    return wide


def compare_cell(df: pd.DataFrame, dataset: str, scenario: str, metric: str) -> Dict:
    """Compare FADO (aranyani) against each baseline in a (dataset, scenario, metric) cell."""
    wide = per_seed_pivot(df, dataset, scenario, metric)
    if FADO_KEY not in wide.columns:
        return {"baselines": {}, "summary": None, "wide": wide}

    higher_better = HIGHER_IS_BETTER[metric]
    fado = wide[FADO_KEY].dropna()

    raw_results: Dict[str, Dict] = {}
    pvals: List[float] = []
    baseline_order: List[str] = []

    for col in wide.columns:
        if col == FADO_KEY:
            continue
        paired = wide[[FADO_KEY, col]].dropna()
        if len(paired) < 2:
            continue
        x = paired[FADO_KEY].to_numpy()
        y = paired[col].to_numpy()
        stat, p = paired_wilcoxon(x, y)
        delta = cliffs_delta_paired(x, y)
        diff = float(x.mean() - y.mean())
        raw_results[col] = {
            "n": int(len(paired)),
            "fado_mean": float(x.mean()),
            "fado_std": float(x.std(ddof=1)),
            "base_mean": float(y.mean()),
            "base_std": float(y.std(ddof=1)),
            "diff_mean": diff,
            "stat": stat,
            "p_raw": p,
            "cliffs_delta": delta,
            "cliffs_label": cliffs_delta_label(delta),
        }
        pvals.append(p)
        baseline_order.append(col)

    p_adj = holm_correct(pvals)
    for col, p_corr in zip(baseline_order, p_adj):
        raw_results[col]["p_holm"] = p_corr
        raw_results[col]["sig"] = sig_marker(p_corr)
        # Direction: positive means FADO wins on this metric.
        if higher_better:
            raw_results[col]["fado_wins"] = raw_results[col]["diff_mean"] > 0
        else:
            raw_results[col]["fado_wins"] = raw_results[col]["diff_mean"] < 0

    summary = {
        "fado_mean": float(fado.mean()),
        "fado_std": float(fado.std(ddof=1)),
        "n_seeds": int(len(fado)),
    }
    return {"baselines": raw_results, "summary": summary, "wide": wide}


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_cell(dataset: str, scenario: str, metric: str, cell: Dict) -> None:
    summary = cell["summary"]
    if summary is None:
        print(f"  [skip {dataset}/{scenario}/{metric}: no FADO data]")
        return
    print(f"\n  >>> {DISPLAY_SCENARIO.get(scenario, scenario)} / {metric.upper()}")
    print(f"      FADO: {summary['fado_mean']:.4f} +/- {summary['fado_std']:.4f}  (n={summary['n_seeds']})")
    for baseline, r in cell["baselines"].items():
        direction = "FADO better" if r["fado_wins"] else "baseline better"
        print(f"      vs {DISPLAY_MODEL.get(baseline, baseline):>5}: "
              f"{r['base_mean']:.4f} +/- {r['base_std']:.4f} | "
              f"diff={r['diff_mean']:+.4f} | "
              f"W={r['stat']:.1f} | p_raw={r['p_raw']:.2e} | p_holm={r['p_holm']:.2e} "
              f"{r['sig'] or '(ns)':>4} | delta={r['cliffs_delta']:+.3f} ({r['cliffs_label']}) | {direction}")


# ---------------------------------------------------------------------------
# LaTeX emission
# ---------------------------------------------------------------------------

def latex_value(mean: float, std: float, bold: bool = False, marker: str = "") -> str:
    body = f"{mean:.4f} \\pm {std:.4f}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    if marker:
        body = f"{body}^{{{marker}}}"
    return f"${body}$"


def best_index(values: List[float], higher_better: bool) -> int:
    arr = np.array(values, float)
    if higher_better:
        return int(np.argmax(arr))
    return int(np.argmin(arr))


def emit_adult_latex(df: pd.DataFrame, scenarios: List[str], models: List[str], metrics: List[str]) -> str:
    """Emit a self-contained LaTeX tabular for the Adult dataset with sig annotations.

    For each (scenario, metric) cell:
      * The model with the best mean is bolded.
      * If FADO is the best, an asterisk superscript (vs. each baseline at Holm-adjusted p<.05)
        is attached to its entry.
      * If a baseline is best AND its advantage over FADO is significant, a dagger is attached.
    """
    lines: List[str] = []
    lines.append("% --- Adult results table (auto-generated) ---")
    lines.append("\\begin{tabular}{ll" + "c" * len(metrics) + "}")
    lines.append("\\toprule")
    header = "\\textbf{Scenario} & \\textbf{Model}"
    for m in metrics:
        arrow = "$\\uparrow$" if HIGHER_IS_BETTER[m] else "$\\downarrow$"
        header += f" & \\textbf{{{m.upper()} {arrow}}}"
    lines.append(header + " \\\\")
    lines.append("\\midrule")

    for s_idx, scenario in enumerate(scenarios):
        # Pre-compute, for each metric, the cell results and the best model index.
        per_metric = {}
        means_per_metric = {}
        for metric in metrics:
            cell = compare_cell(df, "adult", scenario, metric)
            wide = cell["wide"]
            row_means = {m: float(wide[m].mean()) if m in wide.columns else float("nan") for m in models}
            row_stds = {m: float(wide[m].std(ddof=1)) if m in wide.columns else float("nan") for m in models}
            per_metric[metric] = {
                "cell": cell,
                "means": row_means,
                "stds": row_stds,
            }
            means_per_metric[metric] = [row_means[m] for m in models]
        best_idx_per_metric = {
            m: best_index(means_per_metric[m], HIGHER_IS_BETTER[m]) for m in metrics
        }

        for m_idx, model in enumerate(models):
            label = f"\\textbf{{{DISPLAY_SCENARIO.get(scenario, scenario)}}}" if m_idx == 0 else ""
            display_name = (
                f"\\textbf{{{DISPLAY_MODEL.get(model, model)}}}"
                if model == FADO_KEY else DISPLAY_MODEL.get(model, model)
            )
            row = f"{label} & {display_name}"
            for metric in metrics:
                entry = per_metric[metric]
                mean = entry["means"][model]
                std = entry["stds"][model]
                is_best = (best_idx_per_metric[metric] == m_idx)

                marker = ""
                if model == FADO_KEY and is_best:
                    # Worst (largest) Holm p-value of FADO vs each baseline = the strictest claim.
                    cell = entry["cell"]
                    worst_p = max((r["p_holm"] for r in cell["baselines"].values()), default=1.0)
                    marker = sig_marker(worst_p)
                elif model != FADO_KEY and is_best:
                    # Baseline wins. Mark with dagger if it beats FADO significantly.
                    cell = entry["cell"]
                    if model in cell["baselines"] and cell["baselines"][model]["p_holm"] < ALPHA:
                        marker = "\\dagger"

                row += " & " + latex_value(mean, std, bold=is_best, marker=marker)
            lines.append(row + " \\\\")
        if s_idx < len(scenarios) - 1:
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def emit_folktables_latex(df: pd.DataFrame, models: List[str], metrics: List[str]) -> str:
    lines: List[str] = []
    lines.append("% --- Folktables results table (auto-generated) ---")
    lines.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append("\\toprule")
    header = "Model"
    for m in metrics:
        arrow = "$\\uparrow$" if HIGHER_IS_BETTER[m] else "$\\downarrow$"
        header += f" & {m.upper()} {arrow}"
    lines.append(header + " \\\\")
    lines.append("\\midrule")

    per_metric = {}
    means_per_metric = {}
    for metric in metrics:
        cell = compare_cell(df, "folktables", "folktables", metric)
        wide = cell["wide"]
        row_means = {m: float(wide[m].mean()) if m in wide.columns else float("nan") for m in models}
        row_stds = {m: float(wide[m].std(ddof=1)) if m in wide.columns else float("nan") for m in models}
        per_metric[metric] = {"cell": cell, "means": row_means, "stds": row_stds}
        means_per_metric[metric] = [row_means[m] for m in models]
    best_idx_per_metric = {
        m: best_index(means_per_metric[m], HIGHER_IS_BETTER[m]) for m in metrics
    }

    for m_idx, model in enumerate(models):
        display_name = (
            f"\\textbf{{{DISPLAY_MODEL.get(model, model)}}}"
            if model == FADO_KEY else DISPLAY_MODEL.get(model, model)
        )
        row = display_name
        for metric in metrics:
            entry = per_metric[metric]
            mean = entry["means"][model]
            std = entry["stds"][model]
            is_best = (best_idx_per_metric[metric] == m_idx)
            marker = ""
            if model == FADO_KEY and is_best:
                cell = entry["cell"]
                worst_p = max((r["p_holm"] for r in cell["baselines"].values()), default=1.0)
                marker = sig_marker(worst_p)
            elif model != FADO_KEY and is_best:
                cell = entry["cell"]
                if model in cell["baselines"] and cell["baselines"][model]["p_holm"] < ALPHA:
                    marker = "\\dagger"
            row += " & " + latex_value(mean, std, bold=is_best, marker=marker)
        lines.append(row + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Variance test (supports the Folktables variance claim in V.B)
# ---------------------------------------------------------------------------

def variance_test_folktables(df: pd.DataFrame) -> Dict:
    """Brown-Forsythe (Levene-with-median) test for equality of variances on
    Folktables DP across all models."""
    sub = df[(df["dataset"] == "folktables")]
    groups = []
    labels = []
    for model in ["aranyani", "arf", "rfr"]:
        vals = sub[sub["model_raw"] == model]["dp"].dropna().to_numpy()
        if len(vals) >= 2:
            groups.append(vals)
            labels.append(DISPLAY_MODEL.get(model, model))
    if len(groups) < 2:
        return {}
    stat, p = stats.levene(*groups, center="median")
    return {
        "models": labels,
        "stds": [float(g.std(ddof=1)) for g in groups],
        "stat": float(stat),
        "p": float(p),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_long_dataframe(EXP_ROOT)
    if df.empty:
        raise SystemExit("No results.json files found under " + str(EXP_ROOT))

    print(f"Loaded {len(df)} rows.")
    print("Datasets:", sorted(df["dataset"].unique()))
    print("Models:  ", sorted(df["model_raw"].unique()))
    print("Seeds counts per (dataset, model, scenario):")
    print(df.groupby(["dataset", "model_raw", "scenario"]).size().to_string())

    # ----- per-cell paired comparisons -----
    adult_scenarios = ["no_drift", "abrupt_gender", "gradual_gender",
                       "gender_relationship_decouple", "occupation_gender_reversal"]
    metrics = ["accuracy", "dp", "eo"]

    print("\n================ Adult: paired FADO-vs-baseline tests ================")
    for scenario in adult_scenarios:
        for metric in metrics:
            cell = compare_cell(df, "adult", scenario, metric)
            print_cell("adult", scenario, metric, cell)

    print("\n================ Folktables: paired FADO-vs-baseline tests ================")
    for metric in metrics:
        cell = compare_cell(df, "folktables", "folktables", metric)
        print_cell("folktables", "folktables", metric, cell)

    var_res = variance_test_folktables(df)
    if var_res:
        print("\n--- Brown-Forsythe (Levene, median-centred) variance test on Folktables DP ---")
        for label, std in zip(var_res["models"], var_res["stds"]):
            print(f"   sigma_DP({label}) = {std:.5f}")
        print(f"   statistic = {var_res['stat']:.3f}    p = {var_res['p']:.3e}")
        print("   -> " + ("variances differ significantly (p < 0.05)" if var_res['p'] < ALPHA
                          else "no significant variance difference (p >= 0.05)"))

    # ----- LaTeX rows -----
    print("\n================ LaTeX rows (ready to paste) ================")
    model_order_adult = ["arf", "rfr", "aranyani"]
    model_order_folktables = ["aranyani", "arf", "rfr"]

    print("\n----- Adult: full table including 'Decouple' scenario, 3 metrics -----")
    print(emit_adult_latex(
        df,
        scenarios=adult_scenarios,
        models=model_order_adult,
        metrics=metrics,
    ))

    print("\n----- Adult: paper subset (4 scenarios, Acc + DP only) -----")
    print(emit_adult_latex(
        df,
        scenarios=["no_drift", "abrupt_gender", "gradual_gender",
                   "occupation_gender_reversal"],
        models=model_order_adult,
        metrics=["accuracy", "dp"],
    ))

    print("\n----- Folktables: full (3 metrics) -----")
    print(emit_folktables_latex(df, models=model_order_folktables, metrics=metrics))

    print("\n----- Folktables: paper subset (Acc + DP) -----")
    print(emit_folktables_latex(df, models=model_order_folktables, metrics=["accuracy", "dp"]))


if __name__ == "__main__":
    main()
