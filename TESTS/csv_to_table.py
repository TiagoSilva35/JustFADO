"""Pull a sweep's per-seed `arm_metrics` tables from W&B and build seed tables.

    python TESTS/csv_to_table.py                    # fetch sweep jo5dqqhy, then build tables
    python TESTS/csv_to_table.py --sweep ych8qely
    python TESTS/csv_to_table.py --no-fetch         # rebuild tables from the saved CSV

Each finished run logs `arm_metrics` (one row per seed, model, scenario). The
rows are saved to artifacts/arm_metrics_all.csv with the same columns as the
other exports (folktables_30s.csv, ...), then summarised as mean ± std over
seeds into seed_table.{csv,md,tex}.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "artifacts"
INPUT_PATH = OUTPUT_DIR / "arm_metrics_all.csv"
# Already gitignored; keeps W&B's downloaded table files out of the repo.
CACHE_DIR = REPO_DIR / "files" / "experiments" / "wandb_arm_metrics"

ID_COLUMNS = ["seed", "model", "scenario"]
metric_labels = {
    "accuracy": "Accuracy",
    "dp": "DP",
    "eo": "EO",
    "stream_final_accuracy": "Stream accuracy",
    "stream_final_dp": "Stream DP",
    "stream_final_eo": "Stream EO",
    "post_drift_accuracy": "Post-drift accuracy",
    "post_drift_dp": "Post-drift DP",
    "post_drift_eo": "Post-drift EO",
    "stream_final_lambda": "Stream $\\lambda$",
    "post_drift_lambda": "Post-drift $\\lambda$",
    "fairness_resets": "Fairness resets",
    "ms_per_sample": "ms/sample",
}
decimals = {"ms_per_sample": 1, "fairness_resets": 1}


def fetch_arm_metrics(sweep: str, entity: str, project: str) -> pd.DataFrame:
    sys.path.insert(0, str(REPO_DIR))
    from src.fetch_wandb_trajectories import fetch

    raw = fetch(entity, project, sweep,
                cache_dir=str(CACHE_DIR / sweep / "artifacts"),
                table_name="arm_metrics")
    columns = ID_COLUMNS + [c for c in metric_labels if c in raw.columns]
    missing = [c for c in metric_labels if c not in raw.columns]
    if missing:
        print(f"[warn] not in the logged tables: {', '.join(missing)}")
    return raw[columns].sort_values(ID_COLUMNS).reset_index(drop=True)


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--sweep", default="jo5dqqhy", help="W&B sweep id")
parser.add_argument("--entity", default="tjcsilva04-universidade-de-coimbra")
parser.add_argument("--project", default="fado-ablations")
parser.add_argument("--no-fetch", action="store_true",
                    help=f"skip W&B and reuse {INPUT_PATH.name}")
args = parser.parse_args()

if args.no_fetch:
    df = pd.read_csv(INPUT_PATH)
else:
    df = fetch_arm_metrics(args.sweep, args.entity, args.project)
    df.to_csv(INPUT_PATH, index=False)
    print(f"{len(df)} rows written to {INPUT_PATH}")

missing_columns = set(ID_COLUMNS) - set(df.columns)
if missing_columns:
    missing = ", ".join(sorted(missing_columns))
    raise ValueError(f"{INPUT_PATH} is missing required columns: {missing}")

metrics = {
    column: label
    for column, label in metric_labels.items()
    if column in df.columns and df[column].notna().any()
}
if not metrics:
    raise ValueError(f"{INPUT_PATH} contains none of the supported metric columns")

summary = (
    df.groupby(["scenario", "model"], sort=True)[list(metrics)]
    .agg(["mean", "std"])
)


def format_metric(metric: str, plus_minus: str = "±") -> list[str]:
    places = decimals.get(metric, 3)
    means = summary[(metric, "mean")]
    standard_deviations = summary[(metric, "std")]
    return [
        f"{mean:.{places}f} {plus_minus} {std:.{places}f}"
        if pd.notna(mean)
        else "NaN"
        for mean, std in zip(means, standard_deviations)
    ]


grouped = df.groupby(["scenario", "model"], sort=True)
n = grouped["seed"].nunique()

# Readable table (CSV / Markdown); plain-text labels without LaTeX math.
plain_labels = {m: label.replace("$\\lambda$", "λ") for m, label in metrics.items()}
table = pd.DataFrame(
    {plain_labels[metric]: format_metric(metric) for metric in metrics},
    index=summary.index,
)
table.insert(0, "n", n)
table.to_csv(OUTPUT_DIR / "seed_table.csv")
table.reset_index().to_markdown(OUTPUT_DIR / "seed_table.md", index=False)

# LaTeX version (uses $\pm$ and escapes underscores)
tex = pd.DataFrame(
    {label: format_metric(metric, r"$\pm$") for metric, label in metrics.items()},
    index=summary.index,
)
tex.insert(0, "n", n)
tex.index = pd.MultiIndex.from_tuples(
    [(str(scenario).replace("_", r"\_"), str(model).replace("_", r"\_"))
     for scenario, model in tex.index]
)
tex.to_latex(OUTPUT_DIR / "seed_table.tex", multirow=True, escape=False)

print(table.to_string())
