from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "artifacts" / "arm_metrics_all.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"

df = pd.read_csv(INPUT_PATH)
required_columns = {"scenario", "model", "seed"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    missing = ", ".join(sorted(missing_columns))
    raise ValueError(f"{INPUT_PATH} is missing required columns: {missing}")

metric_labels = {
    "accuracy": "Accuracy",
    "dp": "DP",
    "eo": "EO",
    "ms_per_sample": "ms/sample",
}
metrics = {
    column: label
    for column, label in metric_labels.items()
    if column in df.columns
}
if not metrics:
    raise ValueError(f"{INPUT_PATH} contains none of the supported metric columns")

summary = (
    df.groupby(["scenario", "model"], sort=True)[list(metrics)]
    .agg(["mean", "std"])
)


def format_metric(metric: str, plus_minus: str = "±") -> list[str]:
    decimals = 1 if metric == "ms_per_sample" else 3
    means = summary[(metric, "mean")]
    standard_deviations = summary[(metric, "std")]
    return [
        f"{mean:.{decimals}f} {plus_minus} {std:.{decimals}f}"
        if pd.notna(mean)
        else "NaN"
        for mean, std in zip(means, standard_deviations)
    ]


grouped = df.groupby(["scenario", "model"], sort=True)
n = grouped["seed"].nunique()

# Readable table (CSV / Markdown)
table = pd.DataFrame(
    {label: format_metric(metric) for metric, label in metrics.items()},
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
tex.to_latex(OUTPUT_DIR / "seed_table.tex", multirow=True)

print(table.to_string())