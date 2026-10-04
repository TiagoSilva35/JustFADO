import pandas as pd

#df = pd.read_csv("TESTS/artifacts/wandb_export_2026-10-01T21_50_17.968+01_00.csv")
df = pd.read_csv("TESTS/artifacts/folktables5seeds.csv")

summary = (df.drop(columns="seed")
             .groupby(["scenario", "model"])
             .agg(["mean", "std"]))

# Pick the metrics for the table and the label each one gets
metrics = {"post_drift_accuracy": "Accuracy",
           "post_drift_dp": "DP",
           "post_drift_eo": "EO",
           "ms_per_sample": "ms/sample"}
decimals = {"ms_per_sample": 1}   # 3 decimals for everything else

def fmt(m, pm="±"):
    d = decimals.get(m, 3)
    return [f"{a:.{d}f} {pm} {b:.{d}f}" if pd.notna(a) else "NaN"
            for a, b in zip(summary[(m, "mean")], summary[(m, "std")])]

n = df.groupby(["scenario", "model"]).seed.nunique()

# Readable table (CSV / Markdown)
table = pd.DataFrame({lbl: fmt(m) for m, lbl in metrics.items()}, index=summary.index)
table.insert(0, "n", n)
table.to_csv("TESTS/artifacts/seed_table.csv")
table.reset_index().to_markdown("TESTS/artifacts/seed_table.md", index=False)

# LaTeX version (uses $\pm$ and escapes underscores)
tex = pd.DataFrame({lbl: fmt(m, r"$\pm$") for m, lbl in metrics.items()}, index=summary.index)
tex.insert(0, "n", n)
tex.index = tex.index.set_levels([lvl.str.replace("_", r"\_") for lvl in tex.index.levels])
tex.to_latex("TESTS/artifacts/seed_table.tex", multirow=True)

print(table.to_string())