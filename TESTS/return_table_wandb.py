import wandb
import pandas as pd

api = wandb.Api()

runs = api.runs(
    "tjcsilva04-universidade-de-coimbra/fado-ablations",
    filters={
        "sweep": "jo5dqqhy"
    },
)

rows = []

for idx, run in enumerate(runs):
    print(f"Run {idx}")
    arm_metrics = run.summary.get("arm_metrics")

    if arm_metrics is not None:
        row = {
            "run": run.name,
            "run_id": run.id,
            "seed": run.summary.get("seed"),
            "model": run.summary.get("model"),
            "scenario": run.summary.get("scenario"),
            "accuracy": run.summary.get("accuracy"),
            "dp": run.summary.get("dp"),
            "eo": run.summary.get("eo"),
        }

        # Add the contents of arm_metrics
        if isinstance(arm_metrics, dict):
            row.update(arm_metrics)

        rows.append(row)

df = pd.DataFrame(rows)

print(len(df))
df.to_csv("arm_metrics_all.csv", index=False)