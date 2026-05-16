import matplotlib.pyplot as plt
from src.drift.scenarios import SCENARIOS
import json
import numpy as np
from src.drift.scenarios import SCENARIOS
# -------------------------------------------------------------------------
# 1) Average final metrics over available runs
# -------------------------------------------------------------------------
folder_results_path = 'files/experiments'
RESULTS_PATH = f'{folder_results_path}/dataset_adult/seed_pipeline_results.json'
target_metrics = ["accuracy", "dp", "eo"]

with open(RESULTS_PATH, 'r') as f:
    raw_results = json.load(f)


def _normalize_runs(payload):
    if isinstance(payload, dict) and 'seed_runs' in payload:
        return payload['seed_runs']
    if isinstance(payload, list):
        # Legacy format: one run as a flat list of scenario entries.
        return [{'model': 'unknown', 'seed': None, 'results': payload}]
    raise ValueError("Unsupported results format in seed_pipeline_results.json")


seed_runs = _normalize_runs(raw_results)
model_names = sorted({run.get('model', 'unknown') for run in seed_runs})
scenario_order = list(SCENARIOS.keys())

json_results_dict = {
    model: {
        scenario: {metric: [] for metric in target_metrics}
        for scenario in scenario_order
    }
    for model in model_names
}

for run in seed_runs:
    model = run.get('model', 'unknown')
    for result_entry in run.get('results', []):
        scenario_name = result_entry.get('scenario')
        if scenario_name not in json_results_dict[model]:
            continue
        for key in target_metrics:
            value = result_entry.get(key)
            if value is not None:
                json_results_dict[model][scenario_name][key].append(value)

# Print average ± std for each scenario and model
for scenario_name in scenario_order:
    print(f"Scenario: {scenario_name}")
    for model in model_names:
        for metric in target_metrics:
            values = json_results_dict[model][scenario_name][metric]
            if not values:
                continue
            avg = np.mean(values)
            std = np.std(values)
            print(f"  {model}: {metric} = {avg:.4f} ± {std:.4f}")

# -------------------------------------------------------------------------
# 2) New function – plot gradual gender overtime (accuracy example)
# -------------------------------------------------------------------------
def plot_overtime(seed=1383992390, metric='accuracy', scenario_name='gradual_gender'):
    available_models = raw_results.get('models') if isinstance(raw_results, dict) else model_names
    models = available_models or model_names

    plt.figure(figsize=(12, 6))

    for model in models:
        matching_runs = [
            run for run in seed_runs
            if run.get('model') == model and (seed is None or run.get('seed') == seed)
        ]
        if not matching_runs:
            print(f"Warning: no run for model '{model}' and seed '{seed}'")
            continue

        entry = next(
            (result for result in matching_runs[0].get('results', [])
             if result.get('scenario') == scenario_name),
            None
        )
        timestep_values = None
        if entry and isinstance(entry.get('timestep_results'), dict):
            timestep_values = entry['timestep_results'].get(metric)

        if not timestep_values:
            print(f"Warning: metric '{metric}' not found for {model} in scenario '{scenario_name}'")
            continue

        plt.plot(timestep_values, label=model)

    plt.xlabel('Timestep')
    plt.ylabel(metric.capitalize())
    plt.title(f'{scenario_name} – {metric.capitalize()} Over Time (Seed {seed})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the function – adjust seed and metric as you like.
# Here we use the default seed in the dataset and metric='accuracy'
for scenario in SCENARIOS.keys():
    plot_overtime(metric='accuracy', scenario_name=scenario)