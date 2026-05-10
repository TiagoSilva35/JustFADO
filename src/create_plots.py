import matplotlib.pyplot as plt
from src.drift.scenarios import SCENARIOS
import pandas as pd
import json
import numpy as np

# -------------------------------------------------------------------------
# 1) Existing part – average final metrics over seeds (unchanged)
# -------------------------------------------------------------------------
folder_results_path = 'files/experiments'
names = ['aranyani', 'arf', 'rfr']
seeds = [2, 3, 4, 5]
target_metrics = ["accuracy", "dp", "eo"]

json_results_dict = {
    name: {
        scenario: {m: [] for m in target_metrics}
        for scenario in SCENARIOS.keys()
    } for name in names
}

for name in names:
    folder = f"model_{name}"
    for seed in seeds:
        scenario_path = f'{folder_results_path}/{folder}/dataset_adult/seed_{seed}/results.json'
        with open(scenario_path, 'r') as f:
            metrics_list = json.load(f)
            for i, scenario_name in enumerate(SCENARIOS.keys()):
                result_entry = metrics_list[i]
                for key in target_metrics:
                    json_results_dict[name][scenario_name][key].append(result_entry[key])

# Print average ± std for each scenario and model
for scenario_name in SCENARIOS.keys():
    print(f"Scenario: {scenario_name}")
    for name in names:
        for metric in target_metrics:
            values = json_results_dict[name][scenario_name][metric]
            avg = np.mean(values)
            std = np.std(values)
            print(f"  {name}: {metric} = {avg:.4f} ± {std:.4f}")

# -------------------------------------------------------------------------
# 2) New function – plot gradual gender overtime (accuracy example)
# -------------------------------------------------------------------------
def plot_gradual_gender_overtime(seed=2, metric='accuracy'):
    scenario_name = 'gradual_gender'   # exact key used in your results
    models = ['aranyani', 'arf', 'rfr']  # you can add 'fermi' if needed

    plt.figure(figsize=(12, 6))

    for model in models:
        path = f'{folder_results_path}/model_{model}/dataset_adult/seed_{seed}/results.json'
        with open(path, 'r') as f:
            data = json.load(f)

        # data can be either a list of scenario results directly, or
        # a dict with 'seed_runs' (as in the sample).  We handle both.
        timestep_values = None
        if isinstance(data, list):
            # flat list of scenario entries (your actual pipeline output)
            for entry in data:
                if entry.get('scenario') == scenario_name:
                    timestep_values = entry['timestep_results'][metric]
                    break
        elif isinstance(data, dict) and 'seed_runs' in data:
            # structure as in the sample file
            seed_run = data['seed_runs'][0]
            for entry in seed_run['results']:
                if entry['scenario'] == scenario_name:
                    timestep_values = entry['timestep_results'][metric]
                    break

        if timestep_values is None:
            print(f"Warning: scenario '{scenario_name}' not found for model {model}")
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
# Here we use seed=2 and metric='accuracy' (you can also try 'dp', 'eo')
plot_gradual_gender_overtime(seed=2, metric='accuracy')