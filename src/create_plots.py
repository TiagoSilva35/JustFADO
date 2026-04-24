import matplotlib.pyplot as plt
from src.drift.scenarios import SCENARIOS
import pandas as pd
import json
import numpy as np

def plot_all_baselines(
        metrics_over_timesteps: dict[str, list[float]],
        models: list[str]
):
    plt.figure(figsize=(10, 6))
    for model in models:
        plt.plot(metrics_over_timesteps[model], label=model)
    plt.xlabel('Timesteps')
    plt.ylabel('Metric Value')
    plt.title('Baseline Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    folder_results_path = 'files/experiments'
    # 1 plot per scenario in each one consider all baselines on seed 1
    metrics_dict = {
        'aranyani': [],
        'arf': [],
        'rfr': []
    }

    # Configuration
    names = ['aranyani', 'arf', 'rfr']
    seeds = [2, 3, 4, 5]
    target_metrics = ["accuracy", "dp", "eo"]

    # Initialize the dictionary
    json_results_dict = {
        name: {
            scenario: {

                m: [] for m in target_metrics

            } for scenario in SCENARIOS.keys()
        } for name in names
    }

    for name in names:
        folder = f"model_{name}"
        for seed in seeds:
            scenario_path = f'{folder_results_path}/{folder}/dataset_adult/seed_{seed}/results.json'
            with open(scenario_path, 'r') as f:
                metrics_list = json.load(f)
                
                # We iterate through the scenarios and their index simultaneously
                for i, scenario_name in enumerate(SCENARIOS.keys()):
                    # Access the specific result dictionary for this scenario
                    # Assumes metrics_list[i] corresponds to SCENARIOS[i]
                    result_entry = metrics_list[i]
                    
                    for key in target_metrics:
                        json_results_dict[name][scenario_name][key].append(result_entry[key])

    for name in ['aranyani', 'arf', 'rfr']:
        for scenario_name in SCENARIOS.keys():
            folder = f"model_{name}"
            scenario_path = f'{folder_results_path}/{folder}/dataset_adult/seed_2/results.json'
            with open(scenario_path, 'r') as f:
                metrics = json.load(f)
                metrics_dict[name] = list(pd.Series(list(metrics[2]['stream_final_accuracy'])).expanding().mean())

    

    print(json_results_dict)

    # print avg + std for each scenario and model
    for scenario_name in SCENARIOS.keys():
        print(f"Scenario: {scenario_name}")
        for name in names:
            acc_values = json_results_dict[name][scenario_name]['accuracy']
            dp_values = json_results_dict[name][scenario_name]['dp']
            eo_values = json_results_dict[name][scenario_name]['eo']
            avg_acc = np.mean(acc_values)
            std_acc = np.std(acc_values)
            avg_dp = np.mean(dp_values)
            std_dp = np.std(dp_values)
            avg_eo = np.mean(eo_values)
            std_eo = np.std(eo_values)
            print(f"  {name}: Accuracy = {avg_acc:.4f} ± {std_acc:.4f}")
            print(f"  {name}: DP = {avg_dp:.4f} ± {std_dp:.4f}")
            print(f"  {name}: EO = {avg_eo:.4f} ± {std_eo:.4f}")

    #plot_all_baselines(metrics_dict, models=['aranyani', 'arf', 'rfr'])
            


    
    