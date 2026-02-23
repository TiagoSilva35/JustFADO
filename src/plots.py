import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_metric_over_iterations(metric_values, metric_name, model_type, fairness_type, constraint_type):
    """
    Plots the given metric values over iterations for a specific model and fairness constraint.

    Args:
        metric_values (list): List of metric values recorded at each iteration.
        metric_name (str): Name of the metric being plotted (e.g., 'Accuracy', 'DP').
        model_type (str): Type of model used (e.g., 'mlp', 'forest').
        fairness_type (str): Type of fairness constraint applied (e.g., 'dp', 'eo').
        constraint_type (str): Type of constraint applied (e.g., 'node', 'global').
    """
    assert(type(metric_values) == list), "metric_values should be a list of values recorded at each iteration."
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=list(range(1, len(metric_values) + 1)), y=metric_values, marker='o')
    plt.title(f'{metric_name} over Iterations\nModel: {model_type}, Fairness: {fairness_type}, Constraint: {constraint_type}')
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.xticks(range(1, len(metric_values) + 1))
    plt.grid()
    plt.show()
