import pandas as pd
import random

seed = 42
def generate_drifted_dataset(X):
    random.seed(seed)
    split = [0.15, 0.5, 0.6, 0.75, 1]

    columns = X.columns
    X = X.values.tolist()

    n_samples = len(X)

    # Debug: Print first few samples to see format
    print(f"\nDEBUG: First sample: {X[0]}")
    print(f"DEBUG: Gender (index 9): '{X[0][9]}'")
    print(f"DEBUG: Income (index -1): '{X[0][-1]}'")
    print(f"DEBUG: Total samples: {n_samples}")

    warmup = X[0:int(split[0] * n_samples)]
    abrupt_drift = X[int(split[0] * n_samples):int(split[1] * n_samples)]
    recovery_1 = X[int(split[1] * n_samples):int(split[2] * n_samples)]
    slow_drift = X[int(split[2] * n_samples):int(split[3] * n_samples)]
    recovery_2 = X[int(split[3] * n_samples):]

    # Abrupt drift
    changed_count = 0
    for sample in abrupt_drift:
        income_value = str(sample[-1]).strip().rstrip('.')
        gender_value = str(sample[9]).strip()
        if income_value == '>50K' and gender_value == 'Male':
            sample[9] = ' Female'
            changed_count += 1
    print(f"Abrupt drift: Changed {changed_count} Male samples to Female out of {len(abrupt_drift)} samples")

    # Slow drift
    temperature = 0.999
    changed_count = 0
    for sample in slow_drift:
        income_value = str(sample[-1]).strip().rstrip('.')
        gender_value = str(sample[9]).strip()
        if income_value == '>50K' and gender_value == 'Male':
            if random.random() < temperature:
                sample[9] = ' Female'
                changed_count += 1
            temperature *= 0.999
    print(f"Slow drift: Changed {changed_count} Male samples to Female out of {len(slow_drift)} samples")

    # Rebuild dataframe
    df = pd.DataFrame(
        warmup + abrupt_drift + recovery_1 + slow_drift + recovery_2,
        columns=columns
    )

    return df
