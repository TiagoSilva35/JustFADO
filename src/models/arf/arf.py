import numpy as np
from src.helpers.utils import _compute_window_fairness
from river.ensemble import AdaptiveRandomForestClassifier

def evaluate_arf_over_timesteps(
    x_test,
    y_test,
    a_test,
    seed,

    accuracy_window=200,
):
    arf = AdaptiveRandomForestClassifier(seed=seed)

    accuracies = []
    dps = []
    eos = []

    FAIRNESS_WINDOW = int(accuracy_window) if accuracy_window else None
    correct_buffer = []
    USE_ROLLING = bool(accuracy_window)
    y_preds_all = []
    y_true_all = []
    a_all = []
    n_samples = len(x_test)

    print(f"Running ARF baseline prequentially on {n_samples} samples...")

    for t in range(n_samples):
        x_t = np.array(x_test[t], dtype=np.float32)
        y_t = int(y_test[t])
        a_t = int(a_test[t])

        x_dict = {i: float(v) for i, v in enumerate(x_t)}

        y_pred = arf.predict_one(x_dict)
        y_pred = int(y_pred) if y_pred is not None else 0

        y_preds_all.append(y_pred)
        y_true_all.append(y_t)
        a_all.append(a_t)

        correct_buffer.append(int(y_pred == y_t))
        if USE_ROLLING and len(correct_buffer) > accuracy_window:
            correct_buffer.pop(0)
        accuracies.append(float(sum(correct_buffer)) / len(correct_buffer))

        # Fairness over rolling window
        dp_val, eo_val = _compute_window_fairness(
            y_preds_all=y_preds_all,
            y_true_all=y_true_all,
            a_all=a_all,
            fairness_start=0,
            fairness_window=FAIRNESS_WINDOW,
        )
        dps.append(float(dp_val))
        eos.append(float(eo_val))

        # ── STEP 2: TRAIN ─────────────────────────────────────────────────
        arf.learn_one(x_dict, y_t)

    print("ARF baseline evaluation complete.")
    return {
        'accuracy': accuracies,
        'dp': dps,
        'eo': eos,
        'n_samples': n_samples,
        'drifted_points': [],  
    }