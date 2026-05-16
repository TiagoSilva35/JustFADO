import numpy as np

from src.helpers.utils import _compute_window_fairness


def _ensure_2d_column(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _one_hot_sensitive(a_values):
    a_arr = np.asarray(a_values)
    unique = np.unique(a_arr)
    if unique.size != 2:
        raise ValueError(
            f"FERMI currently supports binary sensitive attributes; got {unique.tolist()}"
        )
    encoded = np.zeros((len(a_arr), 2), dtype=np.float64)
    encoded[:, 0] = (a_arr == unique[0]).astype(np.float64)
    encoded[:, 1] = (a_arr == unique[1]).astype(np.float64)
    return encoded


def evaluate_fermi_over_timesteps(
    x_train,
    y_train,
    a_train,
    x_test,
    y_test,
    a_test,
    batch_size=1,
    lam=10.0,
    epochs=5,
    initial_epochs=0,
    accuracy_window=None,
    fairness_window=1000,
):
    if int(batch_size) != 1:
        raise ValueError('FERMI online mode requires batch_size=1.')

    try:
        from FERMI import FERMIBinary  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FERMI model requires the FERMI-ODDS package. Install with: pip install FERMI-ODDS"
        ) from exc

    x_train_arr = np.asarray(x_train, dtype=np.float64)
    y_train_arr = _ensure_2d_column(y_train)
    a_train_arr = _one_hot_sensitive(a_train)

    x_test_arr = np.asarray(x_test, dtype=np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.int32)
    y_test_col = _ensure_2d_column(y_test)
    a_test_arr = np.asarray(a_test, dtype=np.int32)
    a_test_oh = _one_hot_sensitive(a_test)

    fermi = FERMIBinary.FERMI(
        x_train_arr,
        x_test_arr,
        y_train_arr,
        y_test_col,
        a_train_arr,
        a_test_oh,
        batch_size=1,
        epochs=max(1, int(epochs)),
        lam=float(lam),
    )
    theta, _ = FERMIBinary.FERMI_Logistic_Regression(
        fermi,
        batch_size=1,
        epochs=max(1, int(epochs)),
        initial_epochs=max(0, int(initial_epochs)),
        test_mode='off',
    )

    logits = np.dot(x_test_arr, theta.detach().numpy())
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs.reshape(-1) >= 0.5).astype(np.int32)

    accuracies = []
    dps = []
    eos = []
    y_preds_all = []
    y_true_all = []
    a_all = []
    correct_buffer = []
    use_rolling = bool(accuracy_window)
    fairness_window_value = int(fairness_window) if fairness_window else None
    n_samples = len(preds)

    for i in range(n_samples):
        pred = int(preds[i])
        y_i = int(y_test_arr[i])
        a_i = int(a_test_arr[i])

        y_preds_all.append(pred)
        y_true_all.append(y_i)
        a_all.append(a_i)

        correct_buffer.append(int(pred == y_i))
        if use_rolling and len(correct_buffer) > int(accuracy_window):
            correct_buffer.pop(0)
        accuracies.append(float(sum(correct_buffer)) / len(correct_buffer))

        dp_val, eo_val = _compute_window_fairness(
            y_preds_all=y_preds_all,
            y_true_all=y_true_all,
            a_all=a_all,
            fairness_start=0,
            fairness_window=fairness_window_value,
        )
        dps.append(float(dp_val))
        eos.append(float(eo_val))

    return {
        'accuracy': accuracies,
        'dp': dps,
        'eo': eos,
        'n_samples': n_samples,
        'drifted_points': [],
    }
