import time
from collections import deque

import numpy as np

from src.helpers import utils


NUM_SAMPLES = 50000
WINDOW_SIZE = 1000
RNG = np.random.default_rng(42)
PREDICTIONS = RNG.integers(0, 2, size=NUM_SAMPLES)
PROTECTED = RNG.integers(0, 2, size=NUM_SAMPLES)
TRUE_LABELS = RNG.integers(0, 2, size=NUM_SAMPLES)


def run_legacy():
    predictions = deque(maxlen=WINDOW_SIZE)
    protected = deque(maxlen=WINDOW_SIZE)
    true_labels = deque(maxlen=WINDOW_SIZE)
    dp_values = []
    eo_values = []

    start = time.perf_counter()
    for prediction, group, true_label in zip(
        PREDICTIONS, PROTECTED, TRUE_LABELS
    ):
        predictions.append(prediction)
        protected.append(group)
        true_labels.append(true_label)

        dp, _ = utils.get_demographic_parity(
            list(predictions), list(protected)
        )
        eo, _ = utils.get_equalized_odds(
            list(predictions), list(protected), list(true_labels)
        )
        dp_values.append(dp)
        eo_values.append(eo)

    return time.perf_counter() - start, dp_values, eo_values


def run_incremental():
    window = utils.RollingFairnessWindow(WINDOW_SIZE)
    dp_values = []
    eo_values = []

    start = time.perf_counter()
    for prediction, group, true_label in zip(
        PREDICTIONS, PROTECTED, TRUE_LABELS
    ):
        window.append(prediction, group, true_label)
        dp, _ = window.demographic_parity()
        eo, _ = window.equalized_odds()
        dp_values.append(dp)
        eo_values.append(eo)

    return time.perf_counter() - start, dp_values, eo_values


legacy_time, legacy_dp, legacy_eo = run_legacy()
incremental_time, incremental_dp, incremental_eo = run_incremental()

np.testing.assert_allclose(
    incremental_dp,
    legacy_dp,
    rtol=0.0,
    atol=1e-12,
    err_msg="Incremental DP differs from the legacy implementation.",
)
np.testing.assert_allclose(
    incremental_eo,
    legacy_eo,
    rtol=0.0,
    atol=1e-12,
    err_msg="Incremental EO differs from the legacy implementation.",
)

print(f"Samples: {NUM_SAMPLES}, window: {WINDOW_SIZE}")
print(
    f"Legacy:      {legacy_time:.4f}s total, "
    f"{legacy_time / NUM_SAMPLES * 1000:.4f} ms/sample"
)
print(
    f"Incremental: {incremental_time:.4f}s total, "
    f"{incremental_time / NUM_SAMPLES * 1000:.4f} ms/sample"
)
print(f"Speedup: {legacy_time / incremental_time:.2f}x")
print("DP and EO match at every timestep.")
