import random

import numpy as np
from capymoa.stream import NumpyStream

from src.drift.config import STREAM_MEDIANS
from src.drift.inject_drift import DRIFT_CONFIGS, DriftSimulator

DRIFT_REGION = (0.5, 0.7)
GRADUAL_WIDTH_FRACTION = 0.1
BINARY_SWAP_PROBA = 0.5

SIM_SCENARIOS = {
    f'sim_{kind}{suffix}': (kind, gradual)
    for kind in DRIFT_CONFIGS
    for suffix, gradual in (('', False), ('_gradual', True))
}

_KIND_DESCRIPTIONS = {
    'x_permutations': 'feature columns permuted',
    'y_swaps': f'one class relabelled with prob. {BINARY_SWAP_PROBA}',
    'y_prior_skip': "75% of one class's rows dropped",
    'x_exceed_skip': "rows above one feature's median dropped",
}
SIM_SCENARIO_DESCRIPTIONS = {
    name: (f"Injected {'gradual' if gradual else 'abrupt'} drift: "
           f"{_KIND_DESCRIPTIONS[kind]}")
    for name, (kind, gradual) in SIM_SCENARIOS.items()
}


def numeric_medians(x_train):
  x_train = np.asarray(x_train, dtype=np.float64)
  return {
      f'x{j}': float(np.median(x_train[:, j]))
      for j in range(x_train.shape[1])
      if len(np.unique(x_train[:, j])) > 2
  }


class Generator:

  def __init__(self, scenario, seed=0, medians=None):
    if scenario not in SIM_SCENARIOS:
      raise ValueError(f"Unknown injected-drift scenario {scenario!r}; "
                       f"choose from {sorted(SIM_SCENARIOS)}")
    self.kind, self.gradual = SIM_SCENARIOS[scenario]
    self.scenario = scenario
    self.seed = int(seed)
    self.medians = dict(medians or {})
    self.kept_index = None

  def generate(self, x, y, a):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    a = np.asarray(a, dtype=np.int32)
    n = len(x)
    if self.kind == 'x_exceed_skip' and not self.medians:
      raise ValueError('x_exceed_skip needs the training-split medians.')

    name = f'_sim_{self.scenario}_{self.seed}_{id(self)}'
    STREAM_MEDIANS[name] = self.medians or {'x0': 0.0}
    np_state, py_state = np.random.get_state(), random.getstate()
    np.random.seed(self.seed)
    random.seed(self.seed)
    try:
      stream = NumpyStream(x, y, dataset_name=name,
                           feature_names=[f'x{j}' for j in range(x.shape[1])])
      width = int(GRADUAL_WIDTH_FRACTION * n) if self.gradual else 0
      simulator = DriftSimulator(schema=stream.get_schema(),
                                 **DRIFT_CONFIGS[self.kind],
                                 width=width, drift_region=DRIFT_REGION)
      simulator.fit(n)
      binary = len(simulator.schema.get_label_indexes()) == 2

      keep_x, keep_y, keep_a, kept = [], [], [], []
      changed = dropped = 0
      transition_end = None
      for idx in range(n):
        instance = stream.next_instance()
        if idx == simulator.fitted['drift_onset'] + width:
          transition_end = len(keep_y)
        x_row, y_row = x[idx], int(y[idx])
        if simulator.apply_drift(idx):
          out = simulator.transform(instance)
          if out is None:
            dropped += 1
            continue
          new_y = int(out.y_index)
          if (self.kind == 'y_swaps' and binary and new_y != y_row
              and np.random.random() >= BINARY_SWAP_PROBA):
            new_y = y_row
          changed += int(new_y != y_row or not np.array_equal(out.x, x_row))
          x_row, y_row = np.asarray(out.x, dtype=np.float64), new_y
        keep_x.append(x_row)
        keep_y.append(y_row)
        keep_a.append(a[idx])
        kept.append(idx)
    finally:
      np.random.set_state(np_state)
      random.setstate(py_state)
      STREAM_MEDIANS.pop(name, None)

    self.kept_index = np.asarray(kept, dtype=np.int64)
    onset = int(simulator.fitted['drift_onset'])
    info = {
        'scenario': self.scenario,
        'type': self.kind,
        'gradual': bool(self.gradual),
        'onset': onset,
        'transition_end': transition_end if self.gradual else onset,
        'n_in': n,
        'n_out': len(keep_y),
        'rows_changed': changed,
        'rows_dropped': dropped,
        'drifted_class': int(simulator.fitted['y_selected_label']),
    }
    if self.kind == 'x_exceed_skip':
      info['exceed_feature'] = simulator.fitted['x_exceed_attr']
      info['exceed_median'] = float(simulator.fitted['x_exceed_val'])
    return (np.asarray(keep_x, dtype=np.float32), np.asarray(keep_y, dtype=np.int32),
            np.asarray(keep_a, dtype=np.int32), info)
