import numpy as np

SIGNAL_MODES = ('raw_dp', 'subsampled_dp', 'per_sample')
DEFAULT_MODE = 'per_sample'


class FairnessSignal:
    def __init__(self, mode=DEFAULT_MODE, num_groups=2, window=1000,
                 min_group_prob=0.02):
        mode = str(mode or DEFAULT_MODE).lower()
        if mode not in SIGNAL_MODES:
            raise ValueError(f"mode must be one of {SIGNAL_MODES}, got {mode!r}")
        self.mode = mode
        self.num_groups = max(1, int(num_groups))
        self.window = max(1, int(window))
        self.min_group_prob = float(min_group_prob)
        self._group_counts = np.zeros(self.num_groups, dtype=np.float64)
        self._n = 0

    @property
    def channels(self):
        if self.mode == 'per_sample':
            return [f'group_{g}' for g in range(self.num_groups)]
        return ['dp']

    def observe(self, prediction, protected, dp_value=None):
        self._n += 1
        group = int(protected)
        if 0 <= group < self.num_groups:
            self._group_counts[group] += 1.0

        if self.mode == 'raw_dp':
            return {'dp': float(dp_value or 0.0)}

        if self.mode == 'subsampled_dp':
            if self._n % self.window:
                return None
            return {'dp': float(dp_value or 0.0)}

        pi = self._group_counts / max(self._n, 1)
        pi = np.maximum(pi, self.min_group_prob)
        u = np.zeros(self.num_groups, dtype=np.float64)
        if 0 <= group < self.num_groups:
            u[group] = float(prediction) / pi[group]
        v = u.mean() - u
        return {f'group_{g}': float(v[g]) for g in range(self.num_groups)}

    def describe(self):
        return {
            'mode': self.mode,
            'channels': self.channels,
            'num_groups': int(self.num_groups),
            'window': int(self.window),
            'min_group_prob': float(self.min_group_prob),
            'n_observed': int(self._n),
        }
