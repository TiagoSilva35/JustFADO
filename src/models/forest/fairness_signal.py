"""What to feed a drift detector when you want to monitor *fairness*.

Every mainstream drift detector (ADWIN, DDM, HDDM, SEED, STEPD, ...) assumes the
monitored variable is an independent draw. A rolling demographic-parity value
violates that badly: DP_t and DP_{t-1} share W-1 of their W samples, so their
lag-1 autocorrelation is ~1 - 1/W. Measured on a *stationary* 50k stream (no
drift anywhere), feeding a rolling DP to river's ADWIN produces:

    signal                       lag-1 autocorr   delta=1e-5   delta=0.02
    per-sample indicator                 -0.001            0            0
    rolling DP, W=250                     0.991           24           62
    rolling DP subsampled every W        -0.044            0            0

Every one of those detections is a false alarm. This module provides the three
signal designs so the choice is an experimental factor rather than an
assumption:

``raw_dp``
    Feed DP_t at every step. The naive design, kept as the negative control --
    the numbers above are what it buys you.

``subsampled_dp``
    Feed DP_t once every ``window`` steps. Restores approximate independence at
    the cost of up to ``window`` samples of detection latency. Simple to state
    and simple to defend.

``per_sample``
    Feed a per-sample statistic whose *expectation* is the fairness gap, so the
    detector sees an i.i.d. stream and its bound survives. For group g with
    marginal probability pi_g, the Horvitz--Thompson style contribution

        u_t[g] = yhat_t * 1{a_t = g} / pi_g          has  E[u_t[g]] = r_g

    where r_g is group g's positive rate. The deviation of group g from the
    unweighted mean of group rates -- exactly the quantity
    ``RollingFairnessWindow`` reports as DP -- is then

        v_t[g] = mean_h(u_t[h]) - u_t[g],            E[v_t[g]] = D_g

    and each v_t[g] is an independent draw. One detector per group; fairness
    drift is signalled when any of them fires. For two groups this reduces to
    +/- half the signed DP gap. pi_g is estimated from the running counters and
    floored at ``min_group_prob`` to keep the statistic bounded.

The module only *produces* the signal; which detector consumes it, and what the
controller does about a detection, are separate decisions.
"""

import numpy as np

SIGNAL_MODES = ('raw_dp', 'subsampled_dp', 'per_sample')
DEFAULT_MODE = 'per_sample'


class FairnessSignal:
    """Turn the arriving sample into the channels a detector should consume.

    Args:
      mode: one of ``SIGNAL_MODES``.
      num_groups: number of protected groups, A.
      window: fairness window; also the subsampling period for
        ``subsampled_dp``.
      min_group_prob: floor on the estimated group marginal, so the per-sample
        statistic stays bounded when a group is rare.
    """

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
        """Names of the streams this mode emits; one detector is kept per name."""
        if self.mode == 'per_sample':
            return [f'group_{g}' for g in range(self.num_groups)]
        return ['dp']

    def observe(self, prediction, protected, dp_value=None):
        """Return ``{channel: value}`` for this step, or ``None`` to feed nothing.

        Args:
          prediction: the model's 0/1 prediction for the arriving sample.
          protected: the sample's protected-group index.
          dp_value: the current rolling DP, required by the two ``*_dp`` modes.
        """
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

        # per_sample
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
