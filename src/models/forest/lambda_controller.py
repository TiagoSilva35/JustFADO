"""The lambda controller (decision log 2.1).

FADO's fairness penalty weight becomes the dual variable of a demographic
parity constraint, updated online by projected stochastic dual ascent.

The constraint. The pipeline reports DP as ``max_g |D_g|``, where
``D_g = mean_h(r_h) - r_g`` is group g's deviation from the unweighted mean of
the group positive rates (``utils.RollingFairnessWindow``). ``DP <= epsilon``
is the set of 2A linear constraints

    +D_g <= epsilon   and   -D_g <= epsilon      for every group g,

one non-negative multiplier ``mu`` each.

The signal. ``FairnessSignal('per_sample')`` emits ``v_t[g]`` with
``E[v_t[g]] = D_g`` from a single sample (a Horvitz--Thompson weighting of the
prediction by the group's estimated share). Each constraint is linear in D_g,
so ``+/-v_t[g] - epsilon`` is an unbiased per-sample estimate of the
constraint value. No absolute value or window is taken, which would make it
biased or autocorrelated.

The update, every step:

    mu  <- clip(mu + eta * (+/-v_t[g] - epsilon), 0, lambda_max)
    lambda_t = min(lambda_base + sum(mu), lambda_max)

While DP stays above epsilon the active multiplier integrates the violation
and lambda rises; once DP is below epsilon the multiplier drains back to zero
and lambda returns to ``lambda_base``. That is integral control on the
constraint, with no reference to a pre-drift DP (which may itself be unfair).

What this does and does not guarantee:
  * For a convex loss and convex constraints, projected stochastic primal-dual
    methods have known regret / long-term constraint-violation bounds. The soft
    trees are not convex, and the penalty lambda scales is Aranyani's
    node-level surrogate, not the gradient of D_g itself. Those bounds are
    motivation here, not a guarantee.
  * ``pi_g`` inside the signal is estimated, so the estimate is a ratio
    estimator: consistent, only approximately unbiased early in the stream.
  * The predictions come from a model that is itself learning, so the draws
    are independent given the model, not identically distributed.
"""

import numpy as np

from src.models.forest.fairness_signal import FairnessSignal


class LambdaController:
  """Projected stochastic dual ascent on ``|D_g| <= epsilon``.

  Args:
    lambda_base: the floor, i.e. the fixed lambda of Aranyani-Base.
    epsilon: the DP target, on the pipeline's DP scale (max deviation from the
      mean group rate; for two groups that is half the rate gap).
    eta: dual step size.
    lambda_max: cap on lambda and on each multiplier.
    num_groups: number of protected groups.
  """

  def __init__(self, lambda_base, epsilon=0.05, eta=0.01, lambda_max=10.0,
               num_groups=2):
    self.lambda_base = float(lambda_base)
    self.epsilon = float(epsilon)
    self.eta = float(eta)
    self.lambda_max = max(float(lambda_max), self.lambda_base)
    self.num_groups = max(1, int(num_groups))
    self.signal = FairnessSignal('per_sample', num_groups=self.num_groups)
    # Column 0: +D_g <= eps, column 1: -D_g <= eps.
    self.mu = np.zeros((self.num_groups, 2), dtype=np.float64)

  @property
  def value(self):
    return float(min(self.lambda_base + self.mu.sum(), self.lambda_max))

  def update(self, prediction, protected):
    """Consume one prequential prediction and return the new lambda."""
    observed = self.signal.observe(prediction, protected)
    v = np.array([observed[f'group_{g}'] for g in range(self.num_groups)])
    violation = np.stack([v, -v], axis=1) - self.epsilon
    self.mu = np.clip(self.mu + self.eta * violation, 0.0, self.lambda_max)
    return self.value

  def describe(self):
    return {
        'lambda_base': self.lambda_base,
        'epsilon': self.epsilon,
        'eta': self.eta,
        'lambda_max': self.lambda_max,
        'num_groups': self.num_groups,
        'lambda_final': self.value,
    }
