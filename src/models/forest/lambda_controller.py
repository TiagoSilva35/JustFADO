import numpy as np

from src.models.forest.fairness_signal import FairnessSignal


class LambdaController:

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
    if observed is None:
      return self.value
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
