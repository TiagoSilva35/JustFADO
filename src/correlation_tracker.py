"""Real-time correlation tracking between features and protected attributes."""

import numpy as np
import tensorflow as tf


class feature_correlation_tracker:
    """Tracks correlation between features and protected attributes in real-time.
    
    Uses Exponential Moving Averages (EMA) to track:
    - Mean of each feature (E[X_i])
    - Mean of protected attribute (E[Z])
    - Variance of features (Var[X_i])
    - Variance of protected attribute (Var[Z])
    - Covariance between features and protected attribute (Cov[X_i, Z])
    
    Computes Pearson correlation: ρ(X_i, Z) = Cov[X_i, Z] / (σ_X * σ_Z)
    
    Applies aggressive penalty to correlated features that can be
    dynamically removed when correlation decreases.
    """
    
    def __init__(
        self,
        num_features,
        ema_decay=0.99,
        correlation_threshold=0.3,
        penalty_base=1.0,
        penalty_aggression=2.0,
        warmup_samples=50,
        penalty_decay=0.95,
    ):
        """Initialize the correlation tracker.
        
        Args:
            num_features: Number of input features to track.
            ema_decay: Decay factor for EMA (higher = slower adaptation).
            correlation_threshold: Threshold above which features are penalized.
            penalty_base: Base penalty multiplier for correlated features.
            penalty_aggression: Exponent for penalty scaling (higher = more aggressive).
            warmup_samples: Number of samples before penalties are applied.
            penalty_decay: How fast penalties decay when correlation drops.
        """
        self.num_features = num_features
        self.ema_decay = ema_decay
        self.correlation_threshold = correlation_threshold
        self.penalty_base = penalty_base
        self.penalty_aggression = penalty_aggression
        self.warmup_samples = warmup_samples
        self.penalty_decay = penalty_decay
        
        # EMA statistics for features
        self.mean_x = np.zeros(num_features, dtype=np.float64)
        self.mean_sq_x = np.zeros(num_features, dtype=np.float64)  # E[X^2]
        
        # EMA statistics for protected attribute
        self.mean_z = 0.0
        self.mean_sq_z = 0.0  # E[Z^2]
        
        # Cross-moment for covariance: E[X*Z]
        self.mean_xz = np.zeros(num_features, dtype=np.float64)
        
        # Current correlation estimates
        self.correlations = np.zeros(num_features, dtype=np.float64)
        
        # Accumulated penalties (can grow/shrink dynamically)
        self.feature_penalties = np.ones(num_features, dtype=np.float64)
        
        # Track how long each feature has been correlated
        self.correlation_streak = np.zeros(num_features, dtype=np.int32)
        
        # Sample counter for warmup
        self.sample_count = 0
        
        # Bias correction factor for EMA
        self.bias_correction = 0.0
        
    def update(self, x, z):
        """Update statistics with a new sample.
        
        Args:
            x: Feature vector of shape [num_features] or [batch_size, num_features].
            z: Protected attribute (scalar or [batch_size]).
            
        Returns:
            correlations: Current correlation estimates [num_features].
            penalties: Current penalty multipliers [num_features].
        """
        x = np.atleast_2d(x)
        z = np.atleast_1d(z)
        
        batch_size = x.shape[0]
        
        for i in range(batch_size):
            self._update_single(x[i], z[i])
            
        return self.correlations.copy(), self.feature_penalties.copy()
    
    def _update_single(self, x, z):
        """Update with a single sample."""
        self.sample_count += 1
        alpha = 1.0 - self.ema_decay
        
        # Update bias correction (for proper EMA initialization)
        self.bias_correction = 1.0 - (self.ema_decay ** self.sample_count)
        
        # Update EMA of feature means: E[X]
        self.mean_x = self.ema_decay * self.mean_x + alpha * x
        
        # Update EMA of feature squared means: E[X^2]
        self.mean_sq_x = self.ema_decay * self.mean_sq_x + alpha * (x ** 2)
        
        # Update EMA of protected attribute: E[Z]
        self.mean_z = self.ema_decay * self.mean_z + alpha * z
        
        # Update EMA of protected attribute squared: E[Z^2]
        self.mean_sq_z = self.ema_decay * self.mean_sq_z + alpha * (z ** 2)
        
        # Update EMA of cross-moment: E[X*Z]
        self.mean_xz = self.ema_decay * self.mean_xz + alpha * (x * z)
        
        # Compute bias-corrected statistics
        mean_x_corrected = self.mean_x / self.bias_correction
        mean_sq_x_corrected = self.mean_sq_x / self.bias_correction
        mean_z_corrected = self.mean_z / self.bias_correction
        mean_sq_z_corrected = self.mean_sq_z / self.bias_correction
        mean_xz_corrected = self.mean_xz / self.bias_correction
        
        # Compute variances: Var[X] = E[X^2] - E[X]^2
        var_x = np.maximum(mean_sq_x_corrected - mean_x_corrected ** 2, 1e-10)
        var_z = max(mean_sq_z_corrected - mean_z_corrected ** 2, 1e-10)
        
        # Compute covariance: Cov[X,Z] = E[XZ] - E[X]E[Z]
        cov_xz = mean_xz_corrected - mean_x_corrected * mean_z_corrected
        
        # Compute Pearson correlation: ρ = Cov[X,Z] / (σ_X * σ_Z)
        std_x = np.sqrt(var_x)
        std_z = np.sqrt(var_z)
        self.correlations = cov_xz / (std_x * std_z + 1e-10)
        
        # Update penalties based on correlations
        self._update_penalties()
        
    def _update_penalties(self):
        """Update feature penalties based on current correlations."""
        if self.sample_count < self.warmup_samples:
            # Don't apply penalties during warmup
            return
            
        abs_corr = np.abs(self.correlations)
        
        for i in range(self.num_features):
            if abs_corr[i] > self.correlation_threshold:
                # Feature is correlated - increase streak and penalty
                self.correlation_streak[i] += 1
                
                # Aggressive penalty: grows with correlation strength and streak
                # penalty = base * (|ρ| / threshold)^aggression * (1 + log(streak))
                correlation_factor = (abs_corr[i] / self.correlation_threshold) ** self.penalty_aggression
                streak_factor = 1.0 + np.log1p(self.correlation_streak[i])
                
                target_penalty = self.penalty_base * correlation_factor * streak_factor
                
                # Smoothly increase penalty (fast attack)
                self.feature_penalties[i] = max(
                    self.feature_penalties[i],
                    0.7 * self.feature_penalties[i] + 0.3 * target_penalty
                )
            else:
                # Feature is not correlated - decay penalty and reset streak
                self.correlation_streak[i] = 0
                
                # Smoothly decay penalty (slow release)
                self.feature_penalties[i] = (
                    self.penalty_decay * self.feature_penalties[i] +
                    (1 - self.penalty_decay) * 1.0
                )
                
        # Clip penalties to reasonable range
        self.feature_penalties = np.clip(self.feature_penalties, 1.0, 100.0)
        
    def get_gradient_mask(self, as_tensor=True):
        """Get a mask to apply to gradients based on penalties.
        
        Features with high correlation get suppressed gradients.
        
        Args:
            as_tensor: If True, return TensorFlow tensor.
            
        Returns:
            mask: Gradient multiplier for each feature [num_features].
                  Values close to 0 = heavily penalized (correlated).
                  Values close to 1 = not penalized (uncorrelated).
        """
        # Invert penalties to get mask (high penalty = low mask value)
        mask = 1.0 / self.feature_penalties
        
        if as_tensor:
            return tf.constant(mask, dtype=tf.float32)
        return mask
    
    def get_penalty_weights(self, as_tensor=True):
        """Get penalty weights to multiply with fairness loss.
        
        Args:
            as_tensor: If True, return TensorFlow tensor.
            
        Returns:
            weights: Penalty multiplier for each feature [num_features].
        """
        if as_tensor:
            return tf.constant(self.feature_penalties, dtype=tf.float32)
        return self.feature_penalties.copy()
    
    def get_correlated_features(self, return_correlations=False):
        """Get indices of features currently marked as correlated.
        
        Args:
            return_correlations: If True, also return correlation values.
            
        Returns:
            indices: Array of correlated feature indices.
            correlations: (optional) Correlation values for those features.
        """
        abs_corr = np.abs(self.correlations)
        correlated_mask = abs_corr > self.correlation_threshold
        indices = np.where(correlated_mask)[0]
        
        if return_correlations:
            return indices, self.correlations[indices]
        return indices
    
    def get_stats(self):
        """Get current tracking statistics for debugging/logging.
        
        Returns:
            dict: Statistics including correlations, penalties, and streaks.
        """
        return {
            'correlations': self.correlations.copy(),
            'abs_correlations': np.abs(self.correlations),
            'penalties': self.feature_penalties.copy(),
            'streaks': self.correlation_streak.copy(),
            'sample_count': self.sample_count,
            'num_correlated': np.sum(np.abs(self.correlations) > self.correlation_threshold),
            'max_correlation': np.max(np.abs(self.correlations)),
            'mean_penalty': np.mean(self.feature_penalties),
        }
    
    def reset(self):
        """Reset all statistics."""
        self.mean_x.fill(0)
        self.mean_sq_x.fill(0)
        self.mean_z = 0.0
        self.mean_sq_z = 0.0
        self.mean_xz.fill(0)
        self.correlations.fill(0)
        self.feature_penalties.fill(1.0)
        self.correlation_streak.fill(0)
        self.sample_count = 0
        self.bias_correction = 0.0


def apply_correlation_penalty_to_gradients(
    gradients,
    correlation_tracker,
    weight_var_indices,
    penalty_mode='mask'
):
    """Apply correlation-based penalties to gradients.
    
    Args:
        gradients: List of gradient tensors.
        correlation_tracker: FeatureCorrelationTracker instance.
        weight_var_indices: Indices of weight variables in gradients list
                           (these have shape [num_features, ...]).
        penalty_mode: 'mask' to reduce gradient magnitude,
                     'boost' to increase fairness gradient contribution.
    
    Returns:
        modified_gradients: List of modified gradient tensors.
    """
    modified_gradients = list(gradients)
    
    if penalty_mode == 'mask':
        mask = correlation_tracker.get_gradient_mask(as_tensor=True)
        # mask shape: [num_features]
        
        for idx in weight_var_indices:
            if modified_gradients[idx] is not None:
                grad = modified_gradients[idx]
                # Assuming weight shape is [num_features, num_nodes]
                # Apply mask to each feature dimension
                mask_expanded = tf.expand_dims(mask, axis=-1)
                modified_gradients[idx] = grad * mask_expanded
                
    elif penalty_mode == 'boost':
        penalties = correlation_tracker.get_penalty_weights(as_tensor=True)
        # penalties shape: [num_features]
        
        for idx in weight_var_indices:
            if modified_gradients[idx] is not None:
                grad = modified_gradients[idx]
                penalties_expanded = tf.expand_dims(penalties, axis=-1)
                modified_gradients[idx] = grad * penalties_expanded
    
    return modified_gradients