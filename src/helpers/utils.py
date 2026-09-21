"""Commonly used Utilities."""

import math

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import contextlib
from collections import Counter, deque

def construct_penalty_mask(tree_depth=4):
  num_internal_nodes = 2 ** tree_depth - 1
  mask = []
  for i in range(num_internal_nodes):
    power = math.floor(math.log2(i+1))
    factor = 1/(2** power)
    mask.append(factor)
  return np.array(mask).astype(np.float32)


def display_confusion_matrix(y_true, y_pred, save_path='files/confusion_matrix.png',
                             class_names=None, title='Confusion Matrix',
                             log_to_wandb=False, wandb_module=None):
  if class_names is None:
    class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]

  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  
  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  
  # Print confusion matrix
  print("\n" + "="*80)
  print(f"{title.upper()}")
  print("="*80)
  print("\nConfusion Matrix:")
  print(cm)
  
  # Print classification report
  print("\nClassification Report:")
  print(classification_report(y_true, y_pred, 
                              target_names=class_names,
                              digits=4))
  
  # Compute per-class metrics manually for clarity
  if cm.size == 4:  # Binary classification
    tn, fp, fn, tp = cm.ravel()
    
    print("\nDetailed Binary Classification Metrics:")
    print(f"  True Negatives (TN):  {tn:>6d}")
    print(f"  False Positives (FP): {fp:>6d}")
    print(f"  False Negatives (FN): {fn:>6d}")
    print(f"  True Positives (TP):  {tp:>6d}")
    
    # Additional metrics
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    
    if tp + fn > 0:
      sensitivity = tp / (tp + fn)
      recall = sensitivity
      print(f"  Sensitivity/Recall:   {sensitivity:.4f}")
    
    if tn + fp > 0:
      specificity = tn / (tn + fp)
      print(f"  Specificity:          {specificity:.4f}")
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  Precision:            {precision:.4f}")
        if (tp + fn > 0):
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"  F1-Score:             {f1:.4f}")

    print(f"  Accuracy:             {accuracy:.4f}")
  
  # Plot confusion matrix
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
              xticklabels=[f'Pred {name}' for name in class_names],
              yticklabels=[f'True {name}' for name in class_names],
              cbar_kws={'label': 'Count'},
              annot_kws={'size': 14, 'weight': 'bold'})
  plt.title(title, fontsize=16, fontweight='bold', pad=20)
  plt.ylabel('True Label', fontsize=13, fontweight='bold')
  plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
  plt.tight_layout()
  
  # Save figure
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  print(f"\nConfusion matrix saved to: {save_path}")
  
  # Log to wandb if requested
  if log_to_wandb and wandb_module is not None:
    log_dict = {
      "confusion_matrix": wandb_module.Image(save_path),
    }
    if cm.size == 4:
      log_dict.update({
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
      })
    wandb_module.log(log_dict)
  
  plt.close()
  print("="*80 + "\n")
  
  return cm


class PhaseTimer:
  """Accumulate wall-clock time and call counts for named phases of a run.

  Added so the FADO efficiency work is *measurable* rather than asserted: the
  recursive leaf-probability updater and the incremental fairness counters are
  FADO-only code paths, and this timer is what lets a FADO run and an
  Aranyani-Base run be compared on runtime, broken down by where the time
  actually goes, instead of only on accuracy/DP/EO.

  Usage:

      timer = PhaseTimer()
      with timer.phase('predict'):
          ...
      report = timer.report(n_samples=n)

  Overhead is one ``perf_counter`` pair per phase entry (tens of nanoseconds),
  which is negligible next to the TensorFlow ops being measured. Pass
  ``enabled=False`` to make every ``phase`` a no-op.
  """

  def __init__(self, enabled=True):
    self.enabled = bool(enabled)
    self.totals = {}
    self.counts = {}
    self._wall_start = time.perf_counter()

  @contextlib.contextmanager
  def phase(self, name):
    if not self.enabled:
      yield
      return
    start = time.perf_counter()
    try:
      yield
    finally:
      elapsed = time.perf_counter() - start
      self.totals[name] = self.totals.get(name, 0.0) + elapsed
      self.counts[name] = self.counts.get(name, 0) + 1

  def add(self, name, seconds, calls=1):
    """Record time measured elsewhere (e.g. a phase that cannot be wrapped)."""
    if not self.enabled:
      return
    self.totals[name] = self.totals.get(name, 0.0) + float(seconds)
    self.counts[name] = self.counts.get(name, 0) + int(calls)

  def report(self, n_samples=None, extra=None):
    """Return a JSON-serialisable timing breakdown."""
    wall = max(time.perf_counter() - self._wall_start, 1e-12)
    phases = {}
    for name in sorted(self.totals):
      total = self.totals[name]
      calls = self.counts.get(name, 0)
      phases[name] = {
          'seconds': float(total),
          'calls': int(calls),
          'us_per_call': float(total / calls * 1e6) if calls else None,
          'pct_of_wall': float(100.0 * total / wall),
      }
    out = {'wall_seconds': float(wall), 'phases': phases}
    if n_samples:
      out['n_samples'] = int(n_samples)
      out['ms_per_sample'] = float(wall / int(n_samples) * 1000.0)
      out['samples_per_second'] = float(int(n_samples) / wall)
    if extra:
      out.update(extra)
    return out


def format_timing_report(report, tag=''):
  """Render a ``PhaseTimer.report()`` dict as a short printable table."""
  if not isinstance(report, dict):
    return ''
  lines = []
  head = f"[{tag}] timing" if tag else "timing"
  wall = report.get('wall_seconds')
  n = report.get('n_samples')
  summary = f"{head}: {wall:.2f}s wall"
  if n:
    summary += (
        f" over {n} samples"
        f" ({report.get('ms_per_sample', 0.0):.3f} ms/sample,"
        f" {report.get('samples_per_second', 0.0):.1f} samples/s)"
    )
  lines.append(summary)
  for name, stats in (report.get('phases') or {}).items():
    lines.append(
        f"    {name:<20} {stats['seconds']:>8.2f}s"
        f" {stats['pct_of_wall']:>6.1f}%"
        f" {stats['calls']:>8d} calls"
        + (f" {stats['us_per_call']:>9.1f} us/call" if stats['us_per_call'] is not None else "")
    )
  for key in ('code_paths',):
    if report.get(key):
      lines.append(f"    {key}: {report[key]}")
  return "\n".join(lines)


class RollingFairnessWindow:
  """Incrementally maintain DP and EO statistics for a bounded stream window."""

  def __init__(self, maxlen):
    self.maxlen = max(1, int(maxlen))
    self.items = deque()
    self.group_total = Counter()
    self.group_predicted_positive = Counter()
    self.subgroup_total = Counter()
    self.subgroup_predicted_positive = Counter()

  def append(self, prediction, protected, true_label):
    item = (int(prediction), int(protected), int(true_label))
    if len(self.items) == self.maxlen:
      self._remove(self.items.popleft())
    self.items.append(item)
    self._add(item)

  def _add(self, item):
    prediction, protected, true_label = item
    self.group_total[protected] += 1
    self.group_predicted_positive[protected] += prediction
    key = (protected, true_label)
    self.subgroup_total[key] += 1
    self.subgroup_predicted_positive[key] += prediction

  def _remove(self, item):
    prediction, protected, true_label = item
    self.group_total[protected] -= 1
    self.group_predicted_positive[protected] -= prediction
    key = (protected, true_label)
    self.subgroup_total[key] -= 1
    self.subgroup_predicted_positive[key] -= prediction

    if self.group_total[protected] == 0:
      del self.group_total[protected]
      del self.group_predicted_positive[protected]
    if self.subgroup_total[key] == 0:
      del self.subgroup_total[key]
      del self.subgroup_predicted_positive[key]

  @staticmethod
  def _deviation(rates):
    if not rates:
      return 0.0, 1.0
    mean_rate = float(np.mean(rates))
    max_abs_diff = 0.0
    dominant_raw_diff = 0.0
    for rate in rates:
      raw_diff = mean_rate - rate
      if abs(raw_diff) > max_abs_diff:
        max_abs_diff = abs(raw_diff)
        dominant_raw_diff = raw_diff
    return float(max_abs_diff), float(np.copysign(1, dominant_raw_diff))

  def demographic_parity(self):
    rates = [
        self.group_predicted_positive[group] / self.group_total[group]
        for group in sorted(self.group_total)
    ]
    return self._deviation(rates)

  def equalized_odds(self):
    max_abs_diff = 0.0
    dominant_raw_diff = 0.0
    for true_label in (0, 1):
      rates = [
          self.subgroup_predicted_positive[(group, true_label)]
          / self.subgroup_total[(group, true_label)]
          for group in sorted(self.group_total)
          if self.subgroup_total[(group, true_label)] > 0
      ]
      eo, sign = self._deviation(rates)
      if eo > max_abs_diff:
        max_abs_diff = eo
        dominant_raw_diff = sign
    return float(max_abs_diff), float(dominant_raw_diff or 1.0)


def get_demographic_parity(y_predictions, y_protected):
  predictions = np.array(y_predictions)
  protected_group = np.array(y_protected)

  if predictions.size == 0 or protected_group.size == 0:
    return 0.0, 1.0

  unique_groups = np.unique(protected_group)
  if unique_groups.size == 0:
    return 0.0, 1.0

  group_rates = {}
  for group_value in unique_groups:
    group_mask = protected_group == group_value
    if np.sum(group_mask) == 0:
      continue
    group_rates[group_value] = float(np.mean(predictions[group_mask]))

  if not group_rates:
    return 0.0, 1.0

  mean_group_rate = float(np.mean(list(group_rates.values())))
  max_abs_diff = 0.0
  dominant_raw_diff = 0.0
  for group_value in unique_groups:
    if group_value not in group_rates:
      continue
    raw_diff = mean_group_rate - group_rates[group_value]
    abs_diff = abs(raw_diff)
    if abs_diff > max_abs_diff:
      max_abs_diff = abs_diff
      dominant_raw_diff = raw_diff

  return float(max_abs_diff), float(np.copysign(1, dominant_raw_diff))

def get_equalized_odds(y_predictions, y_protected, y_true):
  predictions = np.array(y_predictions)
  protected_group = np.array(y_protected)
  true_labels = np.array(y_true)

  if predictions.size == 0 or protected_group.size == 0 or true_labels.size == 0:
    return 0.0, 1.0

  unique_groups = np.unique(protected_group)
  if unique_groups.size == 0:
    return 0.0, 1.0

  max_abs_diff = 0.0
  dominant_raw_diff = 0.0

  for y_cond in [0, 1]:
    cond_mask = true_labels == y_cond
    if np.sum(cond_mask) == 0:
      continue

    group_rates = {}
    for group_value in unique_groups:
      group_mask = cond_mask & (protected_group == group_value)
      if np.sum(group_mask) == 0:
        continue
      group_rates[group_value] = float(np.mean(predictions[group_mask]))

    if not group_rates:
      continue

    mean_group_rate = float(np.mean(list(group_rates.values())))
    for group_value in unique_groups:
      if group_value not in group_rates:
        continue
      raw_diff = mean_group_rate - group_rates[group_value]
      abs_diff = abs(raw_diff)
      if abs_diff > max_abs_diff:
        max_abs_diff = abs_diff
        dominant_raw_diff = raw_diff

  return float(max_abs_diff), float(np.copysign(1, dominant_raw_diff))


class LegacyFairnessWindow:
  """
  Full-window recomputation of DP and EO (pre-optimisation path).

  This is the fairness monitor Aranyani used *before* the incremental-counter
  optimisation. It is kept deliberately so the pure-Aranyani baseline can run
  on the original code path, leaving ``RollingFairnessWindow`` exclusive to the
  FADO pipeline and making the reported efficiency gain attributable to FADO
  alone.
  """

  def __init__(self, maxlen):
    self.maxlen = max(1, int(maxlen))
    self.predictions = deque(maxlen=self.maxlen)
    self.protected = deque(maxlen=self.maxlen)
    self.true_labels = deque(maxlen=self.maxlen)

  def append(self, prediction, protected, true_label):
    self.predictions.append(int(prediction))
    self.protected.append(int(protected))
    self.true_labels.append(int(true_label))

  def demographic_parity(self):
    return get_demographic_parity(
        list(self.predictions), list(self.protected)
    )

  def equalized_odds(self):
    return get_equalized_odds(
        list(self.predictions), list(self.protected), list(self.true_labels)
    )


def make_fairness_window(maxlen, incremental=True):
  """Return the fairness monitor for the requested code path.

  Args:
    maxlen: fairness window size.
    incremental: ``True`` selects the O(NA) counter-based window (FADO only),
      ``False`` the legacy O(NW) full-window recomputation (Aranyani-Base and
      any other untouched baseline).
  """
  if incremental:
    return RollingFairnessWindow(maxlen)
  return LegacyFairnessWindow(maxlen)


def get_test_performance(model, x_test, y_test, a_test, data_dim,
                        show_confusion_matrix=True, eval_batch_size=64):
  accuracy = tf.keras.metrics.Accuracy()
  auc = tf.keras.metrics.AUC()

  x_test_np = np.asarray(x_test, dtype=np.float32).reshape(-1, data_dim)
  y_true_np = np.asarray(y_test)
  a_test_np = np.asarray(a_test)

  if x_test_np.shape[0] == 0:
    return {
        'accuracy': 0.0,
        'dp': 0.0,
        'eo': 0.0,
        'sensitivity': 0.0,
        'auc': 0.0,
        'f1': 0.0,
    }

  eval_batch_size = int(eval_batch_size) if eval_batch_size is not None else x_test_np.shape[0]
  eval_batch_size = max(1, min(eval_batch_size, x_test_np.shape[0]))

  y_pred_chunks = []
  auc_updated = False
  for start in range(0, x_test_np.shape[0], eval_batch_size):
    end = min(start + eval_batch_size, x_test_np.shape[0])
    x_batch = tf.convert_to_tensor(x_test_np[start:end])
    y_true_batch = tf.convert_to_tensor(y_true_np[start:end])

    y_probs_batch = model(x_batch, training=False)
    y_pred_batch = tf.math.argmax(y_probs_batch, axis=-1, output_type=tf.int32)

    accuracy.update_state(tf.cast(y_true_batch, tf.int32), y_pred_batch)

    if y_probs_batch.shape.rank == 2 and y_probs_batch.shape[-1] is not None and y_probs_batch.shape[-1] > 1:
      auc.update_state(tf.cast(y_true_batch, tf.float32), y_probs_batch[:, 1])
      auc_updated = True

    y_pred_chunks.append(y_pred_batch.numpy())

  y_pred_np = np.concatenate(y_pred_chunks, axis=0)
  acc = accuracy.result().numpy()
  auc_value = auc.result().numpy() if auc_updated else 0.0

  dp, _ = get_demographic_parity(y_pred_np, a_test_np)
  eo, _ = get_equalized_odds(y_pred_np, a_test_np, y_true_np)

  actual_positives = (y_true_np == 1)
  
  if np.sum(actual_positives) > 0:
    sensitivity = np.mean(y_pred_np[actual_positives])
  else:
    sensitivity = 0.0
  
  predicted_positives = (y_pred_np == 1)
  if np.sum(predicted_positives) > 0:
    precision = np.mean(y_true_np[predicted_positives])
  else:
    precision = 0.0

  if (precision + sensitivity) > 0:
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
  else:
    f1 = 0.0

  print(f'Test Accuracy: {acc:.3f}')
  print(f'Test Sensitivity: {sensitivity:.3f}')
  print(f'Test DP: {dp:.3f}')
  print(f'Test EO: {eo:.3f}')
  print(f'Test AUC: {auc_value:.3f}')
  print(f'Test F1-Score: {f1:.3f}')

  if show_confusion_matrix:
    display_confusion_matrix(
        y_true_np,
        y_pred_np,
        save_path='files/test_confusion_matrix.png',
        title='Test Set Confusion Matrix'
    )

  return {'accuracy': float(acc), 'dp': float(dp), 'eo': float(eo), 'sensitivity': float(sensitivity), 'auc': float(auc_value), 'f1': float(f1)}


def gauss_kernel(x1, x2, beta=1.0):
  assert len(x1.shape) == len(x2.shape)
  size = len(x1.shape)
  pairwise = tf.reduce_sum(
      (tf.expand_dims(x1, size - 1) - tf.expand_dims(x2, size - 2)) ** 2, size
  )
  return tf.exp(-0.5 * pairwise / beta)


def maximum_mean_discrepancy(x, y, kernel_scale=1.0):
  # Compute the pairwise squared Euclidean distances
  k_xx = gauss_kernel(x, x, beta=kernel_scale)
  k_yy = gauss_kernel(y, y, beta=kernel_scale)
  k_xy = gauss_kernel(x, y, beta=kernel_scale)

  # Compute the MMD loss
  mmd_loss = (
      tf.reduce_mean(k_xx) - 2.0 * tf.reduce_mean(k_xy) + tf.reduce_mean(k_yy)
  )
  return mmd_loss

def aggregate_fold_results(fold_results):
    """Aggregate metrics across folds."""
    train_acc = [f['train_accuracies'][-1] for f in fold_results if f.get('train_accuracies')]
    valid_folds = [f for f in fold_results if f.get('val_metrics')]

    def _metric_values(metric_name, default=0.0):
        values = []
        for fold in valid_folds:
            metric = fold['val_metrics']
            if metric_name in metric:
                values.append(metric[metric_name])
            else:
                values.append(metric.get(metric_name, default))
        return values

    val_acc = _metric_values('accuracy')
    val_dp = _metric_values('dp')
    val_sens = _metric_values('sensitivity')
    val_auc = _metric_values('auc')
    val_f1 = _metric_values('f1', 0.0)
    val_eo = _metric_values('eo')

    def _mean_or_nan(values):
        return float(np.mean(values)) if values else float('nan')

    def _std_or_nan(values):
        return float(np.std(values)) if values else float('nan')

    return {
        'mean_train_accuracy': _mean_or_nan(train_acc),
        'std_train_accuracy': _std_or_nan(train_acc),
        'mean_val_accuracy': _mean_or_nan(val_acc),
        'std_val_accuracy': _std_or_nan(val_acc),
        'mean_val_dp': _mean_or_nan(val_dp),
        'std_val_dp': _std_or_nan(val_dp),
        'mean_val_sensitivity': _mean_or_nan(val_sens),
        'std_val_sensitivity': _std_or_nan(val_sens),
        'mean_val_auc': _mean_or_nan(val_auc),
        'std_val_auc': _std_or_nan(val_auc),
        'mean_val_f1': _mean_or_nan(val_f1),
        'std_val_f1': _std_or_nan(val_f1),
        'mean_val_eo': _mean_or_nan(val_eo),
        'std_val_eo': _std_or_nan(val_eo),
    }


def _compute_dp_from_preds(preds, groups):
    groups_arr = np.array(groups)
    if groups_arr.size == 0:
        return 0.0
    preds_arr = np.array(preds)
    overall_rate = float(np.mean(preds_arr))
    max_abs_diff = 0.0
    for group_value in np.unique(groups_arr):
        group_mask = groups_arr == group_value
        if np.sum(group_mask) == 0:
            continue
        group_rate = float(np.mean(preds_arr[group_mask]))
        max_abs_diff = max(max_abs_diff, abs(overall_rate - group_rate))
    return float(max_abs_diff)


def _compute_window_fairness(y_preds_all, y_true_all, a_all, fairness_start, fairness_window):
    # Rolling-window fairness over the latest fairness_window samples.
    if fairness_window is None:
        fairness_window = len(y_preds_all)
    fairness_window = max(1, int(fairness_window))
    window_start = max(int(fairness_start), len(y_preds_all) - fairness_window)
    w_preds = y_preds_all[window_start:]
    w_true = y_true_all[window_start:]
    w_a = a_all[window_start:]

    w_a_arr = np.array(w_a)
    if np.unique(w_a_arr).size < 2:
        return 0.0, 0.0

    dp_val, _ = get_demographic_parity(w_preds, w_a)
    eo_val, _ = get_equalized_odds(w_preds, w_a, w_true)
    return float(dp_val), float(eo_val)


