"""Commonly used Utilities."""

import math

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


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


def get_demographic_parity(y_predictions, y_protected):
  predictions = np.array(y_predictions)
  protected_group = np.array(y_protected)

  if predictions.size == 0 or protected_group.size == 0:
    return 0.0, 1.0

  unique_groups = np.unique(protected_group)
  if unique_groups.size == 0:
    return 0.0, 1.0

  # Keep parity reporting aligned with the fairness objective:
  # compare each group against the unweighted mean of group rates.
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


def get_test_performance(model, x_test, y_test, a_test, data_dim,
                        show_confusion_matrix=True):
  accuracy = tf.keras.metrics.Accuracy()
  auc = tf.keras.metrics.AUC()

  y_true = tf.convert_to_tensor(np.array(y_test))
  
  # Get probabilities from model
  y_probs = model(
      tf.convert_to_tensor(
          np.array(x_test, dtype=np.float32).reshape(-1, data_dim)
      ),
      training=False  
  )
  
  y_pred = tf.math.argmax(y_probs, axis=-1)

  accuracy.update_state(y_true, y_pred)
  acc = accuracy.result().numpy()
  
  auc.update_state(y_true, y_probs[:, 1])
  auc_value = auc.result().numpy()
  
  dp, _ = get_demographic_parity(y_pred, a_test)
  eo, _ = get_equalized_odds(y_pred, a_test, y_true.numpy())
  
  y_true_np = y_true.numpy()
  y_pred_np = y_pred.numpy()
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
        y_true.numpy(), 
        y_pred.numpy(), 
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


