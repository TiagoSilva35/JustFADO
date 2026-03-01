"""Commonly used Utilities."""

import math

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from river import drift
from src.helpers.plots import plot_metric_over_iterations


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
  """Display and save confusion matrix with detailed metrics.
  
  Args:
    y_true: Array of true labels.
    y_pred: Array of predicted labels.
    save_path: Path to save the confusion matrix figure.
    class_names: List of class names for display.
    title: Title for the confusion matrix plot.
    log_to_wandb: Whether to log to wandb.
    wandb_module: The wandb module if logging is enabled.
    
  Returns:
    cm: The confusion matrix array.
  """
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
    
    if (tp + fp > 0) and (tp + fn > 0):
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
  """Demographic parity.

  Args:
    y_predictions:
    y_protected:

  Returns:

  """
  predictions = np.array(y_predictions)
  protected_group = np.array(y_protected)

  protected_positive_rate = np.mean(predictions[protected_group == 1])
  unprotected_positive_rate = np.mean(predictions[protected_group == 0])

  raw_diff = protected_positive_rate - unprotected_positive_rate
  demographic_parity = np.abs(raw_diff)
  return demographic_parity, np.copysign(1, raw_diff)

def get_equalized_odds(y_predictions, y_protected, y_true):
  predictions = np.array(y_predictions)
  protected_group = np.array(y_protected)
  true_labels = np.array(y_true)

  mask_p1 = (protected_group == 1) & (true_labels == 1)
  mask_u1 = (protected_group == 0) & (true_labels == 1)
  mask_p0 = (protected_group == 1) & (true_labels == 0)
  mask_u0 = (protected_group == 0) & (true_labels == 0)

  # True Positive Rate (TPR) for protected and unprotected groups
  protected_tpr = np.mean(predictions[mask_p1]) if np.sum(mask_p1) > 0 else 0.0
  unprotected_tpr = np.mean(predictions[mask_u1]) if np.sum(mask_u1) > 0 else 0.0

  # False Positive Rate (FPR) for protected and unprotected groups
  protected_fpr = np.mean(predictions[mask_p0]) if np.sum(mask_p0) > 0 else 0.0
  unprotected_fpr = np.mean(predictions[mask_u0]) if np.sum(mask_u0) > 0 else 0.0

  tpr_diff = protected_tpr - unprotected_tpr
  fpr_diff = protected_fpr - unprotected_fpr

  # Return the max violation and the sign of the dominant term
  if np.abs(tpr_diff) >= np.abs(fpr_diff):
    equalized_odds = np.abs(tpr_diff)
    sign = np.copysign(1, tpr_diff)
  else:
    equalized_odds = np.abs(fpr_diff)
    sign = np.copysign(1, fpr_diff)

  return equalized_odds, sign


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
  
  # Get predicted classes for accuracy
  y_pred = tf.math.argmax(y_probs, axis=-1)

  accuracy.update_state(y_true, y_pred)
  acc = accuracy.result().numpy()
  
  # AUC needs probabilities of the positive class (class 1)
  auc.update_state(y_true, y_probs[:, 1])
  auc_value = auc.result().numpy()
  
  dp, _ = get_demographic_parity(y_pred, a_test)
  eo, _ = get_equalized_odds(y_pred, a_test, y_true.numpy())
  
  # Calculate true sensitivity (recall): TP / (TP + FN)
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

  # Display confusion matrix if requested
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
  """Computes the Maximum Mean Discrepancy (MMD) between two batches of samples.

  Args:
      x: Tensor of shape [batch_size, feature_dim] representing the first batch
        of samples.
      y: Tensor of shape [batch_size, feature_dim] representing the second batch
        of samples.
      kernel_scale: Float specifying the scale of the kernel.

  Returns:
      mmd_loss: Scalar tensor representing the MMD loss.
  """

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
    return {
        'mean_train_accuracy': np.mean([f['train_accuracies'][-1] for f in fold_results]),
        'std_train_accuracy': np.std([f['train_accuracies'][-1] for f in fold_results]),
        'mean_val_accuracy': np.mean([f['val_metrics']['accuracy'] for f in fold_results]),
        'std_val_accuracy': np.std([f['val_metrics']['accuracy'] for f in fold_results]),
        'mean_val_dp': np.mean([f['val_metrics']['dp'] for f in fold_results]),
        'std_val_dp': np.std([f['val_metrics']['dp'] for f in fold_results]),
        'mean_val_sensitivity': np.mean([f['val_metrics']['sensitivity'] for f in fold_results]),
        'std_val_sensitivity': np.std([f['val_metrics']['sensitivity'] for f in fold_results]),
        'mean_val_auc': np.mean([f['val_metrics']['auc'] for f in fold_results]),
        'std_val_auc': np.std([f['val_metrics']['auc'] for f in fold_results]),
        'mean_val_f1': np.mean([f['val_metrics'].get('f1', 0) for f in fold_results]),
        'std_val_f1': np.std([f['val_metrics'].get('f1', 0) for f in fold_results]),
        'mean_val_eo': np.mean([f['val_metrics']['eo'] for f in fold_results]),
        'std_val_eo': np.std([f['val_metrics']['eo'] for f in fold_results]),
    }

def evaluate_over_timesteps(model, x_test, y_test, a_test, data_dim):
    from collections import deque

    # outputs
    accuracies = []
    dps = []
    eos = []
    drifted_points = []
    drifted_points_dp = []
    drifted_points_eo = []

    # detectors
    drift_detector_acc = drift.ADWIN(delta=0.002)
    drift_detector_dp  = drift.ADWIN(delta=1e-5)
    drift_detector_eo  = drift.ADWIN(delta=1e-5)

    # window / warmup params (tune to your drift scale)
    WINDOW_SIZE = 2000
    WARMUP = 2000        # pause after a drift detection (samples)
    MIN_SAMPLES = 200    # minimum window size to evaluate
    MIN_PER_GROUP = 50   # minimum samples per sensitive-group within window

    # warmup markers (initial warmup should be >= WINDOW_SIZE or MIN_SAMPLES)
    dp_warmup_until = WARMUP
    eo_warmup_until = WARMUP

    # buffers for sliding-window computation
    y_preds_all = []
    y_true_all = []
    a_all = []

    # optional EMA smoothing for DP/EO before feeding detector
    use_ema = True
    ema_alpha = 0.2
    dp_ema = None
    eo_ema = None

    n_samples = len(x_test)
    accuracy_metric = tf.keras.metrics.Accuracy()

    for t in range(n_samples):
        x_t = np.array(x_test[t], dtype=np.float32).reshape(1, data_dim)
        y_probs = model(tf.convert_to_tensor(x_t), training=False)
        y_pred = int(tf.math.argmax(y_probs, axis=-1).numpy()[0])

        y_preds_all.append(y_pred)
        y_true_all.append(int(y_test[t]))
        a_all.append(int(a_test[t]))

        # --- accuracy (per-sample feed to detector) ---
        error = 1 if y_pred != int(y_test[t]) else 0
        drift_detector_acc.update(error)
        if drift_detector_acc.change_detected:
            drifted_points.append(t)
            print(f"[ACC] Drift detected at sample {t}")
            # optional: reset accuracy metric
            accuracy_metric = tf.keras.metrics.Accuracy()

        # windowed accuracy for plotting (simple moving avg over last WINDOW_SIZE)
        acc_start = max(0, t + 1 - WINDOW_SIZE)
        acc_window = y_preds_all[acc_start:]
        true_window = y_true_all[acc_start:]
        if len(acc_window) >= 1:
            # compute windowed accuracy (fast)
            acc_val = float(np.mean(np.array(acc_window) == np.array(true_window)))
        else:
            acc_val = 0.0
        accuracies.append(acc_val)

        # --- fairness window slices (bounded by WINDOW_SIZE and any custom window start) ---
        w_start = max(0, t + 1 - WINDOW_SIZE)
        w_preds = y_preds_all[w_start:]
        w_true  = y_true_all[w_start:]
        w_a     = a_all[w_start:]

        # check there are enough samples overall and per-group in the *window*
        if len(w_preds) >= MIN_SAMPLES and (w_a.count(0) >= MIN_PER_GROUP and w_a.count(1) >= MIN_PER_GROUP):
            dp_gap, _ = get_demographic_parity(w_preds, w_a)
            eo_gap, _ = get_equalized_odds(w_preds, w_a, w_true)

            # optional EMA smoothing
            if use_ema:
                dp_ema = dp_gap if dp_ema is None else (ema_alpha * dp_gap + (1 - ema_alpha) * dp_ema)
                eo_ema = eo_gap if eo_ema is None else (ema_alpha * eo_gap + (1 - ema_alpha) * eo_ema)
                dp_for_detector = float(dp_ema)
                eo_for_detector = float(eo_ema)
            else:
                dp_for_detector = float(dp_gap)
                eo_for_detector = float(eo_gap)

            # update detectors after warmup
            if t >= dp_warmup_until:
                drift_detector_dp.update(dp_for_detector)
                if drift_detector_dp.change_detected:
                    drifted_points_dp.append(t)
                    print(f"[DP] Drift detected at sample {t} (dp_window={dp_gap:.4f})")
                    # pause DP detection for WARMUP samples (avoid immediate re-fires)
                    dp_warmup_until = t + WARMUP
                    # optional: reset detector to clear its internal state
                    drift_detector_dp = drift.ADWIN(delta=1e-5)
                    dp_ema = None  # reset ema for new regime

            if t >= eo_warmup_until:
                drift_detector_eo.update(eo_for_detector)
                if drift_detector_eo.change_detected:
                    drifted_points_eo.append(t)
                    print(f"[EO] Drift detected at sample {t} (eo_window={eo_gap:.4f})")
                    eo_warmup_until = t + WARMUP
                    drift_detector_eo = drift.ADWIN(delta=1e-5)
                    eo_ema = None

            # store *windowed* metrics for plotting (not cumulative)
            dps.append(float(dp_gap))
            eos.append(float(eo_gap))
        else:
            # not enough data in window yet — append NaN or previous value to keep alignment
            dps.append(np.nan)
            eos.append(np.nan)

    # print summary
    if drifted_points:
        print(f"Accuracy drift detected at samples: {drifted_points}")
    else:
        print("No accuracy drift detected over the test set.")
    if drifted_points_dp:
        print(f"DP drift detected at samples: {drifted_points_dp}")
    else:
        print("No DP drift detected over the test set.")
    if drifted_points_eo:
        print(f"EO drift detected at samples: {drifted_points_eo}")
    else:
        print("No EO drift detected over the test set.")

    return {
        'accuracy': accuracies,
        'dp': dps,
        'eo': eos,
        'n_samples': n_samples,
        'drifted_points': drifted_points,
        'drifted_points_dp': drifted_points_dp,
        'drifted_points_eo': drifted_points_eo,
    }