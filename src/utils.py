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
    }
      