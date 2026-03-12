"""Commonly used Utilities."""

import math

import numpy as np
import tensorflow as tf
import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from river import drift
from src.helpers.plots import plot_metrics_over_timesteps
from river.ensemble import AdaptiveRandomForestClassifier
from src.forest.initializers import init_fairness_state, accumulate_fairness_stats, compute_fairness_gradients


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

def evaluate_over_timesteps(model, x_test, y_test, a_test, data_dim,
                            test_then_train=True, learning_rate=2e-3,
                            accuracy_window=200,
                            compute_fairness=True, fairness_type='dp',
                            lambda_const=0.1, tree_depth=3, num_trees=3,
                            constraint_type='node', gradient_type='vanilla',
                            base_gamma=0.9):
    accuracies = []
    dps = []
    eos = []
    drifted_points = []

    ADWIN_DELTA_WARN    = 0.05   # fires early (fast but noisier)
    ADWIN_DELTA_CONFIRM = 0.002   # confirmed drift (slower but reliable)
    DRIFT_LR_PREWARM    = learning_rate * 2   # gentle warm-up on warning
    DRIFT_LR_SPIKE      = learning_rate * 5  # full spike on confirmed drift
    LR_DECAY_STEPS      = 500
    FAIRNESS_WINDOW     = 500
    COOLDOWN            = 200    
    MIN_SAMPLES_PER_STREAM = 3

    print(f"Evaluating model over {len(x_test)} timesteps with test-then-train={test_then_train}\n\
          Fairness penalty lambda: {lambda_const}, fairness type: {fairness_type}")


    USE_ROLLING = bool(accuracy_window)
    correct_buffer = []
    warn_det  = drift.ADWIN(delta=ADWIN_DELTA_WARN)
    acc_det   = drift.ADWIN(delta=ADWIN_DELTA_CONFIRM)
    acc_det_n = 0
    in_warning = False           
    last_detected_acc = -COOLDOWN
    y_preds_all = []
    y_true_all = []
    a_all = []
    n_samples = len(x_test)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    steps_since_drift = 0
    fairness_start = 0

    if compute_fairness:
        num_internal_nodes = 2 ** tree_depth - 1
        all_tree_trainable_vars = []
        for tree in model.layers:
            all_tree_trainable_vars.extend(tree.trainable_variables)
        gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count = \
            init_fairness_state(num_trees, data_dim, num_internal_nodes)

    for t in range(n_samples):
        x_t = tf.convert_to_tensor(
            np.array(x_test[t], dtype=np.float32).reshape(1, data_dim)
        )
        y_t = int(y_test[t])
        a_t = int(a_test[t])

        y_probs = model(x_t, training=False)
        y_pred  = int(tf.math.argmax(y_probs, axis=-1).numpy()[0])
        error   = int(y_pred != y_t)

        y_preds_all.append(y_pred)
        y_true_all.append(y_t)
        a_all.append(a_t)

        correct_buffer.append(int(y_pred == y_t))
        if USE_ROLLING and len(correct_buffer) > accuracy_window:
            correct_buffer.pop(0)
        acc_val = float(sum(correct_buffer)) / len(correct_buffer)
        accuracies.append(acc_val)

        w_start = max(fairness_start, len(y_preds_all) - FAIRNESS_WINDOW)
        w_preds = y_preds_all[w_start:]
        w_true  = y_true_all[w_start:]
        w_a     = a_all[w_start:]
        y_probs_np  = tf.nn.softmax(y_probs, axis=-1).numpy()[0]
        model_conf  = float(y_probs_np[y_pred])   
        label_conf  = float(y_probs_np[y_t])      
        is_label_noise = (error == 1) and (model_conf > 0.70)

        warn_det.update(error)
        acc_det.update(error)
        acc_det_n += 1

        if (warn_det.change_detected
                and not in_warning
                and acc_det_n >= MIN_SAMPLES_PER_STREAM
                and t - last_detected_acc >= COOLDOWN
                and not is_label_noise):
            in_warning = True
            optimizer.learning_rate.assign(float(DRIFT_LR_PREWARM))
            steps_since_drift = LR_DECAY_STEPS
            warn_det  = drift.ADWIN(delta=ADWIN_DELTA_WARN)
            print(f"[WARN] Drift warning at sample {t} — pre-warming LR to {DRIFT_LR_PREWARM:.2e}")

        if (acc_det.change_detected
                and acc_det_n >= MIN_SAMPLES_PER_STREAM
                and t - last_detected_acc >= COOLDOWN):
            drifted_points.append(t)
            last_detected_acc = t
            in_warning = False
            acc_det   = drift.ADWIN(delta=ADWIN_DELTA_CONFIRM)
            warn_det  = drift.ADWIN(delta=ADWIN_DELTA_WARN)
            acc_det_n = 0
            optimizer = tf.keras.optimizers.Adam(learning_rate=DRIFT_LR_SPIKE)
            steps_since_drift = LR_DECAY_STEPS
            print(f"[DRIFT] Concept drift confirmed at sample {t} — spiking LR to {DRIFT_LR_SPIKE:.2e}")
            if compute_fairness:
                gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count = \
                    init_fairness_state(num_trees, data_dim, num_internal_nodes)
                fairness_start = len(y_preds_all)
                print(f"[DRIFT] Fairness state reset at sample {t}")
        w_a_arr = np.array(w_a)
        if np.sum(w_a_arr == 0) > 0 and np.sum(w_a_arr == 1) > 0:
            dp_val, _ = get_demographic_parity(w_preds, w_a)
            eo_val, _ = get_equalized_odds(w_preds, w_a, w_true)
        else:
            dp_val = 0.0
            eo_val = 0.0

        dps.append(float(dp_val))
        eos.append(float(eo_val))
        if steps_since_drift > 0:
            alpha = steps_since_drift / LR_DECAY_STEPS
            current_lr = learning_rate + alpha * (DRIFT_LR_SPIKE - learning_rate)
            optimizer.learning_rate.assign(float(current_lr))
            steps_since_drift -= 1

            sample_weight = max(label_conf, 1.0 - model_conf + 1e-6)
            sample_weight = float(np.clip(sample_weight, 0.05, 1.0))

            y_t_tensor = tf.convert_to_tensor([y_t], dtype=tf.int32)
            sw_tensor  = tf.constant([sample_weight], dtype=tf.float32)
            with tf.GradientTape(persistent=compute_fairness) as tape:
                train_out = model(x_t, training=True)
                y_probs_train = train_out[0] if isinstance(train_out, tuple) else train_out
                node_decisions_train = train_out[1] if isinstance(train_out, tuple) else None
                loss = criteria(y_true=y_t_tensor, y_pred=y_probs_train,
                               sample_weight=sw_tensor)
                if compute_fairness and node_decisions_train is not None:
                    accumulate_fairness_stats(
                        tape, [a_t], [y_t],
                        node_decisions_train, y_probs_train,
                        all_tree_trainable_vars, model.trainable_variables,
                        gradient_w, gradient_b, agg_y,
                        subgroup_count, protected_class_count,
                        num_internal_nodes, data_dim,
                        constraint_type, gradient_type, base_gamma,
                    )
                grads = tape.gradient(loss, model.trainable_variables)
                if compute_fairness and node_decisions_train is not None:
                    grads = compute_fairness_gradients(
                        grads, gradient_w, gradient_b, agg_y,
                        subgroup_count, protected_class_count,
                        fairness_type, lambda_const,
                        num_internal_nodes, data_dim, num_trees,
                        gradient_type, base_gamma,
                    )
            if compute_fairness:
                del tape

            assert len(grads) == len(model.trainable_variables) and len(grads) > 0, "Problem with loss gradients"
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if drifted_points:
        print(f"Accuracy drift detected at samples: {drifted_points}")
    else:
        print("No accuracy drift detected over the test set.")

    return {
        'accuracy': accuracies,
        'dp': dps,
        'eo': eos,
        'n_samples': n_samples,
        'drifted_points': drifted_points,
    }


def evaluate_arf_over_timesteps(x_test, y_test, a_test, accuracy_window=200):
    arf = AdaptiveRandomForestClassifier(seed=42)

    accuracies = []
    dps = []
    eos = []

    FAIRNESS_WINDOW = 500
    correct_buffer = []
    USE_ROLLING = bool(accuracy_window)
    y_preds_all = []
    y_true_all = []
    a_all = []
    n_samples = len(x_test)

    print(f"Running ARF baseline prequentially on {n_samples} samples...")

    for t in range(n_samples):
        x_t = np.array(x_test[t], dtype=np.float32)
        y_t = int(y_test[t])
        a_t = int(a_test[t])

        # River expects features as a plain dict keyed by feature index
        x_dict = {i: float(v) for i, v in enumerate(x_t)}

        # ── STEP 1: TEST ──────────────────────────────────────────────────
        y_pred = arf.predict_one(x_dict)
        y_pred = int(y_pred) if y_pred is not None else 0

        y_preds_all.append(y_pred)
        y_true_all.append(y_t)
        a_all.append(a_t)

        correct_buffer.append(int(y_pred == y_t))
        if USE_ROLLING and len(correct_buffer) > accuracy_window:
            correct_buffer.pop(0)
        accuracies.append(float(sum(correct_buffer)) / len(correct_buffer))

        # Fairness over rolling window
        w_start = max(0, t + 1 - FAIRNESS_WINDOW)
        w_preds = y_preds_all[w_start:]
        w_true  = y_true_all[w_start:]
        w_a     = a_all[w_start:]
        w_a_arr = np.array(w_a)
        if np.sum(w_a_arr == 0) > 0 and np.sum(w_a_arr == 1) > 0:
            dp_val, _ = get_demographic_parity(w_preds, w_a)
            eo_val, _ = get_equalized_odds(w_preds, w_a, w_true)
        else:
            dp_val = 0.0
            eo_val = 0.0
        dps.append(float(dp_val))
        eos.append(float(eo_val))

        # ── STEP 2: TRAIN ─────────────────────────────────────────────────
        arf.learn_one(x_dict, y_t)

    print("ARF baseline evaluation complete.")
    return {
        'accuracy': accuracies,
        'dp': dps,
        'eo': eos,
        'n_samples': n_samples,
        'drifted_points': [],  
    }


