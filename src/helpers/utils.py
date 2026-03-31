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

def evaluate_over_timesteps(model, x_test, y_test, a_test, data_dim,
                            test_then_train=True, learning_rate=2e-3,
                            accuracy_window=200,
                            compute_fairness=True, fairness_type='dp',
                            lambda_const=0.1, tree_depth=3, num_trees=3,
                            constraint_type='node', gradient_type='vanilla',
                            base_gamma=0.9,
                            static_params=None):
    accuracies = []
    dps = []
    eos = []
    drifted_points = []

    defaults = {
        'adwin_delta_warn': 0.00001,
        'adwin_delta_confirm': 0.02,
        'drift_lr_prewarm_mult': 5.0,
        'drift_lr_spike_mult': 10.0,
        'lr_decay_steps': 3000,
        'fairness_window': 500,
        'cooldown': 200,
        'min_samples_per_stream': 30,
        'lambda_const': float(lambda_const),
    }
    if static_params:
        print("Overriding default static parameters with provided values:")
        for key, value in static_params.items():
            print(f"  {key}: {value}")
        defaults.update(static_params)

    ADWIN_DELTA_WARN = float(defaults['adwin_delta_warn'])
    ADWIN_DELTA_CONFIRM = float(defaults['adwin_delta_confirm'])
    DRIFT_LR_PREWARM = learning_rate * float(defaults['drift_lr_prewarm_mult'])
    DRIFT_LR_SPIKE = learning_rate * float(defaults['drift_lr_spike_mult'])
    LR_DECAY_STEPS = max(1, int(defaults['lr_decay_steps']))
    FAIRNESS_WINDOW = max(1, int(defaults['fairness_window']))
    COOLDOWN = max(0, int(defaults['cooldown']))
    MIN_SAMPLES_PER_STREAM = max(1, int(defaults['min_samples_per_stream']))
    lambda_const = float(defaults['lambda_const'])
    print(f"Evaluating model over {len(x_test)} timesteps with test-then-train={test_then_train}\n\
          Fairness penalty lambda: {lambda_const}, fairness type: {fairness_type}")
    USE_ROLLING = bool(accuracy_window)
    print(f"Parameters:\n  ADWIN_DELTA_WARN: {ADWIN_DELTA_WARN}\n  ADWIN_DELTA_CONFIRM: {ADWIN_DELTA_CONFIRM}\n\  DRIFT_LR_PREWARM: {DRIFT_LR_PREWARM}\n  DRIFT_LR_SPIKE: {DRIFT_LR_SPIKE}\n  LR_DECAY_STEPS: {LR_DECAY_STEPS}\n  FAIRNESS_WINDOW: {FAIRNESS_WINDOW}\n  COOLDOWN: {COOLDOWN}\n  MIN_SAMPLES_PER_STREAM: {MIN_SAMPLES_PER_STREAM}\n  Lambda for fairness penalty: {lambda_const}\n  Accuracy window: {accuracy_window}\n  Use rolling accuracy: {USE_ROLLING}\n  Compute fairness: {compute_fairness}\n  Fairness type: {fairness_type}")
    correct_buffer = []
    warn_det  = drift.ADWIN(delta=ADWIN_DELTA_WARN)
    acc_det   = drift.ADWIN(delta=ADWIN_DELTA_CONFIRM)
    acc_det_n = 0
    in_warning = False           
    last_detected_acc = -COOLDOWN
    recovering_from_drift = False
    baseline_accuracy = 0.0
    y_preds_all = []
    y_true_all = []
    a_all = []
    n_samples = len(x_test)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    steps_since_drift = 0
    decay_from_lr = float(DRIFT_LR_PREWARM)
    fairness_start = 0

    if compute_fairness:
        num_internal_nodes = 2 ** tree_depth - 1
        all_tree_trainable_vars = []
        for tree in model.layers:
            all_tree_trainable_vars.extend(tree.trainable_variables)
        number_of_attributes = int(np.unique(np.array(a_test)).size)
        gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count = \
            init_fairness_state(num_trees, data_dim, num_internal_nodes, number_of_attributes)

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

        
        
        warn_det.update(error) #type: ignore
        acc_det.update(error) #type: ignore
        acc_det_n += 1

        if (warn_det.change_detected
                and not in_warning
                and acc_det_n >= MIN_SAMPLES_PER_STREAM
                and t - last_detected_acc >= COOLDOWN
                and not is_label_noise):
            in_warning = True
            optimizer.learning_rate.assign(float(DRIFT_LR_PREWARM))
            decay_from_lr = float(DRIFT_LR_PREWARM)
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
            baseline_accuracy = np.mean(accuracies[max(0, t - 1000):t]) if t > 1000 else np.mean(accuracies)
            recovering_from_drift = True
            
            print(f"[DRIFT] Concept drift confirmed at sample {t} — spiking LR to {DRIFT_LR_SPIKE:.2e} and making hard routing decisions")
            for tree in model.layers:
                if hasattr(tree, 'temperature'):
                    tree.temperature.assign(0.1)
        dp_val, eo_val = _compute_window_fairness(
            y_preds_all=y_preds_all,
            y_true_all=y_true_all,
            a_all=a_all,
            fairness_start=fairness_start,
            fairness_window=FAIRNESS_WINDOW,
        )

        dps.append(float(dp_val))
        eos.append(float(eo_val))
        
        if steps_since_drift > 0 and not recovering_from_drift:
            alpha = steps_since_drift / LR_DECAY_STEPS
            current_lr = learning_rate + alpha * (decay_from_lr - learning_rate)
            optimizer.learning_rate.assign(float(current_lr))
            steps_since_drift -= 1
        elif recovering_from_drift:
            current_acc = acc_val
            if current_acc >= baseline_accuracy:
                recovering_from_drift = False
                decay_from_lr = float(optimizer.learning_rate)
                steps_since_drift = LR_DECAY_STEPS
                for tree in model.layers:
                    if hasattr(tree, 'temperature'):
                        tree.temperature.assign(1.0)
                print(f"[RECOVERY] Performance restored at sample {t}. Decaying LR from {decay_from_lr:.2e} to {learning_rate:.2e} over {LR_DECAY_STEPS} steps.")
            else:
                optimizer.learning_rate.assign(float(DRIFT_LR_SPIKE))
                for tree in model.layers:
                    if hasattr(tree, 'temperature'):
                        current_temp = float(tree.temperature.value())
                        new_temp = min(1.0, current_temp + 0.002)
                        tree.temperature.assign(new_temp)

        if test_then_train:
            y_t_tensor = tf.convert_to_tensor([y_t], dtype=tf.int32)
            with tf.GradientTape(persistent=compute_fairness) as tape:
                train_out = model(x_t, training=True)
                y_probs_train = train_out[0] if isinstance(train_out, tuple) else train_out
                node_decisions_train = train_out[1] if isinstance(train_out, tuple) else None
                loss = criteria(y_true=y_t_tensor, y_pred=y_probs_train)
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
                        num_internal_nodes, data_dim, number_of_attributes,
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
        'static_params_used': {
            'adwin_delta_warn': ADWIN_DELTA_WARN,
            'adwin_delta_confirm': ADWIN_DELTA_CONFIRM,
            'drift_lr_prewarm_mult': float(defaults['drift_lr_prewarm_mult']),
            'drift_lr_spike_mult': float(defaults['drift_lr_spike_mult']),
            'lr_decay_steps': LR_DECAY_STEPS,
            'fairness_window': FAIRNESS_WINDOW,
            'cooldown': COOLDOWN,
            'min_samples_per_stream': MIN_SAMPLES_PER_STREAM,
            'lambda_const': lambda_const,
        },
    }


def evaluate_arf_over_timesteps(x_test, y_test, a_test, accuracy_window=200):
    arf = AdaptiveRandomForestClassifier(seed=42, n_models=3, max_depth=3)

    accuracies = []
    dps = []
    eos = []

    FAIRNESS_WINDOW = 1
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
        dp_val, eo_val = _compute_window_fairness(
            y_preds_all=y_preds_all,
            y_true_all=y_true_all,
            a_all=a_all,
            fairness_start=0,
            fairness_window=FAIRNESS_WINDOW,
        )
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


def _choose_group_threshold_for_target_dp(probs_window, groups_window, target_dp,
                                          unprotected_threshold=0.5,
                                          threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.linspace(0.05, 0.95, 37)

    best_threshold = float(unprotected_threshold)
    best_gap = float('inf')
    best_deviation = float('inf')

    probs_arr = np.array(probs_window, dtype=float)
    groups_arr = np.array(groups_window, dtype=int)

    for protected_threshold in threshold_grid:
        preds = [
            int(
                p >= (protected_threshold if g == 1 else unprotected_threshold)
            )
            for p, g in zip(probs_arr, groups_arr)
        ]
        candidate_dp = _compute_dp_from_preds(preds, groups_arr)
        gap = abs(candidate_dp - float(target_dp))
        deviation = abs(float(protected_threshold) - float(unprotected_threshold))

        if (gap < best_gap) or (np.isclose(gap, best_gap) and deviation < best_deviation):
            best_gap = gap
            best_deviation = deviation
            best_threshold = float(protected_threshold)

    return best_threshold


def evaluate_fair_arf_over_timesteps(x_test, y_test, a_test, target_dp_series,
                                     accuracy_window=200, fairness_window=1,
                                     unprotected_threshold=0.5,
                                     threshold_grid=None):
    arf = AdaptiveRandomForestClassifier(seed=42, n_models=3, max_depth=3)

    accuracies = []
    dps = []
    eos = []
    dp_target_errors = []
    protected_thresholds = []

    correct_buffer = []
    USE_ROLLING = bool(accuracy_window)
    y_preds_all = []
    y_true_all = []
    a_all = []
    p1_all = []
    n_samples = len(x_test)

    print(f"Running Fair-ARF (DP-targeted) prequentially on {n_samples} samples...")

    for t in range(n_samples):
        x_t = np.array(x_test[t], dtype=np.float32)
        y_t = int(y_test[t])
        a_t = int(a_test[t])

        x_dict = {i: float(v) for i, v in enumerate(x_t)}

        proba = arf.predict_proba_one(x_dict)
        p1 = float(proba.get(1, 0.0)) if proba is not None else 0.0
        p1_all.append(p1)
        a_all.append(a_t)

        target_dp = float(target_dp_series[t]) if t < len(target_dp_series) else float(target_dp_series[-1])

        probs_window = p1_all
        groups_window = a_all

        protected_threshold = _choose_group_threshold_for_target_dp(
            probs_window=probs_window,
            groups_window=groups_window,
            target_dp=target_dp,
            unprotected_threshold=unprotected_threshold,
            threshold_grid=threshold_grid,
        )
        protected_thresholds.append(float(protected_threshold))

        chosen_threshold = protected_threshold if a_t == 1 else unprotected_threshold
        y_pred = int(p1 >= chosen_threshold)

        y_preds_all.append(y_pred)
        y_true_all.append(y_t)

        correct_buffer.append(int(y_pred == y_t))
        if USE_ROLLING and len(correct_buffer) > accuracy_window:
            correct_buffer.pop(0)
        accuracies.append(float(sum(correct_buffer)) / len(correct_buffer))

        dp_val, eo_val = _compute_window_fairness(
            y_preds_all=y_preds_all,
            y_true_all=y_true_all,
            a_all=a_all,
            fairness_start=0,
            fairness_window=fairness_window,
        )
        dps.append(float(dp_val))
        eos.append(float(eo_val))
        dp_target_errors.append(abs(float(dp_val) - target_dp))

        arf.learn_one(x_dict, y_t)

    print("Fair-ARF evaluation complete.")
    return {
        'accuracy': accuracies,
        'dp': dps,
        'eo': eos,
        'n_samples': n_samples,
        'drifted_points': [],
        'dp_target_errors': dp_target_errors,
        'protected_thresholds': protected_thresholds,
    }
