"""Script for online training."""
import numpy as np
import tensorflow as tf
import tqdm
import src.models.forest.initializers as initializers
import src.helpers.utils as utils
from src.helpers.weight_monitor import ClassWeightMonitor


SUPPORTED_FAIRNESS_TYPES = ['dp', 'eo']

def train_online(
    model,
    inputs,
    targets,
    protected_targets,
    data_dim=13,
    batch_size=1,
    tree_depth=3,
    compute_fairness=True,
    lambda_const=0.3,
    num_trees=3,
    base_gamma=0.9,
    constraint_type='node',
    gradient_type='vanilla',
    local_run=False,
    fairness_type='dp',
    fairness_window=1000,
):
  if fairness_type not in SUPPORTED_FAIRNESS_TYPES:
    raise ValueError(f"Fairness type {fairness_type} not supported. Choose from {SUPPORTED_FAIRNESS_TYPES}.")
  else:
    print(f"Fairness type: {fairness_type}")

  weight_updater = ClassWeightMonitor(
    total_num_samples=0,
    num_classes=2,
    alpha=1.0,
    _lambda=0.99,
  )
  weight_updater = None

  print(f"lambda for fairness penalty: {lambda_const}")

  print("weight updater enabled:", weight_updater is not None)
  # creates a tf dataset
  dataset = tf.data.Dataset.from_tensor_slices(
      (inputs, targets, protected_targets)
  )
  dataset = dataset.batch(batch_size)

  # number of internal nodes in a binary tree
  num_internal_nodes = 2**tree_depth - 1
  number_of_attributes = int(np.unique(np.array(protected_targets)).size)

  # choose the optimizer and loss criteria
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
  # During training the forest returns raw pre-softmax logits, so use from_logits=True
  criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  avg_accuracy = tf.keras.metrics.Accuracy()

  gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count = \
      initializers.init_fairness_state(num_trees, data_dim, num_internal_nodes, number_of_attributes)

  avg_loss = tf.keras.metrics.Mean()
  avg_auc = tf.keras.metrics.AUC()
  y_predictions = []
  y_true_all = []
  demographic_parities = []
  equalized_odds = []
  accuracies = []
  
  # Incremental rolling statistics avoid rescanning the full window.
  fairness_window_state = utils.RollingFairnessWindow(fairness_window)

  # hyperparameters
  huber_loss_delta = 0.1
  all_tree_trainable_vars = []
  for tree in model.layers:
    all_tree_trainable_vars.extend(tree.trainable_variables)

  iterations = tqdm.tqdm(dataset)
  for _, (
      inputs_batch,
      targets_batch,
      protected_batch) in enumerate(iterations):
    # ``persistent`` is only needed when ``.gradient`` is called more than once.
    # The 'node' fairness gradient is now analytic (see
    # ``initializers.accumulate_fairness_stats``), so the only tape traversal
    # left is the task loss below; 'leaf' still needs a second pass.
    with tf.GradientTape(
        persistent=(compute_fairness and constraint_type == 'leaf')
    ) as tape:
      predictions, node_decisions, per_tree_predictions = model(inputs_batch, training=True)
      y_pred = tf.math.argmax(predictions, axis=-1)
      if weight_updater is None:
        target_loss = criteria(y_true=targets_batch, y_pred=predictions)
      else:
        weight_updater.update_weights(sample_label=targets_batch.numpy()[0])
        class_weights = tf.gather(weight_updater.class_weights, targets_batch)
        target_loss = criteria(y_true=targets_batch, y_pred=predictions, sample_weight=class_weights)
        weight_updater.total_num_samples += 1

      # B2 fix (2026-06): per-sample slicing MUST happen inside the tape's
      # with-block so the slice ops are recorded. Slicing the batch tensors
      # outside the tape produces untracked tensors that make
      # ``tape.gradient`` return None for every var, silently disabling the
      # fairness regulariser. See DOCS/BUG_REPORT_fairness_regulariser.md.
      if compute_fairness:
        node_decisions_per_sample = tf.unstack(node_decisions, axis=1)
        predictions_per_sample = tf.unstack(predictions, axis=0)
      else:
        node_decisions_per_sample = None
        predictions_per_sample = None

      y_pred_np = y_pred.numpy()
      targets_np = targets_batch.numpy()
      protected_np = protected_batch.numpy()
      
      # Maintain full history for validation later, rolling window for fairness metrics
      y_predictions.extend(y_pred_np)
      y_true_all.extend(targets_np)
      
      for i in range(len(y_pred_np)):
        fairness_window_state.append(
          y_pred_np[i], protected_np[i], targets_np[i]
        )
      
      # Compute fairness on rolling window only (no O(n²) recomputation)
      dp, dp_sign = fairness_window_state.demographic_parity()
      eo, eo_sign = fairness_window_state.equalized_odds()

      demographic_parities.append(dp)
      equalized_odds.append(eo)
      accuracies.append(avg_accuracy.result().numpy())

      avg_accuracy.update_state(targets_batch, y_pred)
      probs_for_metrics = tf.nn.softmax(predictions, axis=-1)
      avg_auc.update_state(targets_batch, probs_for_metrics[:, 1])  
      avg_loss.update_state(target_loss)
      iterations.set_description(
          f' CE Loss: {avg_loss.result():.3f},'
          f' Accuracy: { avg_accuracy.result():.3%},'
          f' AUC: {avg_auc.result():.3%},'
          f' DP: {dp:.5f},'
          f' EO: {eo:.5f},'
      )
      results = {
        'CE Loss': float(avg_loss.result()),
        'Accuracy': float(avg_accuracy.result()),
        'AUC': float(avg_auc.result()),
        'DP': float(dp),
        'EO': float(eo),
      }

    gradients = tape.gradient(target_loss, model.trainable_variables)
    total_gradients = gradients
    if compute_fairness:
      initializers.accumulate_fairness_stats(
          tape, protected_batch.numpy(), targets_batch.numpy(),
          node_decisions_per_sample, predictions_per_sample,
          all_tree_trainable_vars, model.trainable_variables,
          gradient_w, gradient_b, agg_y,
          subgroup_count, protected_class_count,
          num_internal_nodes, data_dim,
          constraint_type, gradient_type, base_gamma,
          inputs=inputs_batch, model=model,
      )
      total_gradients = initializers.compute_fairness_gradients(
          gradients,
          gradient_w, gradient_b, agg_y,
          subgroup_count, protected_class_count,
          fairness_type, lambda_const,
          num_internal_nodes, data_dim, number_of_attributes,
          gradient_type, base_gamma, huber_loss_delta, dp_sign=dp_sign, constraint_type=constraint_type
      )
    del tape
    optimizer.apply_gradients(zip(total_gradients, model.trainable_variables))

  y_true_array = np.array(y_true_all)
  y_pred_array = np.array(y_predictions)

  cm = utils.display_confusion_matrix(
      y_true_array,
      y_pred_array,
      save_path='files/train_confusion_matrix.png',
      title='Training Set Final Confusion Matrix',
      log_to_wandb=(not local_run),
  )


  return demographic_parities, equalized_odds, accuracies
