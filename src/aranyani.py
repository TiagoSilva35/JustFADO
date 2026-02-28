"""Script for online training."""
import collections
import numpy as np
import tensorflow as tf
import tqdm
import wandb
import utils
from weight_monitor import ClassWeightMonitor

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
    fairness_type='eo',
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

  print(f"lambda for fairness penalty: {lambda_const}")

  print("weight updater enabled:", weight_updater is not None)
  # creates a tf dataset
  dataset = tf.data.Dataset.from_tensor_slices(
      (inputs, targets, protected_targets)
  )
  dataset = dataset.batch(batch_size)

  # number of internal nodes in a binary tree
  num_internal_nodes = 2**tree_depth - 1

  # choose the optimizer and loss criteria
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
  criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

  avg_accuracy = tf.keras.metrics.Accuracy()

  gradient_w = {(a, y): np.zeros([num_trees, data_dim, num_internal_nodes])
                for a in [0, 1] for y in [0, 1]}

  gradient_b = {(a, y): np.zeros([num_trees, num_internal_nodes])
                for a in [0, 1] for y in [0, 1]}

  agg_y = {(a, y): np.zeros([num_trees, num_internal_nodes])
                for a in [0, 1] for y in [0, 1]}

  average_w_fair_grad = []
  average_b_fair_grad = []




  # avoid runtime error
  subgroup_count = collections.defaultdict(int)
  protected_class_count = collections.defaultdict(int)
  dp_function = utils.get_demographic_parity
  avg_loss = tf.keras.metrics.Mean()
  avg_auc = tf.keras.metrics.AUC()
  y_predictions = []
  y_true_all = []
  demographic_parities = []
  equalized_odds = []
  accuracies = []

  # hyperparameters
  huber_loss_delta = 0.1

  # Collect all tree trainable variables for fairness gradients
  # (model.layers is a Python list, so we need to manually collect)
  all_tree_trainable_vars = []
  for tree in model.layers:
    all_tree_trainable_vars.extend(tree.trainable_variables)

  iterations = tqdm.tqdm(dataset)
  for _, (
      inputs_batch,
      targets_batch,
      protected_batch) in enumerate(iterations):
    with tf.GradientTape(persistent=True) as tape:
      # predictions: [batch_size, num_class]
      # y: [num_trees, batch_size, num_internal_nodes]
      predictions, node_decisions = model(inputs_batch, training=True)

      # y_pred: [batch_size]
      y_pred = tf.math.argmax(predictions, axis=-1)
      # Apply class-balanced loss if enabled, otherwise standard CE

      if weight_updater is None:
        target_loss = criteria(y_true=targets_batch, y_pred=predictions)
      else:
        weight_updater.update_weights(sample_label=targets_batch.numpy()[0])
        class_weights = tf.gather(weight_updater.class_weights, targets_batch)
        target_loss = criteria(y_true=targets_batch, y_pred=predictions, sample_weight=class_weights)
        weight_updater.total_num_samples += 1
        # weight_updater.get_stats()
      # get demographic parity scores

      y_predictions.extend(y_pred.numpy())
      y_true_all.extend(targets_batch.numpy())  # Track true labels
      dp, dp_sign = dp_function(
          y_predictions, protected_targets[: len(y_predictions)])
      eo, eo_sign = utils.get_equalized_odds(
          y_predictions, y_true_all, protected_targets[: len(y_predictions)])


      demographic_parities.append(dp)
      equalized_odds.append(eo)
      accuracies.append(avg_accuracy.result().numpy())

      # update the average accuracy
      avg_accuracy.update_state(targets_batch, y_pred)
      avg_auc.update_state(targets_batch, y_pred)
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
      if not local_run:
        wandb.log(results)

      gradients = tape.gradient(target_loss, model.trainable_variables)
      total_gradients = gradients
      if compute_fairness:
        for i, a_label in enumerate(protected_batch.numpy()):
          a_label = int(a_label)
          y_label = int(targets_batch.numpy()[i])
          protected_class_count[a_label] += 1
          subgroup_count[(a_label, y_label)] += 1

          agg_y[(a_label, y_label)] += node_decisions[:, i].numpy()

          if constraint_type == 'node':
            fair_gradients = tape.gradient(node_decisions[:, i],
                                           all_tree_trainable_vars)
          elif constraint_type == 'leaf':
            fair_gradients = tape.gradient(
                predictions[i], model.trainable_variables
            )
          else:
            raise ValueError('Constraint type not identified.')

          idx_b = -1
          idx_w = -1
          for fair_grad in fair_gradients:
            if fair_grad is None:
              continue
            if len(fair_grad.shape) == 1 and fair_grad.shape[0] == num_internal_nodes:
              idx_b += 1
              gradient_theta = gradient_b[(a_label, y_label)]
              idx_theta = idx_b
            elif len(fair_grad.shape) == 2 and fair_grad.shape[0] == data_dim:
              gradient_theta = gradient_w[(a_label, y_label)]
              idx_w += 1
              idx_theta = idx_w
            else:
              continue
            if gradient_type == 'momentum':
              gradient_theta[idx_theta] = (
                  gradient_theta[idx_theta] * base_gamma
                  + fair_grad.numpy()
              )
            elif gradient_type == 'ema':
              gradient_theta[idx_theta] = (
                  gradient_theta[idx_theta] * base_gamma
                  + fair_grad.numpy()
              )
            else:
              factor = 1 / subgroup_count[(a_label, y_label)]
              gradient_theta[idx_theta] = (
                  gradient_theta[idx_theta] * (1 - factor)
                  + fair_grad.numpy() * factor
              )
        if fairness_type == 'dp':
          can_compute = protected_class_count[0] > 0 and protected_class_count[1] > 0
        else:
          can_compute = all(subgroup_count[(a, y)] > 0 for a in [0, 1] for y in [0, 1])

        if not can_compute:
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            continue

        if protected_class_count[0] == 0 or protected_class_count[1] == 0:
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
          continue

        if gradient_type == 'ema':
          if fairness_type == 'dp':
            correction_factor = {
              a: (1 - base_gamma)/(1 - base_gamma ** protected_class_count[a])
              for a in [0, 1]
            }
          else:
            correction_factor = {
              (a, y): (1 - base_gamma)/(1 - base_gamma ** subgroup_count[(a, y)])
              for a in [0, 1] for y in [0, 1]
            }
        else:
          if fairness_type == 'dp':
            correction_factor = {a: 1.0 for a in [0, 1]}
          else:
            correction_factor = {(a, y): 1.0 for a in [0, 1] for y in [0, 1]}

        total_gradients = []
        idx_b = 0
        idx_w = 0

        for idx, grad in enumerate(gradients):
          tree_id = idx // 3
          if fairness_type == 'dp':
            agg_y_a1 = agg_y[(1, 0)] + agg_y[(1, 1)]
            agg_y_a0 = agg_y[(0, 0)] + agg_y[(0, 1)]

            F = agg_y_a1[tree_id] / protected_class_count[1] - agg_y_a0[tree_id] / protected_class_count[0]

            F = tf.convert_to_tensor(F, dtype=tf.float32)

            if constraint_type == 'node':
              signs_y = tf.math.sign(F - huber_loss_delta/2)
            else:
              signs_y = dp_sign

            gradient_b_a0 = gradient_b[(0, 0)] + gradient_b[(0, 1)]
            gradient_b_a1 = gradient_b[(1, 0)] + gradient_b[(1, 1)]
            gradient_w_a0 = gradient_w[(0, 0)] + gradient_w[(0, 1)]
            gradient_w_a1 = gradient_w[(1, 0)] + gradient_w[(1, 1)]
            correction_factor_0 = correction_factor[0]
            correction_factor_1 = correction_factor[1]

            # shape 1 is for bias gradients, shape 2 is for weight gradients
            if len(grad.shape) == 1 and grad.shape[0] == num_internal_nodes:

              grad_b_diff = tf.convert_to_tensor(
                  gradient_b_a1[idx_b] * correction_factor_1
              ) - tf.convert_to_tensor(
                  gradient_b_a0[idx_b] * correction_factor_0)
              grad_b_diff = tf.cast(grad_b_diff, tf.float32)


              huber_check = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
              huber_quadratic = tf.multiply(huber_check * F, grad_b_diff)
              huber_abs = tf.multiply(tf.multiply(1-huber_check, signs_y), grad_b_diff)

              grad_b = lambda_const * (huber_quadratic + huber_abs)
              average_b_fair_grad.append(grad_b)

              total_gradients.append(grad + grad_b)
              idx_b = idx_b + 1
            elif len(grad.shape) == 2 and grad.shape[0] == data_dim:
              grad_w_diff = tf.convert_to_tensor(
                  gradient_w_a1[idx_w] * correction_factor_1
              ) - tf.convert_to_tensor(
                  gradient_w_a0[idx_w] * correction_factor_0)
              grad_w_diff = tf.cast(grad_w_diff, tf.float32)

              huber_check = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
              huber_quadratic = tf.multiply(tf.multiply(huber_check, F), grad_w_diff)
              huber_abs = tf.multiply(tf.multiply(1-huber_check, signs_y), grad_w_diff)

              grad_w = lambda_const * (huber_quadratic + huber_abs)
              average_w_fair_grad.append(grad_w)
              total_gradients.append(grad + grad_w)
              idx_w = idx_w + 1
            else:
              total_gradients.append(grad)

          elif fairness_type == 'eo':
            # EO: penalize average of TPR gap (y=1) and FPR gap (y=0)
            F_y0 = (agg_y[(1, 0)][tree_id] / subgroup_count[(1, 0)]
                    - agg_y[(0, 0)][tree_id] / subgroup_count[(0, 0)])

            F_y1 = (agg_y[(1, 1)][tree_id] / subgroup_count[(1, 1)]
                    - agg_y[(0, 1)][tree_id] / subgroup_count[(0, 1)])

            # Compute gradient contributions from both y=0 and y=1 subgroups
            if len(grad.shape) == 1 and grad.shape[0] == num_internal_nodes:
              fair_penalty = tf.zeros_like(grad)

              # Average both violations with 0.5 weight each
              for y_cond, F_yc_np, weight in [(0, F_y0, 0.5), (1, F_y1, 0.5)]:
                F_yc = tf.convert_to_tensor(F_yc_np, dtype=tf.float32)
                signs_yc = tf.math.sign(F_yc)

                grad_b_diff = tf.cast(
                    tf.convert_to_tensor(gradient_b[(1, y_cond)][idx_b] * correction_factor[(1, y_cond)])
                    - tf.convert_to_tensor(gradient_b[(0, y_cond)][idx_b] * correction_factor[(0, y_cond)]),
                    tf.float32)

                huber_check = tf.cast(tf.math.abs(F_yc) < huber_loss_delta, tf.float32)
                huber_quadratic = tf.multiply(huber_check * F_yc, grad_b_diff)
                huber_abs = tf.multiply(tf.multiply(1 - huber_check, signs_yc), grad_b_diff)
                fair_penalty += weight * (huber_quadratic + huber_abs)

              average_b_fair_grad.append(fair_penalty)
              total_gradients.append(grad + lambda_const * fair_penalty)
              idx_b = idx_b + 1

            elif len(grad.shape) == 2 and grad.shape[0] == data_dim:
              fair_penalty = tf.zeros_like(grad)

              # Average both violations with 0.5 weight each
              for y_cond, F_yc_np, weight in [(0, F_y0, 0.5), (1, F_y1, 0.5)]:
                F_yc = tf.convert_to_tensor(F_yc_np, dtype=tf.float32)
                signs_yc = tf.math.sign(F_yc)


                grad_w_diff = tf.cast(
                    tf.convert_to_tensor(gradient_w[(1, y_cond)][idx_w] * correction_factor[(1, y_cond)])
                    - tf.convert_to_tensor(gradient_w[(0, y_cond)][idx_w] * correction_factor[(0, y_cond)]),
                    tf.float32)

                huber_check = tf.cast(tf.math.abs(F_yc) < huber_loss_delta, tf.float32)
                huber_quadratic = tf.multiply(tf.multiply(huber_check, F_yc), grad_w_diff)
                huber_abs = tf.multiply(tf.multiply(1 - huber_check, signs_yc), grad_w_diff)
                fair_penalty += weight * (huber_quadratic + huber_abs)

              average_w_fair_grad.append(fair_penalty)
              total_gradients.append(grad + lambda_const * fair_penalty)
              idx_w = idx_w + 1
            else:
              total_gradients.append(grad)
        total_gradients = tuple(total_gradients)
      optimizer.apply_gradients(zip(total_gradients, model.trainable_variables))

  y_true_array = np.array(y_true_all)
  y_pred_array = np.array(y_predictions)

  cm = utils.display_confusion_matrix(
      y_true_array,
      y_pred_array,
      save_path='files/train_confusion_matrix.png',
      title='Training Set Final Confusion Matrix',
      log_to_wandb=(not local_run),
      wandb_module=wandb if not local_run else None
  )

  if not local_run:
    wandb.log({
      "final_accuracy": avg_accuracy.result().numpy(),
      "final_auc": avg_auc.result().numpy(),
      "final_dp": dp,
      "final_eo": eo,
    })

  return demographic_parities, equalized_odds, accuracies, average_w_fair_grad, average_b_fair_grad
