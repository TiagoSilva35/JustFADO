# import collections
# import copy
# import numpy as np
# import tensorflow as tf
# import tqdm
# import wandb
# import utils


# def train_online(
#     model,
#     inputs,
#     targets,
#     protected_targets,
#     data_dim=13,
#     batch_size=1,
#     tree_depth=3,
#     compute_fairness=True,
#     lambda_const=0.3,
#     num_trees=3,
#     base_gamma=0.9,
#     constraint_type='node',
#     gradient_type='vanilla',
#     loss_type='ce',
#     local_run=False,
# ):
#   """Online training function with fairness constraints for MLP.
  
#   This function trains an MLP with demographic parity fairness constraints by:
#   1. Computing task loss gradients: ∇_θ L_CE(y, ŷ)
#   2. Computing fairness gradients separately for each protected group (a=0, a=1)
#   3. Combining them as: ∇_total = ∇_task + λ * sign(DP) * (∇_a1 - ∇_a0)
  
#   The model has 4 core trainable parameters (W0, B0, W1, B1). If layer normalization
#   is enabled, additional parameters (gamma, beta) are added. The code matches
#   gradients to the correct parameters by variable name and shape.
  
#   Core parameters:
#   - W0: weight of hidden layer [data_dim, num_internal_nodes]  
#   - B0: bias of hidden layer [num_internal_nodes]
#   - W1: dense.kernel of output layer [num_internal_nodes, num_classes]
#   - B1: dense.bias of output layer [num_classes]
  
#   Args:
#     model: FairReLUNetwork model.
#     inputs: training input features.
#     targets: training labels.
#     protected_targets: protected attribute labels (binary: 0 or 1).
#     data_dim: dimension of input features.
#     batch_size: number of samples per batch.
#     tree_depth: depth parameter (controls hidden layer size).
#     compute_fairness: whether to apply fairness constraints.
#     lambda_const: weight for fairness penalty term.
#     num_trees: unused (for API compatibility).
#     base_gamma: unused (for API compatibility).
#     constraint_type: unused (for API compatibility).
#     gradient_type: unused (for API compatibility).
#     loss_type: loss function type.
#     local_run: whether running locally (affects wandb logging).
  
#   Returns:
#     demographic_parities: list of DP values over training.
#     accuracies: list of accuracy values over training.
#   """
  
#   dataset = tf.data.Dataset.from_tensor_slices(
#       (inputs, targets, protected_targets)
#   )
#   print(set(targets), set(protected_targets))
#   dataset = dataset.batch(batch_size)
#   num_internal_nodes = 2**tree_depth - 1
#   optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
#   criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#   avg_accuracy = tf.keras.metrics.Accuracy()

#   gradient_b0_a0 = np.zeros([num_internal_nodes])
#   gradient_b0_a1 = np.zeros([num_internal_nodes])

#   gradient_w0_a0 = np.zeros([data_dim, num_internal_nodes])
#   gradient_w0_a1 = np.zeros([data_dim, num_internal_nodes])

#   gradient_w1_a0 = np.zeros([num_internal_nodes, 2])
#   gradient_w1_a1 = np.zeros([num_internal_nodes, 2])
    
#   gradient_b1_a0 = np.zeros([2])
#   gradient_b1_a1 = np.zeros([2])

#   agg_y_a0 = 0
#   agg_y_a1 = 0
    
#   grad_norm = []

#   # avoid runtime error
#   protected_class_count = collections.defaultdict(int)
#   dp_function = utils.get_demographic_parity
#   avg_loss = tf.keras.metrics.Mean()
#   avg_auc = tf.keras.metrics.AUC()
#   y_predictions = []
#   demographic_parities = []
#   accuracies = []

#   iterations = tqdm.tqdm(dataset)
#   for _, (
#       inputs_batch,
#       targets_batch,
#       protected_batch) in enumerate(iterations):
#     with tf.GradientTape(persistent=compute_fairness) as tape:
#       # make a forward pass and get the predictions
#       # predictions: [batch_size, num_class]
#       # y: [num_trees, batch_size, num_internal_nodes]
#       predictions = model(inputs_batch, training=True)
#       # y_pred: [batch_size]
#       y_pred = tf.math.argmax(predictions, axis=-1)
#       target_loss = criteria(y_true=targets_batch, y_pred=predictions)
      
#       # get demographic parity scores
#       y_predictions.extend(y_pred.numpy())
      
#       # calculate dp which is the difference in positive outcome rates between groups
#       dp, dp_sign = dp_function(
#           y_predictions, protected_targets[: len(y_predictions)])
      
#       # track demographic parity over time
#       demographic_parities.append(dp)
#       accuracies.append(avg_accuracy.result().numpy())
      
#       # update running averages for accuracy, auc, loss
#       avg_accuracy.update_state(targets_batch, y_pred)
#       avg_auc.update_state(targets_batch, y_pred)
#       avg_loss.update_state(target_loss)
#       iterations.set_description(
#           f' CE Loss: {avg_loss.result():.3f},'
#           f' Accuracy: { avg_accuracy.result():.3%},'
#           f' AUC: {avg_auc.result():.3%},'
#           f' DP: {dp:.5f},'
#       )
#       results = {
#           'CE loss': avg_loss.result(),
#           'Accuracy': avg_accuracy.result(),
#           'AUC': avg_auc.result(),
#           'DP': dp,
#       }

#       if not local_run:
#         wandb.log(results)

      
#       # Calculate the gradients separately for protected groups and computes penalties
       
#       # update the task gradients w.r.t. current sample
#       # d L_{CE}(y, \hat{y})/d\theta
#       gradients = tape.gradient(target_loss, model.trainable_variables)
#       total_gradients = gradients
#       # fairness gradient computation
#       # sign(E_{x~1}[n_i] - E_{x~0}[n_i]) *
#       # (E_{x~1}[dn_i/d\theta] - E_{x~0}[dn_i/d\theta])
#       # where n_i is the decision of the i-th internal node
#       if compute_fairness:
#         # iterate over the current batch and aggregate the gradients
#         for i, a_label in enumerate(protected_batch.numpy()):
#           protected_class_count[a_label] += 1
#           # update the aggregate score based on protected label: A.
#           if a_label == 0:
#             agg_y_a0 += y_pred
#           elif a_label == 1:
#             agg_y_a1 += y_pred
            
          
#           # Take gradient w.r.t. P(y=1) which is what DP measures (positive outcome)
#           fair_gradients = tape.gradient(predictions[i, 1], model.trainable_variables) 
          
#           factor = 1 / protected_class_count[a_label]
        
#           # vanilla averaging: g_t = g_{t-1}*(1-1/t) + dL
#           # Match gradients to stored arrays by shape and variable name
#           for fair_grad, var in zip(fair_gradients, model.trainable_variables):
#             if fair_grad is None:
#               continue
              
#             var_name = var.name
            
#             # Match by variable name and shape
#             if 'W:0' in var_name and fair_grad.shape == gradient_w0_a0.shape:
#               # Hidden layer weight (W0)
#               if a_label == 0:
#                 gradient_w0_a0 = gradient_w0_a0 * (1 - factor) + fair_grad.numpy() * factor
#               elif a_label == 1:
#                 gradient_w0_a1 = gradient_w0_a1 * (1 - factor) + fair_grad.numpy() * factor
            
#             elif 'B:0' in var_name and fair_grad.shape == gradient_b0_a0.shape:
#               # Hidden layer bias (B0)
#               if a_label == 0:
#                 gradient_b0_a0 = gradient_b0_a0 * (1 - factor) + fair_grad.numpy() * factor
#               elif a_label == 1:
#                 gradient_b0_a1 = gradient_b0_a1 * (1 - factor) + fair_grad.numpy() * factor
            
#             elif '/kernel:0' in var_name and fair_grad.shape == gradient_w1_a0.shape:
#               # Dense layer kernel (W1)
#               if a_label == 0:
#                 gradient_w1_a0 = gradient_w1_a0 * (1 - factor) + fair_grad.numpy() * factor
#               elif a_label == 1:
#                 gradient_w1_a1 = gradient_w1_a1 * (1 - factor) + fair_grad.numpy() * factor
            
#             elif '/bias:0' in var_name and fair_grad.shape == gradient_b1_a0.shape:
#               # Dense layer bias (B1)
#               if a_label == 0:
#                 gradient_b1_a0 = gradient_b1_a0 * (1 - factor) + fair_grad.numpy() * factor
#               elif a_label == 1:
#                 gradient_b1_a1 = gradient_b1_a1 * (1 - factor) + fair_grad.numpy() * factor
            
#             # Skip LayerNorm and other optional layer parameters

            
#         # safety check: need both groups AND enough samples for stable DP
#         min_samples_per_group = 10  # warmup period
#         if (protected_class_count[0] < min_samples_per_group or 
#             protected_class_count[1] < min_samples_per_group):
#           # Not enough samples yet, skip fairness correction
#           optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#           continue
        
#         correction_factor_0, correction_factor_1 = 1.0, 1.0
#         total_gradients = []
#         fairness_grad_norms = []
#         task_grad_norms = []
        
#         # iterate over all task gradients and add the fairness term
#         # Need to identify which gradient corresponds to which parameter
#         # since LayerNorm adds extra trainable variables
#         for idx, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
#           grad_f = None
          
#           # Identify the parameter by name to handle optional layers
#           var_name = var.name
          
#           if 'W:0' in var_name or '/kernel:0' in var_name:
#             # Weight matrix (could be W0 or dense.kernel)
#             if grad.shape == gradient_w0_a0.shape:
#               # Hidden layer weight (W0): [data_dim, num_internal_nodes]
#               grad_diff = tf.convert_to_tensor(gradient_w0_a1) - tf.convert_to_tensor(gradient_w0_a0)
#               grad_f = lambda_const * tf.multiply(dp_sign, tf.cast(grad_diff, tf.float32))
#               grad_norm.append(tf.norm(grad_f))
#             elif grad.shape == gradient_w1_a0.shape:
#               # Dense layer kernel (W1): [num_internal_nodes, num_classes]
#               grad_diff = tf.convert_to_tensor(gradient_w1_a1) - tf.convert_to_tensor(gradient_w1_a0)
#               grad_f = lambda_const * tf.multiply(dp_sign, tf.cast(grad_diff, tf.float32))
          
#           elif 'B:0' in var_name or '/bias:0' in var_name:
#             # Bias vector (could be B0 or dense.bias)
#             if grad.shape == gradient_b0_a0.shape:
#               # Hidden layer bias (B0): [num_internal_nodes]
#               grad_diff = tf.convert_to_tensor(gradient_b0_a1) - tf.convert_to_tensor(gradient_b0_a0)
#               grad_f = lambda_const * tf.multiply(dp_sign, tf.cast(grad_diff, tf.float32))
#             elif grad.shape == gradient_b1_a0.shape:
#               # Dense layer bias (B1): [num_classes]
#               grad_diff = tf.convert_to_tensor(gradient_b1_a1) - tf.convert_to_tensor(gradient_b1_a0)
#               grad_f = lambda_const * tf.multiply(dp_sign, tf.cast(grad_diff, tf.float32))
          
#           # For LayerNorm parameters (gamma, beta) or other layers, don't apply fairness
#           if grad_f is not None:
#             total_gradients.append(grad + grad_f)
#             fairness_grad_norms.append(tf.norm(grad_f).numpy())
#             task_grad_norms.append(tf.norm(grad).numpy())
#           else:
#             total_gradients.append(grad)
        
#         # Log fairness gradient statistics every 100 steps
#         if len(y_predictions) % 100 == 0 and len(fairness_grad_norms) > 0:
#           avg_fairness_norm = np.mean(fairness_grad_norms)
#           avg_task_norm = np.mean(task_grad_norms)
#           ratio = avg_fairness_norm / (avg_task_norm + 1e-8)
#           print(f"Step {len(y_predictions)}: Fairness/Task gradient ratio: {ratio:.4f} "
#                 f"(fairness: {avg_fairness_norm:.6f}, task: {avg_task_norm:.6f})")
        
#         total_gradients = tuple(total_gradients)
#       optimizer.apply_gradients(zip(total_gradients, model.trainable_variables))
    
#     # Clean up persistent tape if used
#     if compute_fairness:
#       del tape
  
#   # Return consistent format with other training functions
#   return demographic_parities, accuracies
"""Improved MLP trainer with differentiable fairness loss."""

import collections
import numpy as np
import tensorflow as tf
import tqdm
import wandb
import utils


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
    loss_type='ce',
    local_run=False,
):
  """Online training with gradient-based fairness for MLP.
  
  Mirrors Aranyani's approach: accumulate per-group gradients and add
  a fairness correction term based on the difference.
  """
  
  dataset = tf.data.Dataset.from_tensor_slices(
      (inputs, targets, protected_targets)
  )
  dataset = dataset.batch(batch_size)
  
  # Fixed learning rate for online learning
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
  criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
  
  # Metrics
  avg_accuracy = tf.keras.metrics.Accuracy()
  avg_loss = tf.keras.metrics.Mean()
  avg_auc = tf.keras.metrics.AUC()
  
  # Get model parameter shapes for gradient accumulation
  # We'll accumulate gradients of P(y=1) w.r.t. each parameter for each group
  gradient_accum_a0 = None  # Will initialize on first pass
  gradient_accum_a1 = None
  
  # Accumulate positive prediction rates per group
  agg_pred_a0 = 0.0
  agg_pred_a1 = 0.0
  
  protected_class_count = collections.defaultdict(int)
  dp_function = utils.get_demographic_parity
  y_predictions = []
  demographic_parities = []
  accuracies = []
  
  # Huber loss threshold (like Aranyani)
  huber_delta = 0.01

  iterations = tqdm.tqdm(dataset)
  for step, (inputs_batch, targets_batch, protected_batch) in enumerate(iterations):
    
    with tf.GradientTape(persistent=compute_fairness) as tape:
      # Forward pass
      predictions = model(inputs_batch, training=True)
      y_pred = tf.math.argmax(predictions, axis=-1)
      
      # Task loss
      task_loss = criteria(y_true=targets_batch, y_pred=predictions)
      
      # Get demographic parity
      y_predictions.extend(y_pred.numpy())
      dp, dp_sign = dp_function(y_predictions, protected_targets[:len(y_predictions)])
      
      # Update metrics
      avg_accuracy.update_state(targets_batch, y_pred)
      avg_auc.update_state(targets_batch, predictions[:, 1])
      avg_loss.update_state(task_loss)
      
      demographic_parities.append(dp)
      accuracies.append(avg_accuracy.result().numpy())
      
      iterations.set_description(
          f' Loss: {avg_loss.result():.3f},'
          f' Acc: {avg_accuracy.result():.3%},'
          f' AUC: {avg_auc.result():.3%},'
          f' DP: {dp:.4f}'
      )
      
      results = {
          'CE loss': avg_loss.result(),
          'Accuracy': avg_accuracy.result(),
          'AUC': avg_auc.result(),
          'DP': dp,
      }
      if not local_run:
        wandb.log(results)
      
      # Task gradients
      gradients = tape.gradient(task_loss, model.trainable_variables)
      total_gradients = list(gradients)
      
      if compute_fairness:
        # Process each sample in batch
        for i, a_label in enumerate(protected_batch.numpy()):
          protected_class_count[a_label] += 1
          
          # Accumulate predictions per group
          pred_y1 = float(predictions[i, 1])
          if a_label == 0:
            agg_pred_a0 += pred_y1
          else:
            agg_pred_a1 += pred_y1
          
          fair_grads = tape.gradient(predictions[i, 1], model.trainable_variables)
          
          # Initialize accumulators on first pass
          if gradient_accum_a0 is None:
            gradient_accum_a0 = [np.zeros_like(g.numpy()) if g is not None else None for g in fair_grads]
            gradient_accum_a1 = [np.zeros_like(g.numpy()) if g is not None else None for g in fair_grads]
          
          # Accumulate with running average (like Aranyani's vanilla mode)
          factor = 1.0 / protected_class_count[a_label]
          for j, fg in enumerate(fair_grads):
            if fg is None or gradient_accum_a0[j] is None:
              continue
            if a_label == 0:
              gradient_accum_a0[j] = gradient_accum_a0[j] * (1 - factor) + fg.numpy() * factor
            else:
              gradient_accum_a1[j] = gradient_accum_a1[j] * (1 - factor) + fg.numpy() * factor
        
        # Safety check: need samples from both groups
        if protected_class_count[0] > 0 and protected_class_count[1] > 0:
          # Compute current disparity in positive rates
          rate_a0 = agg_pred_a0 / protected_class_count[0]
          rate_a1 = agg_pred_a1 / protected_class_count[1]
          F = rate_a1 - rate_a0  # Disparity (signed)
          
          # Apply Huber-like loss (like Aranyani): smooth near 0, linear far from 0
          # This prevents over-correction when DP is small
          for j in range(len(total_gradients)):
            if gradient_accum_a0[j] is None:
              continue
            
            # Gradient difference: E[∂P(y=1)/∂θ | a=1] - E[∂P(y=1)/∂θ | a=0]
            grad_diff = gradient_accum_a1[j] - gradient_accum_a0[j]
            grad_diff = tf.cast(grad_diff, tf.float32)
            
            # Huber-style: quadratic for small F, linear for large F
            if abs(F) < huber_delta:
              # Quadratic region: gentle correction
              fairness_grad = lambda_const * F * grad_diff
            else:
              # Linear region: stronger correction
              fairness_grad = lambda_const * np.sign(F) * grad_diff
            
            total_gradients[j] = total_gradients[j] + fairness_grad
    
    # Apply gradients
    optimizer.apply_gradients(zip(total_gradients, model.trainable_variables))
    
    if compute_fairness:
      del tape
  
  return demographic_parities, accuracies