import collections
import numpy as np
import tensorflow as tf


def init_fairness_state(num_trees, data_dim, num_internal_nodes):
    gradient_w = {(a, y): np.zeros((num_trees, data_dim, num_internal_nodes)) 
                  for a in [0, 1] for y in [0, 1]}
    gradient_b = {(a, y): np.zeros((num_trees, num_internal_nodes)) 
                  for a in [0, 1] for y in [0, 1]}
    agg_y = {(a, y): np.zeros((num_trees, num_internal_nodes)) 
             for a in [0, 1] for y in [0, 1]}
    subgroup_count = collections.defaultdict(int)
    protected_class_count = collections.defaultdict(int)
    return gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count

def accumulate_fairness_stats(
        tape, protected_batch, targets_batch, 
        node_decisions, predictions, 
        all_tree_trainable_vars, model_trainable_vars,
        gradient_w, gradient_b, agg_y,
        subgroup_count, protected_class_count,
        num_internal_nodes, data_dim,
        constraint_type='node', gradient_type='vanilla', base_gamma=0.9
):
    for i, a_label in enumerate(protected_batch):
        a_label = int(a_label)
        y_label = int(targets_batch[i])
        protected_class_count[a_label] += 1
        subgroup_count[(a_label, y_label)] += 1
        agg_y[(a_label, y_label)] += node_decisions[:, i].numpy()
        
        if constraint_type == 'node':
            fair_gradients = tape.gradient(node_decisions[:, i], all_tree_trainable_vars)
        elif constraint_type == 'leaf':
            fair_gradients = tape.gradient(predictions[i], model_trainable_vars)

        idx_b = idx_w = -1
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
            
            if gradient_type in ('momentum', 'ema'):
                gradient_theta[idx_theta] = (
                    gradient_theta[idx_theta] * base_gamma + fair_grad.numpy()
                )
            else: 
                factor = 1 / subgroup_count[(a_label, y_label)]
                gradient_theta[idx_theta] = (
                    gradient_theta[idx_theta] * (1 - factor) + fair_grad.numpy() * factor
                )

def compute_fairness_gradients(
    gradients,
    gradient_w, gradient_b, agg_y,
    subgroup_count, protected_class_count,
    fairness_type, lambda_const,
    num_internal_nodes, data_dim, num_trees,
    gradient_type='vanilla', base_gamma=0.9,
    huber_loss_delta=0.1, dp_sign=1.0, constraint_type='node',
):
    if fairness_type == 'dp':
        can_compute = protected_class_count[0] > 0 and protected_class_count[1] > 0
    else:
        can_compute = all(subgroup_count[(a, y)] > 0 for a in [0, 1] for y in [0, 1])

    if not can_compute:
        return gradients

    if gradient_type == 'ema':
        if fairness_type == 'dp':
            correction_factor = {a: (1 - base_gamma) / (1 - base_gamma ** protected_class_count[a])
                                 for a in [0, 1]}
        else:
            correction_factor = {(a, y): (1 - base_gamma) / (1 - base_gamma ** subgroup_count[(a, y)])
                                 for a in [0, 1] for y in [0, 1]}
    else:
        correction_factor = ({a: 1.0 for a in [0, 1]} if fairness_type == 'dp'
                             else {(a, y): 1.0 for a in [0, 1] for y in [0, 1]})

    total_gradients = []
    idx_b = idx_w = 0

    for idx, grad in enumerate(gradients):
        tree_id = idx // 3

        if fairness_type == 'dp':
            agg_a1 = agg_y[(1, 0)] + agg_y[(1, 1)]
            agg_a0 = agg_y[(0, 0)] + agg_y[(0, 1)]
            F = tf.convert_to_tensor(
                agg_a1[tree_id] / protected_class_count[1]
                - agg_a0[tree_id] / protected_class_count[0], dtype=tf.float32)
            signs_y = tf.math.sign(F - huber_loss_delta / 2) if constraint_type == 'node' else dp_sign
            cf0, cf1 = correction_factor[0], correction_factor[1]
            gb_a0 = gradient_b[(0, 0)] + gradient_b[(0, 1)]
            gb_a1 = gradient_b[(1, 0)] + gradient_b[(1, 1)]
            gw_a0 = gradient_w[(0, 0)] + gradient_w[(0, 1)]
            gw_a1 = gradient_w[(1, 0)] + gradient_w[(1, 1)]

            if len(grad.shape) == 1 and grad.shape[0] == num_internal_nodes:
                diff = tf.cast(tf.convert_to_tensor(gb_a1[idx_b] * cf1)
                               - tf.convert_to_tensor(gb_a0[idx_b] * cf0), tf.float32)
                hc = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
                penalty = lambda_const * (tf.multiply(hc * F, diff) + tf.multiply(tf.multiply(1 - hc, signs_y), diff))
                total_gradients.append(grad + penalty)
                idx_b += 1
            elif len(grad.shape) == 2 and grad.shape[0] == data_dim:
                diff = tf.cast(tf.convert_to_tensor(gw_a1[idx_w] * cf1)
                               - tf.convert_to_tensor(gw_a0[idx_w] * cf0), tf.float32)
                hc = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
                penalty = lambda_const * (tf.multiply(tf.multiply(hc, F), diff) + tf.multiply(tf.multiply(1 - hc, signs_y), diff))
                total_gradients.append(grad + penalty)
                idx_w += 1
            else:
                total_gradients.append(grad)

        elif fairness_type == 'eo':
            F_y0 = (agg_y[(1, 0)][tree_id] / subgroup_count[(1, 0)]
                    - agg_y[(0, 0)][tree_id] / subgroup_count[(0, 0)])
            F_y1 = (agg_y[(1, 1)][tree_id] / subgroup_count[(1, 1)]
                    - agg_y[(0, 1)][tree_id] / subgroup_count[(0, 1)])

            if len(grad.shape) == 1 and grad.shape[0] == num_internal_nodes:
                fair_penalty = tf.zeros_like(grad)
                for y_cond, F_yc_np in [(0, F_y0), (1, F_y1)]:
                    F_yc = tf.convert_to_tensor(F_yc_np, dtype=tf.float32)
                    diff = tf.cast(
                        tf.convert_to_tensor(gradient_b[(1, y_cond)][idx_b] * correction_factor[(1, y_cond)])
                        - tf.convert_to_tensor(gradient_b[(0, y_cond)][idx_b] * correction_factor[(0, y_cond)]),
                        tf.float32)
                    hc = tf.cast(tf.math.abs(F_yc) < huber_loss_delta, tf.float32)
                    fair_penalty += 0.5 * (tf.multiply(hc * F_yc, diff) + tf.multiply(tf.multiply(1 - hc, tf.math.sign(F_yc)), diff))
                total_gradients.append(grad + lambda_const * fair_penalty)
                idx_b += 1
            elif len(grad.shape) == 2 and grad.shape[0] == data_dim:
                fair_penalty = tf.zeros_like(grad)
                for y_cond, F_yc_np in [(0, F_y0), (1, F_y1)]:
                    F_yc = tf.convert_to_tensor(F_yc_np, dtype=tf.float32)
                    diff = tf.cast(
                        tf.convert_to_tensor(gradient_w[(1, y_cond)][idx_w] * correction_factor[(1, y_cond)])
                        - tf.convert_to_tensor(gradient_w[(0, y_cond)][idx_w] * correction_factor[(0, y_cond)]),
                        tf.float32)
                    hc = tf.cast(tf.math.abs(F_yc) < huber_loss_delta, tf.float32)
                    fair_penalty += 0.5 * (tf.multiply(tf.multiply(hc, F_yc), diff) + tf.multiply(tf.multiply(1 - hc, tf.math.sign(F_yc)), diff))
                total_gradients.append(grad + lambda_const * fair_penalty)
                idx_w += 1
            else:
                total_gradients.append(grad)

    return tuple(total_gradients)