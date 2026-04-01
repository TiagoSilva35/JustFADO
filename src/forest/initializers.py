import collections
import numpy as np
import tensorflow as tf


def init_fairness_state(num_trees, data_dim, num_internal_nodes, number_of_atributes):
    # number of atributtes is the numner of unique values in the protected attribute
    group_ids = range(0, number_of_atributes)
    gradient_w = {(a, y): np.zeros((num_trees, data_dim, num_internal_nodes)) 
                  for a in group_ids for y in [0, 1]}
    gradient_b = {(a, y): np.zeros((num_trees, num_internal_nodes)) 
                  for a in group_ids for y in [0, 1]}
    agg_y = {(a, y): np.zeros((num_trees, num_internal_nodes)) 
             for a in group_ids for y in [0, 1]}
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

        # none_count = sum(1 for g in fair_gradients if g is None)
        # nonzero_norms = [float(tf.norm(g).numpy()) for g in fair_gradients if g is not None]
        # if len(nonzero_norms) == 0 or all(n == 0.0 for n in nonzero_norms[:3]):
        #     print(f"[ACCUM DEBUG] a={a_label} y={y_label} none={none_count}/{len(fair_gradients)} norms={nonzero_norms[:6]}")

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
    num_internal_nodes, data_dim, number_of_atributes,
    gradient_type='vanilla', base_gamma=0.9,
    huber_loss_delta=0.1, dp_sign=1.0, constraint_type='node',
):
    group_ids = range(1, number_of_atributes + 1)
    if fairness_type == 'dp':
        can_compute = all(protected_class_count[a] > 0 for a in group_ids)
    else:
        can_compute = all(subgroup_count[(a, y)] > 0 for a in group_ids for y in [0, 1])

    if not can_compute:
        return gradients

    if gradient_type == 'ema':
        if fairness_type == 'dp':
            correction_factor = {a: (1 - base_gamma) / (1 - base_gamma ** protected_class_count[a])
                                 for a in group_ids}
        else:
            correction_factor = {(a, y): (1 - base_gamma) / (1 - base_gamma ** subgroup_count[(a, y)])
                                 for a in group_ids for y in [0, 1]}
    else:
        correction_factor = ({a: 1.0 for a in group_ids} if fairness_type == 'dp'
                             else {(a, y): 1.0 for a in group_ids for y in [0, 1]})

    total_gradients = []
    idx_b = idx_w = 0

    for grad in gradients:
        if len(grad.shape) == 2 and grad.shape[0] == data_dim:
            tree_id = idx_w
        elif len(grad.shape) == 1 and grad.shape[0] == num_internal_nodes:
            tree_id = idx_b
        else:
            # theta or other variable — assign to the most recent tree
            tree_id = max(idx_w, idx_b) - 1 if max(idx_w, idx_b) > 0 else 0

        if fairness_type == 'dp':
            agg_a = {
                a: agg_y[(a, 0)] + agg_y[(a, 1)]
                for a in group_ids
            }
            F_k = {
                a: tf.convert_to_tensor(
                    (
                        sum(
                            agg_a[g][tree_id] / protected_class_count[g]
                            for g in group_ids
                        ) / number_of_atributes
                    ) - (agg_a[a][tree_id] / protected_class_count[a]),
                    dtype=tf.float32
                )
                for a in group_ids
            }
            if len(grad.shape) == 1 and grad.shape[0] == num_internal_nodes:
                fair_penalty = tf.zeros_like(grad)
                for k in group_ids:
                    cf_a = correction_factor[k]
                    gb_a = gradient_b[(k, 0)] + gradient_b[(k, 1)]
                    mean_other = tf.convert_to_tensor(
                        sum(
                            (gradient_b[(g, 0)] + gradient_b[(g, 1)])[idx_b] * correction_factor[g]
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    diff = mean_other - tf.cast(tf.convert_to_tensor(gb_a[idx_b] * cf_a), tf.float32)
                    F = F_k[k]
                    signs_y = tf.math.sign(F - huber_loss_delta / 2) if constraint_type == 'node' else dp_sign
                    hc = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
                    fair_penalty += tf.multiply(hc * F, diff) + tf.multiply(tf.multiply(1 - hc, signs_y), diff)
                fair_penalty = fair_penalty / float(number_of_atributes)
                total_gradients.append(grad + lambda_const * fair_penalty)
                idx_b += 1
            elif len(grad.shape) == 2 and grad.shape[0] == data_dim:
                fair_penalty = tf.zeros_like(grad)
                for k in group_ids:
                    cf_a = correction_factor[k]
                    gw_a = gradient_w[(k, 0)] + gradient_w[(k, 1)]
                    mean_other = tf.convert_to_tensor(
                        sum(
                            (gradient_w[(g, 0)] + gradient_w[(g, 1)])[idx_w] * correction_factor[g]
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    diff = mean_other - tf.cast(tf.convert_to_tensor(gw_a[idx_w] * cf_a), tf.float32)
                    F = F_k[k]
                    signs_y = tf.math.sign(F - huber_loss_delta / 2) if constraint_type == 'node' else dp_sign
                    hc = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
                    fair_penalty += tf.multiply(tf.multiply(hc, F), diff) + tf.multiply(tf.multiply(1 - hc, signs_y), diff)
                fair_penalty = fair_penalty / float(number_of_atributes)
                total_gradients.append(grad + lambda_const * fair_penalty)
                idx_w += 1
            else:
                total_gradients.append(grad)

        elif fairness_type == 'eo':
            F_y = {
                y_cond: {
                    a: tf.convert_to_tensor(
                        (
                            sum(
                                agg_y[(g, y_cond)][tree_id] / subgroup_count[(g, y_cond)]
                                for g in group_ids
                            ) / number_of_atributes
                        ) - (agg_y[(a, y_cond)][tree_id] / subgroup_count[(a, y_cond)]),
                        dtype=tf.float32
                    )
                    for a in group_ids
                }
                for y_cond in [0, 1]
            }

            if len(grad.shape) == 1 and grad.shape[0] == num_internal_nodes:
                fair_penalty = tf.zeros_like(grad)
                for y_cond in [0, 1]:
                    mean_grad = tf.convert_to_tensor(
                        sum(
                            gradient_b[(g, y_cond)][idx_b] * correction_factor[(g, y_cond)]
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    for a in group_ids:
                        F_yc = F_y[y_cond][a]
                        diff = mean_grad - tf.cast(
                            tf.convert_to_tensor(
                                gradient_b[(a, y_cond)][idx_b] * correction_factor[(a, y_cond)]
                            ),
                            tf.float32,
                        )
                        hc = tf.cast(tf.math.abs(F_yc) < huber_loss_delta, tf.float32)
                        fair_penalty += tf.multiply(hc * F_yc, diff) + tf.multiply(
                            tf.multiply(1 - hc, tf.math.sign(F_yc)), diff
                        )
                fair_penalty = fair_penalty / float(2 * number_of_atributes)
                total_gradients.append(grad + lambda_const * fair_penalty)
                idx_b += 1
            elif len(grad.shape) == 2 and grad.shape[0] == data_dim:
                fair_penalty = tf.zeros_like(grad)
                for y_cond in [0, 1]:
                    mean_grad = tf.convert_to_tensor(
                        sum(
                            gradient_w[(g, y_cond)][idx_w] * correction_factor[(g, y_cond)]
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    for a in group_ids:
                        F_yc = F_y[y_cond][a]
                        diff = mean_grad - tf.cast(
                            tf.convert_to_tensor(
                                gradient_w[(a, y_cond)][idx_w] * correction_factor[(a, y_cond)]
                            ),
                            tf.float32,
                        )
                        hc = tf.cast(tf.math.abs(F_yc) < huber_loss_delta, tf.float32)
                        fair_penalty += tf.multiply(tf.multiply(hc, F_yc), diff) + tf.multiply(
                            tf.multiply(1 - hc, tf.math.sign(F_yc)), diff
                        )
                fair_penalty = fair_penalty / float(2 * number_of_atributes)
                total_gradients.append(grad + lambda_const * fair_penalty)
                idx_w += 1
            else:
                total_gradients.append(grad)

    return tuple(total_gradients)
