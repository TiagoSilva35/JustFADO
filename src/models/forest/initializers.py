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
        node_decisions_per_sample, predictions_per_sample,
        all_tree_trainable_vars, model_trainable_vars,
        gradient_w, gradient_b, agg_y,
        subgroup_count, protected_class_count,
        num_internal_nodes, data_dim,
        constraint_type='node', gradient_type='vanilla', base_gamma=0.9
):
    """Accumulate per-(group, label) running fair-gradient stats.

    IMPORTANT (B2 fix, 2026-06): ``node_decisions_per_sample`` and
    ``predictions_per_sample`` MUST be Python lists of per-sample tensors
    that were produced INSIDE the ``with tf.GradientTape(...) as tape:``
    block (via ``tf.unstack(node_decisions_batch, axis=1)`` and
    ``tf.unstack(predictions_batch, axis=0)`` respectively). Slicing the
    batch tensors outside the tape's ``with``-block creates new tensors
    the tape has not recorded, so ``tape.gradient(sliced_tensor, vars)``
    returns ``None`` for every var -- which silently disables the entire
    fairness regulariser. See ``DOCS/BUG_REPORT_fairness_regulariser.md``
    for the full diagnosis.
    """
    for i, a_label in enumerate(protected_batch):
        a_label = int(a_label)
        y_label = int(targets_batch[i])
        protected_class_count[a_label] += 1
        subgroup_count[(a_label, y_label)] += 1

        # Per-sample node decisions are pre-sliced (list indexing, not
        # tensor slicing) so this update sees the tape-recorded tensors.
        agg_y[(a_label, y_label)] += node_decisions_per_sample[i].numpy()

        if constraint_type == 'node':
            fair_gradients = tape.gradient(
                node_decisions_per_sample[i], all_tree_trainable_vars
            )
        elif constraint_type == 'leaf':
            fair_gradients = tape.gradient(
                predictions_per_sample[i], model_trainable_vars
            )

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
    # NOTE: must match the indexing convention used by ``init_fairness_state``
    # (line 8) and ``accumulate_fairness_stats`` (lines 28-34), both of which
    # key state dicts by ``int(a_label)`` -- i.e. the raw 0/1 protected-attribute
    # value, not a 1-indexed group id. The previous ``range(1, n+1)`` here
    # caused ``can_compute`` to fail on binary protected attributes
    # (protected_class_count[2] was never populated because no sample has
    # a_label == 2), short-circuiting this function to ``return gradients``
    # unmodified and making ``lambda_const`` dead code during the prequential
    # train phase. Confirmed via bit-identical FADO/Aranyani-Base results at
    # lambda=0.1 vs lambda=10.0 on COMPAS abrupt_race (seed 42, 2025-11).
    group_ids = range(0, number_of_atributes)
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
                    cf_a = correction_factor[k] # type: ignore
                    gb_a = gradient_b[(k, 0)] + gradient_b[(k, 1)]
                    mean_other = tf.convert_to_tensor(
                        sum(
                            (gradient_b[(g, 0)] + gradient_b[(g, 1)])[idx_b] * correction_factor[g] # type: ignore
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
                    cf_a = correction_factor[k] # type: ignore
                    gw_a = gradient_w[(k, 0)] + gradient_w[(k, 1)]
                    mean_other = tf.convert_to_tensor(
                        sum(
                            (gradient_w[(g, 0)] + gradient_w[(g, 1)])[idx_w] * correction_factor[g] # type: ignore
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
                            gradient_b[(g, y_cond)][idx_b] * correction_factor[(g, y_cond)] # type: ignore
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    for a in group_ids:
                        F_yc = F_y[y_cond][a]
                        diff = mean_grad - tf.cast(
                            tf.convert_to_tensor(
                                gradient_b[(a, y_cond)][idx_b] * correction_factor[(a, y_cond)] #type: ignore
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
                            gradient_w[(g, y_cond)][idx_w] * correction_factor[(g, y_cond)] # type: ignore
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    for a in group_ids:
                        F_yc = F_y[y_cond][a]
                        diff = mean_grad - tf.cast(
                            tf.convert_to_tensor(
                                gradient_w[(a, y_cond)][idx_w] * correction_factor[(a, y_cond)] # type: ignore
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
