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
        constraint_type='node', gradient_type='vanilla', base_gamma=0.9,
        inputs=None, model=None,
):
    use_analytic = (
        constraint_type == 'node' and inputs is not None and model is not None
    )
    if use_analytic:
        x_batch = np.asarray(inputs, dtype=np.float32)
        # tau is modulated per-sample by the drift controller, so read it fresh.
        taus = np.array(
            [float(tree.temperature.numpy()) for tree in model.layers],
            dtype=np.float32,
        )

    for i, a_label in enumerate(protected_batch):
        a_label = int(a_label)
        y_label = int(targets_batch[i])
        protected_class_count[a_label] += 1
        subgroup_count[(a_label, y_label)] += 1

        # [num_trees, num_internal_nodes] == sigma(z) for this sample.
        node_decisions = node_decisions_per_sample[i].numpy()
        agg_y[(a_label, y_label)] += node_decisions

        if use_analytic:
            grad_w_ay = gradient_w[(a_label, y_label)]
            grad_b_ay = gradient_b[(a_label, y_label)]
            x_i = x_batch[i]
            factor = 1 / subgroup_count[(a_label, y_label)]
            for tree_id in range(node_decisions.shape[0]):
                n = node_decisions[tree_id]
                sigma_prime = n * (1.0 - n) / taus[tree_id]   # sigma'(z)/tau
                fair_grad_w = np.outer(x_i, sigma_prime)      # dn/dW
                fair_grad_b = sigma_prime                     # dn/db
                if gradient_type in ('momentum', 'ema'):
                    grad_w_ay[tree_id] = grad_w_ay[tree_id] * base_gamma + fair_grad_w
                    grad_b_ay[tree_id] = grad_b_ay[tree_id] * base_gamma + fair_grad_b
                else:
                    grad_w_ay[tree_id] = (
                        grad_w_ay[tree_id] * (1 - factor) + fair_grad_w * factor
                    )
                    grad_b_ay[tree_id] = (
                        grad_b_ay[tree_id] * (1 - factor) + fair_grad_b * factor
                    )
            continue

        if constraint_type == 'node':
            fair_vars = all_tree_trainable_vars
            fair_gradients = tape.gradient(
                node_decisions_per_sample[i], fair_vars
            )
        elif constraint_type == 'leaf':
            fair_vars = model_trainable_vars
            fair_gradients = tape.gradient(
                predictions_per_sample[i], fair_vars
            )
        fair_layout = (
            build_variable_layout(model, fair_vars) if model is not None
            else _infer_variable_layout(fair_gradients, data_dim, num_internal_nodes)
        )

        for position, fair_grad in enumerate(fair_gradients):
            if fair_grad is None:
                continue
            idx_theta, role = fair_layout[position]
            if role == 'bias':
                gradient_theta = gradient_b[(a_label, y_label)]
            elif role == 'weight':
                gradient_theta = gradient_w[(a_label, y_label)]
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

def build_variable_layout(model, variables=None):
    variables = list(model.trainable_variables if variables is None else variables)
    position_of = {id(v): i for i, v in enumerate(variables)}
    layout = [(None, 'other')] * len(variables)
    for tree_id, tree in enumerate(getattr(model, 'layers', [])):
        for role, var in (('weight', getattr(tree, 'weight', None)),
                          ('bias', getattr(tree, 'bias', None)),
                          ('theta', getattr(tree, 'theta', None))):
            if var is None:
                continue
            position = position_of.get(id(var))
            if position is None:
                raise ValueError(
                    f"tree {tree_id} {role} is not in model.trainable_variables"
                )
            if layout[position] != (None, 'other'):
                raise ValueError(f"variable at position {position} claimed twice")
            layout[position] = (tree_id, role)
    return layout


def _infer_variable_layout(gradients, data_dim, num_internal_nodes):
    """Shape-based fallback for callers that cannot supply a layout."""
    layout = []
    idx_w = idx_b = 0
    for grad in gradients:
        if len(grad.shape) == 2 and grad.shape[0] == data_dim:
            layout.append((idx_w, 'weight'))
            idx_w += 1
        elif len(grad.shape) == 1 and grad.shape[0] == num_internal_nodes:
            layout.append((idx_b, 'bias'))
            idx_b += 1
        else:
            layout.append((max(idx_w, idx_b) - 1 if max(idx_w, idx_b) > 0 else 0, 'other'))
    return layout


def compute_fairness_gradients(
    gradients,
    gradient_w, gradient_b, agg_y,
    subgroup_count, protected_class_count,
    fairness_type, lambda_const,
    num_internal_nodes, data_dim, number_of_atributes,
    gradient_type='vanilla', base_gamma=0.9,
    huber_loss_delta=0.1, dp_sign=1.0, constraint_type='node',
    variable_layout=None,
):

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
    layout = variable_layout
    if layout is None:
        layout = _infer_variable_layout(gradients, data_dim, num_internal_nodes)
    if len(layout) != len(gradients):
        raise ValueError(
            f"variable_layout has {len(layout)} entries for {len(gradients)} gradients"
        )

    for position, grad in enumerate(gradients):
        tree_id, role = layout[position]

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
            if role == 'bias':
                fair_penalty = tf.zeros_like(grad)
                for k in group_ids:
                    cf_a = correction_factor[k] # type: ignore
                    gb_a = gradient_b[(k, 0)] + gradient_b[(k, 1)]
                    mean_other = tf.convert_to_tensor(
                        sum(
                            (gradient_b[(g, 0)] + gradient_b[(g, 1)])[tree_id] * correction_factor[g] # type: ignore
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    diff = mean_other - tf.cast(tf.convert_to_tensor(gb_a[tree_id] * cf_a), tf.float32)
                    F = F_k[k]
                    signs_y = tf.math.sign(F - huber_loss_delta / 2) if constraint_type == 'node' else dp_sign
                    hc = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
                    fair_penalty += tf.multiply(hc * F, diff) + tf.multiply(tf.multiply(1 - hc, signs_y), diff)
                fair_penalty = fair_penalty / float(number_of_atributes)
                total_gradients.append(grad + lambda_const * fair_penalty)
            elif role == 'weight':
                fair_penalty = tf.zeros_like(grad)
                for k in group_ids:
                    cf_a = correction_factor[k] # type: ignore
                    gw_a = gradient_w[(k, 0)] + gradient_w[(k, 1)]
                    mean_other = tf.convert_to_tensor(
                        sum(
                            (gradient_w[(g, 0)] + gradient_w[(g, 1)])[tree_id] * correction_factor[g] # type: ignore
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    diff = mean_other - tf.cast(tf.convert_to_tensor(gw_a[tree_id] * cf_a), tf.float32)
                    F = F_k[k]
                    signs_y = tf.math.sign(F - huber_loss_delta / 2) if constraint_type == 'node' else dp_sign
                    hc = tf.cast(tf.math.abs(F) < huber_loss_delta, tf.float32)
                    fair_penalty += tf.multiply(tf.multiply(hc, F), diff) + tf.multiply(tf.multiply(1 - hc, signs_y), diff)
                fair_penalty = fair_penalty / float(number_of_atributes)
                total_gradients.append(grad + lambda_const * fair_penalty)
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

            if role == 'bias':
                fair_penalty = tf.zeros_like(grad)
                for y_cond in [0, 1]:
                    mean_grad = tf.convert_to_tensor(
                        sum(
                            gradient_b[(g, y_cond)][tree_id] * correction_factor[(g, y_cond)] # type: ignore
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    for a in group_ids:
                        F_yc = F_y[y_cond][a]
                        diff = mean_grad - tf.cast(
                            tf.convert_to_tensor(
                                gradient_b[(a, y_cond)][tree_id] * correction_factor[(a, y_cond)] #type: ignore
                            ),
                            tf.float32,
                        )
                        hc = tf.cast(tf.math.abs(F_yc) < huber_loss_delta, tf.float32)
                        fair_penalty += tf.multiply(hc * F_yc, diff) + tf.multiply(
                            tf.multiply(1 - hc, tf.math.sign(F_yc)), diff
                        )
                fair_penalty = fair_penalty / float(2 * number_of_atributes)
                total_gradients.append(grad + lambda_const * fair_penalty)
            elif role == 'weight':
                fair_penalty = tf.zeros_like(grad)
                for y_cond in [0, 1]:
                    mean_grad = tf.convert_to_tensor(
                        sum(
                            gradient_w[(g, y_cond)][tree_id] * correction_factor[(g, y_cond)] # type: ignore
                            for g in group_ids
                        ) / number_of_atributes,
                        dtype=tf.float32,
                    )
                    for a in group_ids:
                        F_yc = F_y[y_cond][a]
                        diff = mean_grad - tf.cast(
                            tf.convert_to_tensor(
                                gradient_w[(a, y_cond)][tree_id] * correction_factor[(a, y_cond)] # type: ignore
                            ),
                            tf.float32,
                        )
                        hc = tf.cast(tf.math.abs(F_yc) < huber_loss_delta, tf.float32)
                        fair_penalty += tf.multiply(tf.multiply(hc, F_yc), diff) + tf.multiply(
                            tf.multiply(1 - hc, tf.math.sign(F_yc)), diff
                        )
                fair_penalty = fair_penalty / float(2 * number_of_atributes)
                total_gradients.append(grad + lambda_const * fair_penalty)
            else:
                total_gradients.append(grad)

    return tuple(total_gradients)
