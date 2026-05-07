import numpy as np
import tensorflow as tf
from river import drift
from src.models.forest.initializers import init_fairness_state, accumulate_fairness_stats, compute_fairness_gradients
from src.helpers.utils import _compute_window_fairness
from collections import deque
from src.helpers import utils

def _infer_forest_geometry(model, fallback_tree_depth, fallback_num_trees):
    """Infer tree depth and number of trees from a trained forest model."""
    inferred_num_trees = int(fallback_num_trees)
    if hasattr(model, 'layers'):
        inferred_num_trees = int(len(model.layers))

    inferred_tree_depth = int(fallback_tree_depth)
    if inferred_num_trees > 0:
        first_tree = model.layers[0]
        internal_nodes = None
        if hasattr(first_tree, 'num_internal_nodes'):
            internal_nodes = int(first_tree.num_internal_nodes)
        elif hasattr(first_tree, 'weight'):
            internal_nodes = int(first_tree.weight.shape[1])
        if internal_nodes is not None and internal_nodes > 0:
            inferred_tree_depth = int(round(np.log2(internal_nodes + 1)))

    return inferred_tree_depth, inferred_num_trees


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
        'fairness_window': 1000,
        'cooldown': 200,
        'min_samples_per_stream': 30,
        'lambda_const': float(lambda_const),
        'temperature_on_drift': 0.1,
        'temperature_recovery_target': 1.0,
        'temperature_recovery_step': 0.002,
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
    TEMP_ON_DRIFT = max(1e-4, float(defaults['temperature_on_drift']))
    TEMP_RECOVERY_TARGET = max(TEMP_ON_DRIFT, float(defaults['temperature_recovery_target']))
    TEMP_RECOVERY_STEP = max(1e-6, float(defaults['temperature_recovery_step']))
    lambda_const = float(defaults['lambda_const'])
    
    print(f"Evaluating model over {len(x_test)} timesteps with test-then-train={test_then_train}\n\
          Fairness penalty lambda: {lambda_const}, fairness type: {fairness_type}")
    
    USE_ROLLING = False
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
    print(f"Fairness window: {FAIRNESS_WINDOW}, starting fairness computations at sample index: {fairness_start}")
    if compute_fairness:
        tree_depth, num_trees = _infer_forest_geometry(model, tree_depth, num_trees)    
        num_internal_nodes = 2 ** tree_depth - 1
        all_tree_trainable_vars = []
        for tree in model.layers:
            all_tree_trainable_vars.extend(tree.trainable_variables)
        number_of_attributes = int(np.unique(np.array(a_test)).size)
        gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count = \
            init_fairness_state(num_trees, data_dim, num_internal_nodes, number_of_attributes)

    print(f"Inferred tree depth: {tree_depth}, number of trees: {num_trees}, internal nodes per tree: {num_internal_nodes}")
    pred_window = deque(maxlen=FAIRNESS_WINDOW)
    true_window = deque(maxlen=FAIRNESS_WINDOW)
    protected_window = deque(maxlen=FAIRNESS_WINDOW)
    
    huber_loss_delta = 0.1

    for t in range(n_samples):
        if (t + 1) % 1000 == 0 or t == 0:
            print(f"[DBG] Processing sample {t + 1}/{n_samples}...")
            print(f"Avg accuracy until now: {np.mean(accuracies) if accuracies else 0}")
            print(f"Avg DP until now: {np.mean(dps) if dps else 0}")
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
        
        pred_window.append(y_pred)
        true_window.append(y_t)
        protected_window.append(a_t)

        correct_buffer.append(int(y_pred == y_t))
        if USE_ROLLING and len(correct_buffer) > accuracy_window:
            correct_buffer.pop(0)
        acc_val = float(sum(correct_buffer)) / len(correct_buffer)
        accuracies.append(acc_val)

        y_probs_np  = tf.nn.softmax(y_probs, axis=-1).numpy()[0]
        model_conf  = float(y_probs_np[y_pred])   
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
                    tree.temperature.assign(TEMP_ON_DRIFT)
        dp_val, eo_val = _compute_window_fairness(
            y_preds_all=y_preds_all,
            y_true_all=y_true_all,
            a_all=a_all,
            fairness_start=fairness_start,
            fairness_window=None,
        )

        dp_val, dp_sign = utils.get_demographic_parity(list(pred_window), list(protected_window))
        eo_val, eo_sign = utils.get_equalized_odds(list(pred_window), list(protected_window), list(true_window))
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
                        tree.temperature.assign(TEMP_RECOVERY_TARGET)
                print(f"[RECOVERY] Performance restored at sample {t}. Decaying LR from {decay_from_lr:.2e} to {learning_rate:.2e} over {LR_DECAY_STEPS} steps.")
            else:
                optimizer.learning_rate.assign(float(DRIFT_LR_SPIKE))
                for tree in model.layers:
                    if hasattr(tree, 'temperature'):
                        current_temp = float(tree.temperature.value())
                        new_temp = min(TEMP_RECOVERY_TARGET, current_temp + TEMP_RECOVERY_STEP)
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
                    gradient_type, base_gamma, huber_loss_delta=huber_loss_delta, dp_sign=dp_sign, constraint_type=constraint_type
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
            'temperature_on_drift': TEMP_ON_DRIFT,
            'temperature_recovery_target': TEMP_RECOVERY_TARGET,
            'temperature_recovery_step': TEMP_RECOVERY_STEP,
        },
    }
