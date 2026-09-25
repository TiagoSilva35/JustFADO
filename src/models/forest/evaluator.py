import time
from collections import deque

import numpy as np
import tensorflow as tf
from src.models.forest.initializers import (
    init_fairness_state, accumulate_fairness_stats,
    compute_fairness_gradients, build_variable_layout,
)
from src.helpers import utils
from src.models.forest.controller import ControllerConfig
from src.models.forest.detectors import make_detector, DEFAULT_SPEC
from src.models.forest.fairness_signal import FairnessSignal, DEFAULT_MODE
from src.models.forest.lambda_controller import LambdaController

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


def _warn_expected_leaf_path(model, expected, tag):
    """Warn (without mutating the model) if the forest is on the wrong path.

    ``expected`` is a resolved path name ('mask' / 'recursive') or None to skip
    the check. Only the baseline arm has a hard expectation: it must stay on
    the original mask path for the comparison to isolate the FADO-only
    optimisations. The FADO arm runs 'auto', which legitimately resolves to
    either path depending on tree depth.
    """
    if expected is None:
        return
    trees = getattr(model, 'layers', None)
    if not trees:
        return
    actual = {str(getattr(tree, 'leaf_probability', 'recursive')) for tree in trees}
    if actual != {expected}:
        print(
            f"[{tag}][WARN] forest is on leaf path {sorted(actual)} but this "
            f"evaluator expects {expected!r}. The FADO vs Aranyani-Base "
            f"efficiency comparison will not be isolated."
        )


def evaluate_over_timesteps(model, x_test, y_test, a_test, data_dim,
                            test_then_train=True, learning_rate=2e-3,
                            accuracy_window=None,
                            compute_fairness=True, fairness_type='dp',
                            lambda_const=0.1, tree_depth=3, num_trees=3,
                            constraint_type='node', gradient_type='vanilla',
                            base_gamma=0.9,
                            static_params=None,
                            controller=None):
    accuracies = []
    dps = []
    eos = []
    drifted_points = []

    defaults = {
        # A1 fix: in river, a LARGER delta means a MORE sensitive detector.
        # The warning stage must therefore have the larger delta so it trips
        # before the confirmation stage. The previous values were the other way
        # round (warn=1e-5, confirm=0.02), which made confirmation fire ~2x
        # earlier than the warning and left the pre-warm stage -- and
        # ``drift_lr_prewarm_mult`` -- dead code.
        'adwin_delta_warn': 0.02,
        'adwin_delta_confirm': 0.00001,
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
        # None -> follow ``fairness_window`` so every stream metric shares one
        # time scale (B2).
        'accuracy_window': accuracy_window,
        # None -> 2 x lr_decay_steps. Upper bound on how long the controller
        # may stay in the post-drift recovery regime (A3 safety net).
        'max_recovery_steps': None,
        # Detector backends, so the monitor is an experimental factor rather
        # than a hard-coded river ADWIN. See src/models/forest/detectors.py.
        'accuracy_detector': DEFAULT_SPEC,
        'fairness_detector': DEFAULT_SPEC,
        'fairness_signal_mode': DEFAULT_MODE,
        'fairness_detector_params': None,
        # Lambda controller (decision log 2.1): DP target, dual step size, cap.
        'fairness_target': 0.05,
        'lambda_dual_lr': 0.01,
        'lambda_max': 10.0,
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
    controller = ControllerConfig.from_spec(controller)
    _acc_window = defaults.get('accuracy_window')
    ACCURACY_WINDOW = FAIRNESS_WINDOW if not _acc_window else max(1, int(_acc_window))
    _max_rec = defaults.get('max_recovery_steps')
    MAX_RECOVERY_STEPS = (2 * LR_DECAY_STEPS) if not _max_rec else max(1, int(_max_rec))

    ACC_DETECTOR = str(defaults.get('accuracy_detector') or DEFAULT_SPEC)
    FAIR_DETECTOR = str(defaults.get('fairness_detector') or DEFAULT_SPEC)
    FAIR_SIGNAL_MODE = str(defaults.get('fairness_signal_mode') or DEFAULT_MODE)
    FAIR_PARAMS = dict(defaults.get('fairness_detector_params') or {})
    FAIRNESS_TARGET = float(defaults['fairness_target'])
    LAMBDA_DUAL_LR = float(defaults['lambda_dual_lr'])
    LAMBDA_MAX = float(defaults['lambda_max'])

    if ADWIN_DELTA_WARN <= ADWIN_DELTA_CONFIRM:
        print(
            f"[FADO][WARN] adwin_delta_warn={ADWIN_DELTA_WARN:g} <= "
            f"adwin_delta_confirm={ADWIN_DELTA_CONFIRM:g}. In river a larger delta is "
            f"MORE sensitive, so the warning stage will not trip before confirmation "
            f"and the pre-warm phase (drift_lr_prewarm_mult) will never run."
        )

    print(f"Evaluating model over {len(x_test)} timesteps with test-then-train={test_then_train}\n\
          Fairness penalty lambda: {lambda_const}, fairness type: {fairness_type}")
    print(f"Accuracy window: {ACCURACY_WINDOW} (rolling), max recovery steps: {MAX_RECOVERY_STEPS}")
    print(f"Controller components: {controller.describe()}")

    # B1 fix: report a ROLLING accuracy, not the cumulative curve. Averaging a
    # cumulative curve gives sample i a weight ~ ln(N/i), so everything after
    # 60% of the stream -- the only region where the controller can differ from
    # the baseline -- was worth under 10% of the reported number.
    # B2 fix: it defaults to the fairness window, so accuracy and DP/EO finally
    # share one time scale. The cumulative curve is still returned as
    # ``accuracy_cumulative`` for continuity with earlier results.
    USE_ROLLING = True
    fairness_drift_points = []
    fairness_signal = None
    fairness_detectors = None
    correct_buffer = deque()
    rolling_correct = 0
    cumulative_correct = 0
    accuracies_cumulative = []
    recovery_deadline = 0
    just_confirmed_drift = False
    timer = utils.PhaseTimer()
    warn_det = make_detector(ACC_DETECTOR, delta=ADWIN_DELTA_WARN)
    acc_det  = make_detector(ACC_DETECTOR, delta=ADWIN_DELTA_CONFIRM)
    acc_det_n = 0
    in_warning = False           
    last_detected_acc = -COOLDOWN
    recovering_from_drift = False
    baseline_accuracy = 0.0
    y_preds_all = []
    y_true_all = []
    n_samples = len(x_test)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    steps_since_drift = 0
    decay_from_lr = float(DRIFT_LR_PREWARM)
    fairness_start = 0
    print(f"Fairness window: {FAIRNESS_WINDOW}, starting fairness computations at sample index: {fairness_start}")
    # Needed by the fairness monitor whether or not the regulariser is on.
    number_of_attributes = int(np.unique(np.array(a_test)).size)
    if compute_fairness:
        tree_depth, num_trees = _infer_forest_geometry(model, tree_depth, num_trees)    
        num_internal_nodes = 2 ** tree_depth - 1
        all_tree_trainable_vars = []
        for tree in model.layers:
            all_tree_trainable_vars.extend(tree.trainable_variables)
        gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count = \
            init_fairness_state(num_trees, data_dim, num_internal_nodes, number_of_attributes)

    print(f"Inferred tree depth: {tree_depth}, number of trees: {num_trees}, internal nodes per tree: {num_internal_nodes}")

    # 2.1: lambda becomes the dual variable of DP <= FAIRNESS_TARGET. Without
    # the controller it stays at lambda_const, as in Aranyani-Base.
    lambda_controller = (
        LambdaController(lambda_const, epsilon=FAIRNESS_TARGET,
                         eta=LAMBDA_DUAL_LR, lambda_max=LAMBDA_MAX,
                         num_groups=number_of_attributes)
        if controller.react_lambda else None)
    lambda_t = float(lambda_const)
    lambdas = []
    fairness_resets = []

    def _reset_fairness_stats(t, reason):
        # 2.1 prerequisite: the penalty's statistics are running means since
        # the stream began; after a confirmed drift they describe the old
        # concept, so they are forgotten and rebuilt from post-drift samples.
        nonlocal gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count
        if not (compute_fairness and controller.reset_fairness_stats):
            return
        gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count = \
            init_fairness_state(num_trees, data_dim, num_internal_nodes, number_of_attributes)
        fairness_resets.append(t)
        print(f"[FAIRNESS] Reset fairness statistics at sample {t} ({reason} drift).")
    # FADO runs the optimised monitors: incremental O(NA) counters here and the
    # recursive O(B x 2^n) leaf-probability updater in the forest. The
    # Aranyani-Base evaluator deliberately keeps the legacy paths so the
    # efficiency gain is attributable to FADO alone.
    fairness_window = utils.make_fairness_window(FAIRNESS_WINDOW, incremental=True)
    _warn_expected_leaf_path(model, expected=None, tag='FADO')
    
    # D2: resolve the fairness penalty onto variables by identity, not by
    # tensor shape (theta collides with weight when data_dim == num_leaves).
    variable_layout = build_variable_layout(model)
    huber_loss_delta = 0.1

    for t in range(n_samples):
        if (t + 1) % 1000 == 0 or t == 0:
            print(f"[DBG] Processing sample {t + 1}/{n_samples}...")
            print(f"Avg accuracy until now: {np.mean(accuracies) if accuracies else 0}")
            print(f"Last mean accuracy over window: {np.mean(accuracies[-ACCURACY_WINDOW:]) if accuracies else 0}")
            print(f"Avg DP until now: {np.mean(dps) if dps else 0}")
        x_t = tf.convert_to_tensor(
            np.array(x_test[t], dtype=np.float32).reshape(1, data_dim)
        )
        y_t = int(y_test[t])
        a_t = int(a_test[t])

        with timer.phase('predict'):
            y_probs = model(x_t, training=False)
            y_pred  = int(tf.math.argmax(y_probs, axis=-1).numpy()[0])
        error   = int(y_pred != y_t)

        y_preds_all.append(y_pred)
        y_true_all.append(y_t)

        with timer.phase('fairness_metrics'):
            fairness_window.append(y_pred, a_t, y_t)

        # Rolling accuracy kept with an incremental counter so the reporting
        # fix does not reintroduce an O(N*W) rescan of the window.
        correct = int(y_pred == y_t)
        cumulative_correct += correct
        correct_buffer.append(correct)
        rolling_correct += correct
        if USE_ROLLING and len(correct_buffer) > ACCURACY_WINDOW:
            rolling_correct -= correct_buffer.popleft()
        acc_val = rolling_correct / len(correct_buffer)
        accuracies.append(acc_val)
        accuracies_cumulative.append(cumulative_correct / (t + 1))

        y_probs_np  = tf.nn.softmax(y_probs, axis=-1).numpy()[0]
        model_conf  = float(y_probs_np[y_pred])   
        is_label_noise = (error == 1) and (model_conf > 0.70)

        
        just_confirmed_drift = False
        warn_fired = acc_fired = False
        if controller.detect_accuracy_drift:
            with timer.phase('drift_detect'):
                warn_fired = warn_det.update(error)
                acc_fired = acc_det.update(error)
            acc_det_n += 1
        if not controller.label_noise_guard:
            is_label_noise = False

        if (controller.detect_accuracy_drift
                and controller.prewarm
                and warn_fired
                and not in_warning
                and acc_det_n >= MIN_SAMPLES_PER_STREAM
                and t - last_detected_acc >= COOLDOWN
                and not is_label_noise):
            in_warning = True
            optimizer.learning_rate.assign(float(DRIFT_LR_PREWARM))
            decay_from_lr = float(DRIFT_LR_PREWARM)
            steps_since_drift = LR_DECAY_STEPS
            warn_det.reset()
            print(f"[WARN] Drift warning at sample {t} — pre-warming LR to {DRIFT_LR_PREWARM:.2e}")

        if (controller.detect_accuracy_drift
                and acc_fired
                and acc_det_n >= MIN_SAMPLES_PER_STREAM
                and t - last_detected_acc >= COOLDOWN
                # A2 fix: the label-noise guard used to sit only on the warning
                # branch. Since (pre-A1) confirmation always fired first, the
                # guard was unreachable and a single high-confidence
                # mislabelled sample could trigger a full LR spike. The
                # scenarios inject Bernoulli label flips by design, so this
                # branch needs the guard at least as much as the warning one.
                and not is_label_noise):
            drifted_points.append(t)
            last_detected_acc = t
            in_warning = False
            acc_det.reset()
            warn_det.reset()
            acc_det_n = 0
            if controller.react_lr:
                optimizer = tf.keras.optimizers.Adam(learning_rate=DRIFT_LR_SPIKE)
            steps_since_drift = LR_DECAY_STEPS
            # A3 fix: the recovery reference must come from BEFORE the drift.
            # ADWIN confirms with a lag, so a window ending at ``t`` is already
            # contaminated by the post-drift samples that triggered it; the
            # reference is taken one further accuracy window back. The old
            # version compared the *cumulative* accuracy curve against its own
            # trailing mean -- a test that in practice never passed, leaving
            # the controller pinned at DRIFT_LR_SPIKE for the rest of the
            # stream and making ``lr_decay_steps`` dead code.
            ref_end = max(1, t - ACCURACY_WINDOW)
            ref_start = max(0, ref_end - ACCURACY_WINDOW)
            baseline_accuracy = float(np.mean(accuracies[ref_start:ref_end]))
            recovering_from_drift = True
            just_confirmed_drift = True
            recovery_deadline = t + MAX_RECOVERY_STEPS
            
            print(f"[DRIFT] Concept drift confirmed at sample {t} — spiking LR to {DRIFT_LR_SPIKE:.2e} and making hard routing decisions")
            _reset_fairness_stats(t, 'accuracy')
            if controller.react_temperature:
                for tree in model.layers:
                    if hasattr(tree, 'temperature'):
                        tree.temperature.assign(TEMP_ON_DRIFT)
        with timer.phase('fairness_metrics'):
            dp_val, dp_sign = fairness_window.demographic_parity()
            eo_val, eo_sign = fairness_window.equalized_odds()
        dps.append(float(dp_val))
        eos.append(float(eo_val))

        # Fairness monitoring. Observation only for now: it records when a
        # fairness drift would have been signalled, so detector/signal choices
        # can be compared on detection quality before any reaction is wired to
        # them. The signal design matters more than the detector -- see
        # src/models/forest/fairness_signal.py.
        if fairness_detectors is None and controller.detect_fairness_drift:
            fairness_signal = FairnessSignal(
                FAIR_SIGNAL_MODE, num_groups=number_of_attributes,
                window=FAIRNESS_WINDOW)
            fairness_detectors = {
                channel: make_detector(FAIR_DETECTOR, **FAIR_PARAMS)
                for channel in fairness_signal.channels
            }
            print(f"Fairness monitor: signal={FAIR_SIGNAL_MODE}, "
                  f"detector={FAIR_DETECTOR}, channels={fairness_signal.channels}")
        if fairness_detectors is not None:
            with timer.phase('fairness_detect'):
                observed = fairness_signal.observe(y_pred, a_t, dp_value=dp_val)
                if observed:
                    for channel, value in observed.items():
                        if fairness_detectors[channel].update(value):
                            fairness_drift_points.append(t)
                            _reset_fairness_stats(t, 'fairness')
                            break

        if lambda_controller is not None:
            with timer.phase('lambda_update'):
                lambda_t = lambda_controller.update(y_pred, a_t)
        lambdas.append(lambda_t)
        
        if just_confirmed_drift:
            # A4 fix: hold TEMP_ON_DRIFT for this timestep. Previously the
            # recovery ramp ran in the same iteration as the confirmation, so
            # the configured drift temperature was overwritten (0.1 -> 0.102)
            # before any forward or backward pass ever used it.
            pass
        elif steps_since_drift > 0 and not recovering_from_drift:
            alpha = steps_since_drift / LR_DECAY_STEPS
            current_lr = learning_rate + alpha * (decay_from_lr - learning_rate)
            if controller.react_lr:
                optimizer.learning_rate.assign(float(current_lr))
            steps_since_drift -= 1
        elif recovering_from_drift:
            current_acc = acc_val
            recovery_timed_out = t >= recovery_deadline
            if current_acc >= baseline_accuracy or recovery_timed_out:
                recovering_from_drift = False
                if recovery_timed_out:
                    print(
                        f"[RECOVERY] Timed out at sample {t} after "
                        f"{MAX_RECOVERY_STEPS} steps without regaining the "
                        f"pre-drift accuracy ({baseline_accuracy:.4f}); leaving "
                        f"the recovery regime anyway."
                    )
                decay_from_lr = float(optimizer.learning_rate)
                steps_since_drift = LR_DECAY_STEPS
                if controller.react_temperature:
                    for tree in model.layers:
                        if hasattr(tree, 'temperature'):
                            tree.temperature.assign(TEMP_RECOVERY_TARGET)
                print(f"[RECOVERY] Performance restored at sample {t}. Decaying LR from {decay_from_lr:.2e} to {learning_rate:.2e} over {LR_DECAY_STEPS} steps.")
            else:
                if controller.react_lr:
                    optimizer.learning_rate.assign(float(DRIFT_LR_SPIKE))
                if controller.react_temperature:
                    for tree in model.layers:
                        if hasattr(tree, 'temperature'):
                            current_temp = float(tree.temperature.value())
                            new_temp = min(TEMP_RECOVERY_TARGET, current_temp + TEMP_RECOVERY_STEP)
                            tree.temperature.assign(new_temp)

        if test_then_train:
            _train_t0 = time.perf_counter()
            y_t_tensor = tf.convert_to_tensor([y_t], dtype=tf.int32)
            # The 'node' fairness gradient is analytic (no tape traversal), so
            # only 'leaf' needs a second ``.gradient`` call and thus a
            # persistent tape.
            with tf.GradientTape(
                persistent=(compute_fairness and constraint_type == 'leaf')
            ) as tape:
                train_out = model(x_t, training=True)
                y_probs_train = train_out[0] if isinstance(train_out, tuple) else train_out
                node_decisions_train = train_out[1] if isinstance(train_out, tuple) else None
                loss = criteria(y_true=y_t_tensor, y_pred=y_probs_train)
                # B2 fix (2026-06): per-sample slicing MUST happen inside the
                # tape's with-block so the slice ops are recorded. Slicing
                # ``node_decisions_train[:, i]`` later in
                # ``accumulate_fairness_stats`` (after tape __exit__) creates
                # an untracked tensor and makes ``tape.gradient`` return all
                # None, silently disabling the fairness regulariser. See
                # ``DOCS/BUG_REPORT_fairness_regulariser.md``.
                if compute_fairness and node_decisions_train is not None:
                    node_decisions_per_sample = tf.unstack(
                        node_decisions_train, axis=1
                    )
                    predictions_per_sample = tf.unstack(y_probs_train, axis=0)
                else:
                    node_decisions_per_sample = None
                    predictions_per_sample = None
            if compute_fairness and node_decisions_train is not None:
                accumulate_fairness_stats(
                    tape, [a_t], [y_t],
                    node_decisions_per_sample, predictions_per_sample,
                    all_tree_trainable_vars, model.trainable_variables,
                    gradient_w, gradient_b, agg_y,
                    subgroup_count, protected_class_count,
                    num_internal_nodes, data_dim,
                    constraint_type, gradient_type, base_gamma,
                    inputs=x_t, model=model,
                )
            grads = tape.gradient(loss, model.trainable_variables)
            if compute_fairness and node_decisions_train is not None:
                grads = compute_fairness_gradients(
                    grads, gradient_w, gradient_b, agg_y,
                    subgroup_count, protected_class_count,
                    fairness_type, lambda_t,
                    num_internal_nodes, data_dim, number_of_attributes,
                    gradient_type, base_gamma, huber_loss_delta=huber_loss_delta, dp_sign=dp_sign, constraint_type=constraint_type,
                    variable_layout=variable_layout,
                )
            if compute_fairness:
                del tape

            assert len(grads) == len(model.trainable_variables) and len(grads) > 0, "Problem with loss gradients"
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            timer.add('train_step', time.perf_counter() - _train_t0)

    if drifted_points:
        print(f"Accuracy drift detected at samples: {drifted_points}")
    else:
        print("No accuracy drift detected over the test set.")

    timing_report = timer.report(
        n_samples=n_samples,
        extra={'code_paths': {
            'leaf_probability': sorted({
                str(getattr(tr, 'leaf_probability', 'recursive'))
                for tr in getattr(model, 'layers', [])
            }),
            'fairness_metrics': type(fairness_window).__name__,
            'accuracy_window': ACCURACY_WINDOW,
        }},
    )
    print(utils.format_timing_report(timing_report, tag='FADO'))

    return {
        'accuracy': accuracies,
        'accuracy_cumulative': accuracies_cumulative,
        'dp': dps,
        'eo': eos,
        'n_samples': n_samples,
        'drifted_points': drifted_points,
        'fairness_drifted_points': fairness_drift_points,
        # 2.1: lambda at every step (constant lambda_const without the
        # controller) and the samples where the fairness statistics were reset.
        'lambda': lambdas,
        'fairness_resets': fairness_resets,
        'lambda_controller': (lambda_controller.describe()
                              if lambda_controller else None),
        'detectors': {
            'accuracy_warn': warn_det.describe(),
            'accuracy_confirm': acc_det.describe(),
            'fairness': (
                {ch: d.describe() for ch, d in fairness_detectors.items()}
                if fairness_detectors else None),
            'fairness_signal': fairness_signal.describe() if fairness_signal else None,
        },
        'y_preds_all': list(y_preds_all),
        'y_true_all': list(y_true_all),
        'timing': timing_report,
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
            'accuracy_window': ACCURACY_WINDOW,
            'max_recovery_steps': MAX_RECOVERY_STEPS,
            'accuracy_detector': ACC_DETECTOR,
            'fairness_detector': FAIR_DETECTOR,
            'fairness_signal_mode': FAIR_SIGNAL_MODE,
            'fairness_target': FAIRNESS_TARGET,
            'lambda_dual_lr': LAMBDA_DUAL_LR,
            'lambda_max': LAMBDA_MAX,
        },
        'controller': controller.as_dict(),
    }
