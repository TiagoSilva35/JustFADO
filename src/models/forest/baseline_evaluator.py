"""Pure Aranyani prequential evaluator (no drift detection / no reaction).

This module mirrors `src.models.forest.evaluator.evaluate_over_timesteps` but
removes every component of the FADO reaction controller, so that a fair
comparison can be drawn between:

    - `evaluate_over_timesteps` (FADO: Aranyani + ADWIN drift detection + LR/temperature reaction)
    - `evaluate_aranyani_baseline_over_timesteps` (pure Aranyani, no controller)

Specifically, this evaluator:
  * does NOT instantiate any ADWIN detector (no warning / no confirmation),
  * does NOT modulate the optimizer learning rate (fixed at ``learning_rate``),
  * does NOT modulate the soft-routing temperature (left untouched),
  * does NOT apply the label-noise guard or cooldown logic,
  * does NOT track drift events,
  * does NOT use any of the FADO efficiency optimisations: fairness metrics are
    recomputed over the full window (legacy O(NW)) instead of maintained with
    incremental counters, and the forest is expected to run the original
    mask-product leaf-probability path (O(B x 4^n)) rather than the recursive
    updater. Both legacy paths are numerically identical to their optimised
    counterparts, so only runtime differs -- which is exactly what makes the
    reported speed-up attributable to FADO alone.

It keeps the prequential test-then-train protocol, the rolling-window
fairness monitoring, and the fairness-aware online updates (node-level
running statistics + Aranyani gradient correction).
"""

import time
from collections import deque

import numpy as np
import tensorflow as tf

from src.helpers import utils
from src.models.forest.initializers import (
    build_variable_layout,
    accumulate_fairness_stats,
    compute_fairness_gradients,
    init_fairness_state,
)


def _infer_forest_geometry(model, fallback_tree_depth, fallback_num_trees):
    """Infer tree depth and number of trees from a trained forest model.

    Identical to the helper in `evaluator.py`; duplicated here to keep this
    module self-contained and free of any drift-controller imports.
    """
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


def evaluate_aranyani_baseline_over_timesteps(
    model,
    x_test,
    y_test,
    a_test,
    data_dim,
    test_then_train=True,
    learning_rate=2e-3,
    accuracy_window=None,
    compute_fairness=True,
    fairness_type='dp',
    lambda_const=0.1,
    tree_depth=3,
    num_trees=3,
    constraint_type='node',
    gradient_type='vanilla',
    base_gamma=0.9,
    fairness_window=1000,
    static_params=None,
    use_incremental_fairness=False,
):
    """Run the Aranyani base learner prequentially without any drift response.

    Parameters mirror `evaluate_over_timesteps` so the caller can swap the two
    evaluators without changing its call site. Drift-controller-specific keys
    in `static_params` are silently ignored; only `fairness_window` and
    `lambda_const` are honoured (matching the parameters that *do* apply to
    pure Aranyani).

    Returns a dict with the same shape as `evaluate_over_timesteps`, so the
    downstream aggregation code does not need to special-case this baseline.
    `drifted_points` is always empty and `static_params_used` records only the
    knobs that were actually applied.
    """
    accuracies = []
    dps = []
    eos = []

    defaults = {
        'fairness_window': int(fairness_window),
        'lambda_const': float(lambda_const),
        # None -> follow ``fairness_window`` (B2: one time scale for all
        # stream metrics). Must mirror the FADO evaluator exactly, otherwise
        # the two arms are compared on differently-smoothed accuracy curves.
        'accuracy_window': accuracy_window,
    }
    if static_params:
        # Only the two knobs that apply to a controller-free Aranyani run are
        # honoured. We silently ignore drift-controller keys so callers can
        # reuse `_build_aranyani_static_params()` unchanged.
        for key in ('fairness_window', 'lambda_const', 'accuracy_window'):
            if key in static_params:
                defaults[key] = static_params[key]

    FAIRNESS_WINDOW = max(1, int(defaults['fairness_window']))
    lambda_const = float(defaults['lambda_const'])
    _acc_window = defaults.get('accuracy_window')
    ACCURACY_WINDOW = FAIRNESS_WINDOW if not _acc_window else max(1, int(_acc_window))

    print(
        f"[ARANYANI-BASELINE] Evaluating model over {len(x_test)} timesteps "
        f"(test-then-train={test_then_train}, fairness lambda={lambda_const}, "
        f"window={FAIRNESS_WINDOW}). No drift detection / no reaction controller."
    )

    # B1/B2 fix: rolling accuracy on the same window as the fairness metrics,
    # matching the FADO evaluator. The cumulative curve is still returned as
    # ``accuracy_cumulative`` for continuity with earlier results.
    USE_ROLLING = True
    correct_buffer = deque()
    rolling_correct = 0
    cumulative_correct = 0
    accuracies_cumulative = []
    timer = utils.PhaseTimer()
    y_preds_all = []
    y_true_all = []
    a_all = []
    n_samples = len(x_test)

    # Fixed-learning-rate optimizer for the entire stream.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    criteria = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if compute_fairness:
        tree_depth, num_trees = _infer_forest_geometry(model, tree_depth, num_trees)
        num_internal_nodes = 2 ** tree_depth - 1
        all_tree_trainable_vars = []
        for tree in model.layers:
            all_tree_trainable_vars.extend(tree.trainable_variables)
        number_of_attributes = int(np.unique(np.array(a_test)).size)
        gradient_w, gradient_b, agg_y, subgroup_count, protected_class_count = init_fairness_state(
            num_trees, data_dim, num_internal_nodes, number_of_attributes
        )
        print(
            f"[ARANYANI-BASELINE] Inferred geometry: depth={tree_depth}, "
            f"trees={num_trees}, internal_nodes={num_internal_nodes}"
        )

    # Legacy O(NW) recomputation by default: the incremental counter window is
    # a FADO-only optimisation. Values are identical either way.
    fairness_window_state = utils.make_fairness_window(
        FAIRNESS_WINDOW, incremental=use_incremental_fairness
    )
    print(
        "[ARANYANI-BASELINE] fairness monitor: "
        + ("incremental counters (O(NA))" if use_incremental_fairness
           else "legacy full-window recomputation (O(NW))")
    )
    _warn_expected_leaf_path(model, expected='mask', tag='ARANYANI-BASELINE')

    # D2: resolve the fairness penalty onto variables by identity, not by
    # tensor shape (theta collides with weight when data_dim == num_leaves).
    variable_layout = build_variable_layout(model)
    huber_loss_delta = 0.1

    for t in range(n_samples):
        if (t + 1) % 1000 == 0 or t == 0:
            print(f"[ARANYANI-BASELINE] Sample {t + 1}/{n_samples}")
            if accuracies:
                print(f"  cumulative acc: {np.mean(accuracies):.4f}")
            if dps:
                print(f"  cumulative DP : {np.mean(dps):.4f}")

        x_t = tf.convert_to_tensor(
            np.array(x_test[t], dtype=np.float32).reshape(1, data_dim)
        )
        y_t = int(y_test[t])
        a_t = int(a_test[t])

        # ---- Test phase --------------------------------------------------
        with timer.phase('predict'):
            y_probs = model(x_t, training=False)
            y_pred = int(tf.math.argmax(y_probs, axis=-1).numpy()[0])

        y_preds_all.append(y_pred)
        y_true_all.append(y_t)
        a_all.append(a_t)

        with timer.phase('fairness_metrics'):
            fairness_window_state.append(y_pred, a_t, y_t)

        correct = int(y_pred == y_t)
        cumulative_correct += correct
        correct_buffer.append(correct)
        rolling_correct += correct
        if USE_ROLLING and len(correct_buffer) > ACCURACY_WINDOW:
            rolling_correct -= correct_buffer.popleft()
        accuracies.append(rolling_correct / len(correct_buffer))
        accuracies_cumulative.append(cumulative_correct / (t + 1))

        # ---- Fairness monitoring on rolling window ----------------------
        with timer.phase('fairness_metrics'):
            dp_val, dp_sign = fairness_window_state.demographic_parity()
            eo_val, _ = fairness_window_state.equalized_odds()
        dps.append(float(dp_val))
        eos.append(float(eo_val))

        # ---- Train phase (fairness-aware Aranyani update) ---------------
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
                # B2 fix (2026-06): per-sample slicing MUST happen inside
                # the tape. See DOCS/BUG_REPORT_fairness_regulariser.md.
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
                    fairness_type, lambda_const,
                    num_internal_nodes, data_dim, number_of_attributes,
                    gradient_type, base_gamma,
                    huber_loss_delta=huber_loss_delta,
                    dp_sign=dp_sign,
                    constraint_type=constraint_type,
                    variable_layout=variable_layout,
                )
            if compute_fairness:
                del tape

            assert len(grads) == len(model.trainable_variables) and len(grads) > 0, \
                "Problem with loss gradients"
            # Fixed learning rate, fixed temperature, no controller.
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            timer.add('train_step', time.perf_counter() - _train_t0)

    print("[ARANYANI-BASELINE] Evaluation complete (no drift events tracked).")

    timing_report = timer.report(
        n_samples=n_samples,
        extra={'code_paths': {
            'leaf_probability': sorted({
                str(getattr(tr, 'leaf_probability', 'recursive'))
                for tr in getattr(model, 'layers', [])
            }),
            'fairness_metrics': type(fairness_window_state).__name__,
            'accuracy_window': ACCURACY_WINDOW,
        }},
    )
    print(utils.format_timing_report(timing_report, tag='ARANYANI-BASELINE'))

    return {
        'accuracy': accuracies,
        'accuracy_cumulative': accuracies_cumulative,
        'dp': dps,
        'eo': eos,
        'n_samples': n_samples,
        'drifted_points': [],
        'y_preds_all': list(y_preds_all),
        'y_true_all': list(y_true_all),
        'timing': timing_report,
        'static_params_used': {
            'fairness_window': FAIRNESS_WINDOW,
            'accuracy_window': ACCURACY_WINDOW,
            'lambda_const': lambda_const,
            'controller_enabled': False,
        },
    }
