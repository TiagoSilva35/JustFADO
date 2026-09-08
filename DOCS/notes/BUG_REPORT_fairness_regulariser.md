# Bug report: fairness regulariser silently disabled in all FADO / Aranyani-Base runs
---

## Summary

Two independent structural bugs combine to make the fairness regulariser dead
code. Either bug alone is sufficient to kill the regulariser; both are present
in the released code.

| Bug | Location | Effect |
|---|---|---|
| **B1: Tape slicing after `__exit__`** | `initializers.py:51,53` (`accumulate_fairness_stats`) | `node_decisions[:, i]` slice is computed *after* the `tf.GradientTape` `with`-block has exited → tape did not record the slice op → `tape.gradient(...)` returns all `None` → `gradient_b` / `gradient_w` stay at zero-initialisation → `fair_penalty` evaluates to a pure-zero tensor even after B1 is fixed. |

**Symptom:** `lambda_const` is dead code. Any tuning, any λ schedule, any
fairness override has zero effect on the model trajectory or the produced
metrics. FADO and Aranyani-Base behave as **vanilla classification-loss-only
soft-routed decision forests** during the prequential phase, with FADO
additionally executing its drift-detection + LR-spike + temperature-sharpening
pathway on top.

---

## Evidence

### B1: the tape never records the per-sample slice

The forest forward pass produces a per-batch tensor:

```python
# forest.py
all_node_decisions = tf.stack(all_node_decisions, axis=0)
return final_prediction, all_node_decisions, stacked_predictions
```

`all_node_decisions` has shape `[num_trees, batch_size, num_internal_nodes]`.

In the evaluator (`evaluator.py`, FADO test-then-train branch):

```python
with tf.GradientTape(persistent=compute_fairness) as tape:
    train_out = model(x_t, training=True)
    y_probs_train       = train_out[0]
    node_decisions_train = train_out[1]
    loss = criteria(y_true=y_t_tensor, y_pred=y_probs_train)
# ← tape's `with`-block has exited here
if compute_fairness and node_decisions_train is not None:
    accumulate_fairness_stats(tape, ..., node_decisions_train, ...)
```

Then inside `accumulate_fairness_stats`:

```python
fair_gradients = tape.gradient(node_decisions[:, i], all_tree_trainable_vars)
```

The slice `node_decisions[:, i]` is a TensorFlow op (`__getitem__`) that
**executes when this line runs**, which is **after the `with` block exited**.
The tape stops recording new ops once `__exit__` runs. The sliced tensor was
never seen by the tape. `tape.gradient(unseen_tensor, sources)` returns `None`
for every source.

Diagnostic captured live during a COMPAS `abrupt_race` seed 42 run:

```
[ACC-DBG] constraint_type=node len(fair_gradients)=12 num_internal_nodes=15 data_dim=19
[ACC-DBG]     [0..11] None        ← all 12 sliced gradients are None
[ACC-DBG]   alt sources:
              node_decisions(whole)->all_tree_vars None_count=4/12   ← 8 of 12 work (W and B); the 4 None are theta
              predictions->all_tree_vars         None_count=0/12   ← all 12 work
              predictions->model_vars            None_count=0/12   ← all 12 work
[ACC-DBG]   var-set overlap: |all_tree_vars|=12 |model_vars|=12 intersection=12  ← variables are identical objects
```

Interpretation:
- `predictions` (computed inside the tape) → all-tree-vars: 0 None, full gradient flow.
- `node_decisions` (whole, computed inside the tape) → all-tree-vars: 4 None, but those are the `theta` variables which legitimately don't affect node_decisions. The 8 non-None are W and B per tree — correct.
- `node_decisions[:, i]` (slice computed **outside** the tape) → all-tree-vars: all 12 None — the slice broke the gradient graph.

This confirms the bug is the slice timing, not the variables or the math.

### Consequence

Because `tape.gradient(node_decisions[:, i], ...)` returns all `None`, the loop
inside `accumulate_fairness_stats` that writes to `gradient_b[(a, y)]` and
`gradient_w[(a, y)]` **never executes its body** — every iteration hits
`if fair_grad is None: continue`. So `gradient_b` and `gradient_w` stay at
their zero initialisation for the entire stream.

When `compute_fairness_gradients` later reads:

```python
gb_a = gradient_b[(k, 0)] + gradient_b[(k, 1)]    # zero
mean_other = sum(...)/number_of_atributes         # zero
diff = mean_other - gb_a[idx_b] * cf_a            # zero
fair_penalty += hc*F*diff + (1-hc)*signs_y*diff   # zero
```

every term is zero, so `fair_penalty = 0.0` exactly. Then:

```python
grad + lambda_const * fair_penalty  ==  grad + 0  ==  grad
```

The "fairness-aware" gradient is identical to the pure classification
gradient, regardless of λ.

Live confirmation from the same diagnostic run:
```
[FAIR-DBG]   b-grad: |fair_penalty|_max=0.000000e+00 |grad|_max=8.394037e-03 |lambda*fair|_max=0.000000e+00
```

Every single call across 20 observations: `fair_penalty == 0`.

