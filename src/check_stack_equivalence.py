#!/usr/bin/env python3
"""Compare a prequential run across two dependency stacks.

The 2026-09 upgrade crosses the Keras 2 -> Keras 3 boundary (tensorflow 2.8 ->
2.21). Everything runs, but the optimizer implementation was rewritten, and
that is the one place where numbers could move. This script pins down whether
they did.

Run it once in each environment, then diff:

    # old env (python 3.9, tensorflow 2.8)
    python src/check_stack_equivalence.py --out files/equiv_old.json

    # new env (python 3.12, tensorflow 2.21)
    python src/check_stack_equivalence.py --out files/equiv_new.json

    # compare
    python src/check_stack_equivalence.py --compare files/equiv_old.json files/equiv_new.json

Deliberately synthetic and self-contained: no dataset files, no wandb, one
seed, a few hundred timesteps. It is checking the numerics of the forest +
optimizer, not the science.

If the streams diverge, the first suspect is the optimizer. Try pinning
``tf.keras.optimizers.legacy.Adam`` (TF 2.11-2.15) or
``tf_keras.optimizers.Adam`` as a bridge and re-run.
"""

import argparse
import json
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MPLBACKEND', 'Agg')

N_TRAIN, N_TEST, DIM, DEPTH, TREES, SEED = 300, 400, 6, 3, 2, 42


def _synthetic():
    import numpy as np
    rng = np.random.default_rng(0)
    xtr = rng.normal(size=(N_TRAIN, DIM)).astype(np.float32)
    atr = rng.integers(0, 2, N_TRAIN).astype(np.int32)
    ytr = (xtr[:, 0] + 0.8 * atr > 0).astype(np.int32)
    xte = rng.normal(size=(N_TEST, DIM)).astype(np.float32)
    ate = rng.integers(0, 2, N_TEST).astype(np.int32)
    yte = (xte[:, 0] + 0.8 * ate > 0).astype(np.int32)
    yte[N_TEST // 2:] = 1 - yte[N_TEST // 2:]      # abrupt concept change
    return xtr, ytr, atr, xte, yte, ate


def run(out_path):
    import numpy as np
    import tensorflow as tf
    from src.models.forest import forest, aranyani
    from src.models.forest.evaluator import evaluate_over_timesteps
    from src.models.forest.train import _set_global_seed

    _set_global_seed(SEED)
    xtr, ytr, atr, xte, yte, ate = _synthetic()

    model = forest.FairDecisionForest(
        num_trees=TREES, data_dim=DIM, tree_depth=DEPTH, num_classes=2,
        leaf_probability='mask')          # pinned: not the thing under test
    aranyani.train_online(
        model, xtr, ytr, atr, data_dim=DIM, batch_size=1, tree_depth=DEPTH,
        compute_fairness=True, lambda_const=0.5, num_trees=TREES,
        local_run=True, fairness_window=100)

    _set_global_seed(SEED)
    res = evaluate_over_timesteps(
        model, xte, yte, ate, data_dim=DIM, test_then_train=True,
        lambda_const=0.5, tree_depth=DEPTH, num_trees=TREES,
        static_params={'fairness_window': 100, 'cooldown': 50,
                       'min_samples_per_stream': 20, 'lr_decay_steps': 150,
                       'lambda_const': 0.5})

    weights = np.concatenate([v.numpy().reshape(-1) for v in model.trainable_variables])
    payload = {
        'versions': {
            'python': sys.version.split()[0],
            'tensorflow': tf.__version__,
            'numpy': np.__version__,
        },
        'accuracy': [float(v) for v in res['accuracy']],
        'dp': [float(v) for v in res['dp']],
        'eo': [float(v) for v in res['eo']],
        'drifted_points': list(res['drifted_points']),
        'final_weights_sha': _sha(weights),
        'weight_summary': {
            'mean': float(weights.mean()), 'std': float(weights.std()),
            'min': float(weights.min()), 'max': float(weights.max()),
        },
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as handle:
        json.dump(payload, handle)
    print(f"wrote {out_path}")
    print(f"  {payload['versions']}")
    print(f"  mean accuracy {np.mean(payload['accuracy']):.6f} | "
          f"mean DP {np.mean(payload['dp']):.6f} | drifts {payload['drifted_points']}")


def _sha(array):
    import hashlib
    import numpy as np
    return hashlib.sha1(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()[:16]


def compare(path_a, path_b):
    import numpy as np
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    print(f"A: {a['versions']}")
    print(f"B: {b['versions']}")
    worst = 0.0
    for key in ('accuracy', 'dp', 'eo'):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape:
            print(f"  {key:<10} LENGTH MISMATCH {x.shape} vs {y.shape}")
            worst = float('inf')
            continue
        diff = float(np.max(np.abs(x - y)))
        worst = max(worst, diff)
        print(f"  {key:<10} max |A-B| = {diff:.3e}   mean A {x.mean():.6f}  mean B {y.mean():.6f}")
    print(f"  drift points   A {a['drifted_points']}  B {b['drifted_points']}")
    print(f"  weight sha     A {a['final_weights_sha']}  B {b['final_weights_sha']}")
    print()
    if worst == 0.0:
        print("VERDICT: bit-identical. The upgrade does not move the numbers.")
    elif worst < 1e-5:
        print(f"VERDICT: equivalent to {worst:.1e} (float noise from op ordering). Safe.")
    elif worst < 1e-2:
        print(f"VERDICT: small drift ({worst:.1e}). Re-run the reported experiments on "
              "one stack; do not mix results across stacks.")
    else:
        print(f"VERDICT: DIVERGENT ({worst:.1e}). Suspect the optimizer "
              "(Keras 2 -> Keras 3). Try legacy Adam as a bridge and re-run.")
    return worst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='files/equiv.json')
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.out)
