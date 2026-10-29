"""Check that a forest reused from --pretrain_cache_dir is the forest pre-training produced.

Pre-trains on a small slice of Adult, then clears the in-process cache and asks again, so the
second forest comes from the persistent checkpoint. Both must give bit-identical outputs and
carry the same routing temperature; otherwise reusing a cached forest would change results.

    PYTHONHASHSEED=0 python -m TESTS.check_pretrain_cache
"""
import tempfile

import numpy as np
from absl import flags

import src.main as pipeline


def main():
    cache = tempfile.mkdtemp(prefix='pretrain_cache_check_')
    flags.FLAGS(['check', f'--pretrain_cache_dir={cache}', '--pipeline_dataset=adult'])
    x_train, x_test, y_train, _, a_train, _, _, _ = pipeline._load_dataset_splits('adult', 'no_drift', 66)
    x = np.asarray(x_train[:400], dtype=np.float32)
    y, a = np.asarray(y_train[:400], dtype=np.int32), np.asarray(a_train[:400], dtype=np.int32)
    probe = np.asarray(x_test[:500], dtype=np.float32)

    fresh, _ = pipeline._pretrain_aranyani(x, y, a, x.shape[1], seed=66)
    pipeline._PRETRAIN_CACHE.clear()  # force the persistent path
    cached, _ = pipeline._pretrain_aranyani(x, y, a, x.shape[1], seed=66)

    out_fresh, out_cached = fresh(probe, training=False).numpy(), cached(probe, training=False).numpy()
    temps = [(float(getattr(t, 'temperature', 1.0)), float(getattr(u, 'temperature', 1.0)))
             for t, u in zip(fresh.layers, cached.layers)]
    same_out = np.array_equal(out_fresh, out_cached)
    same_temp = all(f == c for f, c in temps)
    print(f'outputs identical: {same_out}; temperatures identical: {same_temp} {temps}')
    if not (same_out and same_temp):
        raise SystemExit('FAIL: the cached forest differs from the freshly pre-trained one')
    print('ok')


if __name__ == '__main__':
    main()
