"""Decision log 1.2: did any reported (dataset, depth) pair hit data_dim == 2**depth?

Before `build_variable_layout`, `compute_fairness_gradients` dispatched
variables by shape. `theta` is [2**depth, num_classes] and the weight test was
`shape[0] == data_dim`, so the fairness penalty landed on the wrong variable
exactly when the encoded feature count equals the number of leaves.

The encoded feature count is data-dependent (one-hot columns are fitted on the
train split), and for COMPAS the split is seed-dependent, so this loads every
dataset the way the pipeline does and reports data_dim per seed.

    python -m TESTS.check_depth_collision --seeds 1,11,22,...  --depths 3,4,5,7,9
"""

import argparse
import contextlib
import io

from src.helpers import data


def _quiet(fn, *args, **kwargs):
  with contextlib.redirect_stdout(io.StringIO()):
    return fn(*args, **kwargs)


def data_dims(seeds, folktables_states=('CA',)):
  dims = {}
  x_train = _quiet(data.read_adult, False)[0]
  dims['adult'] = {x_train.shape[1]}
  dims['compas'] = {
      _quiet(data.read_compas_train_test, 'no_drift', seed=s)[0].shape[1]
      for s in seeds
  }
  dims['folktables'] = {
      _quiet(data.read_folktables, state=list(folktables_states),
             download=False)[0].shape[1]
  }
  return dims


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seeds', default='1,11,22,33,44,55,66,77,88,99,101,'
                      '111,122,133,144,155,166,177,188,199,202,'
                      '21,22,23,24,25,26,27,28,29,30')
  parser.add_argument('--depths', default='3,4,5,6,7,8,9,10')
  args = parser.parse_args()
  seeds = sorted({int(s) for s in args.seeds.split(',') if s.strip()})
  depths = [int(d) for d in args.depths.split(',') if d.strip()]

  dims = data_dims(seeds)
  collisions = []
  print(f"{'dataset':<12} {'data_dim':<14} " + ' '.join(f'd={d:<4}' for d in depths))
  for name, values in dims.items():
    cells = []
    for d in depths:
      hit = any(v == 2 ** d for v in values)
      cells.append('HIT  ' if hit else 'ok   ')
      if hit:
        collisions.append((name, d, sorted(values)))
    print(f"{name:<12} {str(sorted(values)):<14} " + ' '.join(f'{c:<6}' for c in cells))

  if collisions:
    print('\nCOLLISIONS -- results for these pairs must be re-run:')
    for name, d, values in collisions:
      print(f'  {name}: data_dim={values}, depth={d}, 2**depth={2 ** d}')
  else:
    print('\nNo collisions.')


if __name__ == '__main__':
  main()
