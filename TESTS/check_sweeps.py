"""Static check of every W&B sweep in src/configs/sweeps before launching it.

A sweep that names an unknown flag, a scenario that is not registered for its
dataset, or a model the pipeline rejects only fails once an agent picks up
that cell -- possibly hours into a grid. This catches all three up front, plus
a command that cannot import ``src``.

    python -m TESTS.check_sweeps
"""

import glob
import itertools
import os
import sys

import yaml

import src.main  # noqa: F401 -- registers every pipeline flag
from src.drift.compas_scenarios import COMPAS_SCENARIOS
from src.drift.scenarios import SCENARIOS
from src.main import FLAGS, _ABLATION_MODELS, _supported_models_for_dataset

SWEEP_DIR = os.path.join('src', 'configs', 'sweeps')
SCENARIOS_BY_DATASET = {'compas': set(COMPAS_SCENARIOS), 'adult': set(SCENARIOS)}


def _values(spec):
  if 'values' in spec:
    return list(spec['values'])
  if 'value' in spec:
    return [spec['value']]
  return None  # a distribution


def check(path):
  errors = []
  with open(path) as f:
    sweep = yaml.safe_load(f)
  params = sweep.get('parameters', {})

  for name in params:
    if name not in FLAGS:
      errors.append(f'unknown flag --{name}')

  command = [str(token) for token in sweep.get('command', [])]
  if '-m' not in command or 'src.main' not in command:
    errors.append('command must run "-m src.main" (python src/main.py cannot import src)')

  datasets = _values(params.get('pipeline_dataset', {})) or [None]
  scenarios = _values(params.get('drift_scenario', {})) or [None]
  models = _values(params.get('pipeline_model', {})) or ['']
  for dataset, scenario, model_spec in itertools.product(datasets, scenarios, models):
    if dataset is None:
      errors.append('pipeline_dataset must be fixed by the sweep')
      break
    registered = SCENARIOS_BY_DATASET.get(dataset)
    if scenario is not None and registered is not None and scenario not in registered:
      errors.append(f'scenario {scenario!r} is not registered for {dataset}')
    selectable = set(_supported_models_for_dataset(dataset)) | set(_ABLATION_MODELS)
    for model in filter(None, (m.strip() for m in str(model_spec).split(','))):
      if model not in selectable:
        errors.append(f'model {model!r} is not selectable for {dataset}')

  if sweep.get('method') == 'grid':
    for name, spec in params.items():
      if _values(spec) is None:
        errors.append(f'grid sweep has a distribution for --{name}')
  return sorted(set(errors))


def main():
  failed = False
  for path in sorted(glob.glob(os.path.join(SWEEP_DIR, '*.yaml'))):
    errors = check(path)
    print(f"{'FAIL' if errors else 'ok  '} {path}")
    for error in errors:
      print(f'       {error}')
    failed |= bool(errors)
  sys.exit(1 if failed else 0)


if __name__ == '__main__':
  main()
