"""Per-sample results of the paper's runs, for analyses that need more than W&B's summaries.

The run folders (files/experiments/wandb/<run>/) hold every prediction, label, rolling DP and
lambda_t, but not each sample's group. Streams are deterministic per seed, so the groups are
regenerated with the pipeline's own loader, checked against the stored labels, and cached in
files/experiments/stream_groups/.

    from runs import local_runs, stream_groups
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402

GROUPS_CACHE = os.path.join(mt.REPO, 'files', 'experiments', 'stream_groups')
# Settings that define a run's stream but that runs before Oct 2026 did not save in their config.
STREAM_FLAGS = {'folktables': {'folktables_subsample_fraction': 0.1},
                'intersectional_folktables': {'folktables_subsample_fraction': 0.1, 'intersectional': True}}


def local_runs(sweeps, scenario=None, models=None):
    """Every arm of every local run folder of `sweeps`, as a list of dicts: the arm's
    timestep_results plus 'model', 'seed', 'scenario', 'dataset', 'sweep' and the run's 'config'."""
    out = []
    for path in glob.glob(os.path.join(mt.REPO, 'files', 'experiments', 'wandb', '*', 'dataset_*',
                                       'seed_pipeline_results.json')):
        d = json.load(open(path))
        cfg = d.get('config', {})
        if cfg.get('wandb_sweep_id') not in sweeps:
            continue
        for run in d['seed_runs']:
            res = run['results'][0]
            if (scenario and res['scenario'] != scenario) or (models and run['model'] not in models):
                continue
            ts = dict(res['timestep_results'])
            ts.update(model=run['model'], seed=int(run['seed']), scenario=res['scenario'],
                      dataset=cfg.get('dataset') or path.split(os.sep)[-2][8:], sweep=cfg['wandb_sweep_id'],
                      config=cfg)
            out.append(ts)
    return out


def _stream_key(sweep):
    for key, sweeps in mt.SWEEPS.items():
        if sweep in sweeps:
            return key
    raise KeyError(f'sweep {sweep} is not in make_tables.SWEEPS')


def stream_groups(run):
    """The group of every sample of `run`'s stream (as returned by local_runs), regenerated with
    the pipeline loader. Raises if the regenerated labels differ from the stored ones."""
    key = _stream_key(run['sweep'])
    flags_ = dict(STREAM_FLAGS.get(key) or STREAM_FLAGS.get(key.split('_', 1)[1], {}))
    for name in ('folktables_subsample_fraction', 'intersectional'):  # saved by newer runs
        if name in run['config']:
            flags_[name] = run['config'][name]
    dataset, scenario, seed = run['dataset'], run['config'].get('drift_scenario'), run['seed']
    tag = '_'.join([dataset, scenario or 'stream', str(seed)] + [f'{k}={v}' for k, v in sorted(flags_.items())])
    path = os.path.join(GROUPS_CACHE, tag + '.npz')
    if not os.path.isfile(path):
        sys.path.insert(0, mt.REPO)
        os.environ.setdefault('PYTHONHASHSEED', '0')
        from absl import flags
        import src.main as pipeline
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(['runs'])
        for name, value in flags_.items():
            setattr(flags.FLAGS, name, value)
        splits = pipeline._load_dataset_splits(dataset, scenario or 'no_drift', int(seed))
        os.makedirs(GROUPS_CACHE, exist_ok=True)
        np.savez_compressed(path, a=np.asarray(splits[5], dtype=np.int32), y=np.asarray(splits[3], dtype=np.int32))
    cached = np.load(path)
    if not np.array_equal(cached['y'], np.asarray(run['y_true_all'], dtype=np.int32)):
        raise ValueError(f'regenerated stream differs from the stored labels: {tag}')
    return cached['a']
