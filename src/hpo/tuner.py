import os
import yaml
import numpy as np
import copy
from src.models.forest.train import _build_forest_model, aranyani
from src.hpo.nsga2 import run_nsga2
from src.helpers import utils
from src.helpers.constants import NSGA2_TREE_CONFIG_PATH


def _load_tree_nsga2_config():
  cfg = {
      'enabled': False,
      'population_size': 8,
      'generations': 4,
      'seed': 42,
      'tuning_subset_ratio': 0.25,
      'eval_batch_size': 64,
      'validation_ratio': 0.2,
      'depth_bounds': [2, 6],
      'num_trees_bounds': [1, 7],
      'lambda_bounds': [0.0, 5.0],
  }
  if not os.path.exists(NSGA2_TREE_CONFIG_PATH):
    return cfg
  with open(NSGA2_TREE_CONFIG_PATH) as f:
    loaded = yaml.safe_load(f) or {}
  if not isinstance(loaded, dict):
    return cfg
  section = loaded.get('nsga2_tree', loaded)
  if not isinstance(section, dict):
    return cfg
  cfg.update(section)

  # Normalize numeric fields and bounds.
  cfg['population_size'] = int(cfg.get('population_size', 8))
  cfg['generations'] = int(cfg.get('generations', 4))
  cfg['seed'] = int(cfg.get('seed', 42))
  cfg['tuning_subset_ratio'] = float(cfg.get('tuning_subset_ratio', 0.25))
  cfg['tuning_subset_ratio'] = max(0.01, min(1.0, cfg['tuning_subset_ratio']))
  cfg['eval_batch_size'] = int(cfg.get('eval_batch_size', 64))
  cfg['validation_ratio'] = float(cfg.get('validation_ratio', 0.2))

  depth_bounds = cfg.get('depth_bounds', [2, 6])
  num_trees_bounds = cfg.get('num_trees_bounds', [1, 7])
  lambda_bounds = cfg.get('lambda_bounds', [0.0, 5.0])
  if not isinstance(depth_bounds, list) or len(depth_bounds) != 2:
    depth_bounds = [2, 6]
  if not isinstance(num_trees_bounds, list) or len(num_trees_bounds) != 2:
    num_trees_bounds = [1, 7]
  if not isinstance(lambda_bounds, list) or len(lambda_bounds) != 2:
    lambda_bounds = [0.0, 20.0]

  cfg['depth_bounds'] = [int(depth_bounds[0]), int(depth_bounds[1])]
  cfg['num_trees_bounds'] = [int(num_trees_bounds[0]), int(num_trees_bounds[1])]
  cfg['lambda_bounds'] = [float(lambda_bounds[0]), float(lambda_bounds[1])]
  return cfg

def _tune_tree_hyperparameters_nsga2(dataset, x_train, y_train, a_train,
                                     x_val, y_val, a_val,
                                     data_dim, num_class,
                                     base_depth, base_num_trees,
                                     compute_fairness, base_lambda_const,
                                     batch_size, activation, compute_mode,
                                     base_gamma, constraint_type, gradient_type,
                                     local_run, tree_cfg):
  if not bool(tree_cfg['enabled']):
    return None
  if len(x_train) == 0 or len(x_val) == 0:
    print('[NSGA2-TREE] Insufficient samples for train/validation split. Skipping.')
    return None

  print('SECOND TUNING ON VALIDATION SPLIT (TREE HYPERPARAMETERS)')
  print(
      f"[NSGA2-TREE] Train size={len(x_train)}, Val size={len(x_val)}, "
      f"base(depth={base_depth}, num_trees={base_num_trees}, lambda={base_lambda_const})"
  )

  x_train_arr = np.asarray(x_train)
  y_train_arr = np.asarray(y_train)
  a_train_arr = np.asarray(a_train)
  x_val_arr = np.asarray(x_val)
  y_val_arr = np.asarray(y_val)
  a_val_arr = np.asarray(a_val)

  subset_ratio = float(tree_cfg.get('tuning_subset_ratio', 0.25))
  subset_ratio = max(0.01, min(1.0, subset_ratio))
  tune_train_n = max(1, int(round(len(x_train_arr) * subset_ratio)))
  tune_val_n = max(1, int(round(len(x_val_arr) * subset_ratio)))

  rng = np.random.default_rng(int(tree_cfg.get('seed', 42)))
  train_indices = np.arange(len(x_train_arr))
  if tune_train_n < len(x_train_arr):
    train_indices = rng.choice(len(x_train_arr), size=tune_train_n, replace=False)
  val_indices = np.arange(len(x_val_arr))
  if tune_val_n < len(x_val_arr):
    val_indices = rng.choice(len(x_val_arr), size=tune_val_n, replace=False)

  x_tune_train = x_train_arr[train_indices]
  y_tune_train = y_train_arr[train_indices]
  a_tune_train = a_train_arr[train_indices]
  x_tune_val = x_val_arr[val_indices]
  y_tune_val = y_val_arr[val_indices]
  a_tune_val = a_val_arr[val_indices]

  print(
      f"[NSGA2-TREE] Using {subset_ratio * 100:.1f}% subset: "
      f"tune_train={len(x_tune_train)}, tune_val={len(x_tune_val)}"
  )

  bounds = {
      'depth': (float(tree_cfg['depth_bounds'][0]), float(tree_cfg['depth_bounds'][1])),
      'num_trees': (
          float(tree_cfg['num_trees_bounds'][0]),
          float(tree_cfg['num_trees_bounds'][1]),
      ),
      'lambda_const': (
          float(tree_cfg['lambda_bounds'][0]),
          float(tree_cfg['lambda_bounds'][1]),
      ),
  }

  def _objective(candidate):
    depth = int(round(candidate['depth']))
    depth = max(int(tree_cfg['depth_bounds'][0]), min(int(tree_cfg['depth_bounds'][1]), depth))

    num_trees = int(round(candidate['num_trees']))
    num_trees = max(
        int(tree_cfg['num_trees_bounds'][0]),
        min(int(tree_cfg['num_trees_bounds'][1]), num_trees),
    )

    lambda_const = float(candidate['lambda_const'])
    lambda_const = max(
        float(tree_cfg['lambda_bounds'][0]),
        min(float(tree_cfg['lambda_bounds'][1]), lambda_const),
    )

    model = _build_forest_model(
        dataset=dataset,
        data_dim=data_dim,
        num_class=num_class,
        depth=depth,
        num_trees=num_trees,
        activation=activation,
        compute_mode=compute_mode,
    )

    aranyani.train_online(
        model,
        x_tune_train,
        y_tune_train,
        a_tune_train,
        data_dim=data_dim,
        batch_size=batch_size,
        tree_depth=depth,
        compute_fairness=compute_fairness,
        lambda_const=lambda_const,
        num_trees=num_trees,
        base_gamma=base_gamma,
        constraint_type=constraint_type,
        gradient_type=gradient_type,
        local_run=local_run,
    )

    val_metrics = utils.get_test_performance(
        model,
        x_tune_val,
        y_tune_val,
        a_tune_val,
        data_dim=data_dim,
        show_confusion_matrix=False,
        eval_batch_size=int(tree_cfg.get('eval_batch_size', 64)),
    )
    return (1.0 - float(val_metrics['accuracy']), float(val_metrics['dp']))

  best_candidate, best_objective, _, _ = run_nsga2(
      objective_fn=_objective,
      bounds=bounds,
      population_size=int(tree_cfg['population_size']),
      generations=int(tree_cfg['generations']),
      seed=int(tree_cfg['seed']),
  )

  depth = int(round(best_candidate['depth']))
  depth = max(int(tree_cfg['depth_bounds'][0]), min(int(tree_cfg['depth_bounds'][1]), depth))
  num_trees = int(round(best_candidate['num_trees']))
  num_trees = max(
      int(tree_cfg['num_trees_bounds'][0]),
      min(int(tree_cfg['num_trees_bounds'][1]), num_trees),
  )
  lambda_const = float(best_candidate['lambda_const'])
  lambda_const = max(
      float(tree_cfg['lambda_bounds'][0]),
      min(float(tree_cfg['lambda_bounds'][1]), lambda_const),
  )

  selected = {
      'depth': depth,
      'num_trees': num_trees,
      'lambda_const': lambda_const,
      'objective': best_objective,
  }
  print(f"[NSGA2-TREE] Selected hyperparameters: {selected}")
  return selected

def _tune_prequential_static_params(base_model, x_stream, y_stream, a_stream, data_dim,
                                    compute_fairness, lambda_const, depth, num_trees,
                                    constraint_type, gradient_type, base_gamma, preq_cfg):
  print("FIRST TUNING ON TRAINING DATA")
  if not bool(preq_cfg['enabled']):
    return None
  if x_stream is None or len(x_stream) == 0:
    return None
  print(f"\nRunning NSGA-II tuning on training stream ({len(x_stream)} samples available)...")
  model = copy.deepcopy(base_model)
  tune_n = min(len(x_stream), int(preq_cfg['sample_size']))
  x_tune = x_stream[:tune_n]
  y_tune = y_stream[:tune_n]
  a_tune = a_stream[:tune_n]

  model_vars = list(getattr(model, 'variables', [])) or list(getattr(model, 'trainable_variables', []))
  if not model_vars:
    print("[NSGA2] Model has no variables to snapshot; skipping tuner.")
    return None
  base_state = [v.numpy().copy() for v in model_vars]

  bounds = {
      'adwin_delta_warn': (1e-6, 5e-3),
      'adwin_delta_confirm': (5e-4, 5e-2),
      'drift_lr_prewarm_mult': (1.0, 12.0),
      'drift_lr_spike_mult': (0.0, 20.0),
      'lr_decay_steps': (1.0, 6000.0),
      'fairness_window': (100.0, 2000.0),
      'cooldown': (0.0, 800.0),
      'min_samples_per_stream': (1.0, 120.0),
      'lambda_const': (0.0, 10.0),
  }

  def _restore_state():
    for var, value in zip(model_vars, base_state):
      var.assign(value)

  def _objective(candidate):
    candidate = dict(candidate)
    candidate['lr_decay_steps'] = int(round(candidate['lr_decay_steps']))
    candidate['fairness_window'] = int(round(candidate['fairness_window']))
    candidate['cooldown'] = int(round(candidate['cooldown']))
    candidate['min_samples_per_stream'] = int(round(candidate['min_samples_per_stream']))
    _restore_state()
    run = utils.evaluate_over_timesteps(
        model, x_tune, y_tune, a_tune, data_dim=data_dim,
        test_then_train=True,
        compute_fairness=compute_fairness,
        fairness_type=preq_cfg['fairness_type'],
        lambda_const=lambda_const,
        tree_depth=depth,
        num_trees=num_trees,
        constraint_type=constraint_type,
        gradient_type=gradient_type,
        base_gamma=base_gamma,
        static_params=candidate,
    )
    window = min(100, len(run['accuracy']))
    if window:
      recent_acc = float(sum(run['accuracy'][-window:]) / window)
      fairness_series = run['dp'] if preq_cfg['fairness_type'] == 'dp' else run['eo']
      recent_fair = float(sum(abs(v) for v in fairness_series[-window:]) / window)
    else:
      recent_acc = 0.0
      recent_fair = 0.0
    drift_rate = float(len(run['drifted_points'])) / max(1, int(run['n_samples']))
    drift_obj = abs(drift_rate - float(preq_cfg['target_drift_rate']))
    return (1.0 - recent_acc, recent_fair, drift_obj)

  best_candidate, best_objective, _, _ = run_nsga2(
      objective_fn=_objective,
      bounds=bounds,
      population_size=int(preq_cfg['population_size']),
      generations=int(preq_cfg['generations']),
      seed=int(preq_cfg['seed']),
  )
  best_candidate['lr_decay_steps'] = int(round(best_candidate['lr_decay_steps']))
  best_candidate['fairness_window'] = int(round(best_candidate['fairness_window']))
  best_candidate['cooldown'] = int(round(best_candidate['cooldown']))
  best_candidate['min_samples_per_stream'] = int(round(best_candidate['min_samples_per_stream']))
  _restore_state()
  print(f"[NSGA2] Tuned params selected: {best_candidate} with objectives {best_objective}")
  return best_candidate