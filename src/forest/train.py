"""Driver code."""


import os
import yaml
import copy

import numpy as np
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from absl import flags

import src.helpers.data as data
import src.forest.forest as forest
import src.forest.aranyani as aranyani

from src.helpers.plots import plot_metrics_over_timesteps

import src.helpers.utils as utils
import src.forest.clip_forest as clip_forest

from src.drift.create_drifted_ds import generate_drifted_dataset


flags.DEFINE_string('sweep_id', '-1', 'Wandb sweep ID.')
flags.DEFINE_float('lambda_const', 0.1, 'Cell to run.')
flags.DEFINE_string('dataset', 'civil', 'Dataset name.')
flags.DEFINE_integer('max_iter', 1, 'Total number of iterations.')
flags.DEFINE_integer('depth', 4, 'Tree depth.')
flags.DEFINE_integer('num_trees', 3, 'Number of trees.')
flags.DEFINE_bool(
    'compute_fairness', True, 'Whether to apply fairness constraints.'
)
flags.DEFINE_integer('batch_size', 1, 'Samples in an online batch.')
flags.DEFINE_string('activation', 'sigmoid', 'Activation function.')
flags.DEFINE_string('compute_mode', 'default', 'log or default.')
flags.DEFINE_string('base_gamma', None, 'gamma for the gradients.')
flags.DEFINE_string('constraint_type', 'node', '[node, leaf]')
flags.DEFINE_string('gradient_type', 'vanilla', '[vanilla, momentum, ema]')
flags.DEFINE_string('encoder_model', 'instructor', '[bert, instructor]')
flags.DEFINE_string('offline_loss_type', 'mmd', '[mmd, l2, l1]')
flags.DEFINE_bool('use_correlation_penalty', False, 
                  'Whether to use dynamic correlation-based penalties.')
flags.DEFINE_float('correlation_threshold', 0.3,
                   'Threshold for marking features as correlated.')
flags.DEFINE_float('penalty_aggression', 2.0,
                   'How aggressively to penalize correlated features.')
flags.DEFINE_bool('use_class_weights', False,
                  'Whether to apply inverse-frequency class weights to the loss.')
flags.DEFINE_bool('drift', False, 'Whether to apply drift to the dataset.')
flags.DEFINE_bool('run_all_scenarios', False,
                  'Run Aranyani on every drift scenario, collect results, and plot comparisons.')
flags.DEFINE_string('drift_scenario', None,
                    'Specific drift scenario name (e.g. abrupt_gender). Overrides --drift.')
flags.DEFINE_bool('save_model', False, 'Whether to save the trained model.')
flags.DEFINE_bool('load_model', False, 'Whether to load an existing model instead of training.')
flags.DEFINE_string('model_path', None,
                    'Path to save/load the model. If not specified, will use models/{dataset}_model.')
flags.DEFINE_bool('prequential', False,
                  'Use test-then-train (prequential) evaluation with drift reaction '
                  'instead of inference-only evaluation.')
flags.DEFINE_string('folktables_sensitive_attribute', 'sex',
                    'Sensitive attribute for folktables: sex or race.')
flags.DEFINE_string('folktables_states', 'CA',
                    'Comma-separated state codes for folktables (e.g., CA or CA,TX).')
flags.DEFINE_integer('folktables_train_year', 2015,
                     'Training survey year for folktables.')
flags.DEFINE_string('folktables_test_years', '2016,2017,2018',
                    'Comma-separated test survey years for folktables.')
flags.DEFINE_string('folktables_horizon', '1-Year',
                    'ACS horizon for folktables (e.g., 1-Year).')


FLAGS = flags.FLAGS
NSGA2_PREQ_CONFIG_PATH = 'files/nsga2_prequential_config.yaml'
NSGA2_TREE_CONFIG_PATH = 'files/nsga2_tree_config.yaml'


def _load_prequential_nsga2_config():
  cfg = {
      'enabled': False,
      'fairness_type': 'dp',
      'population_size': 8,
      'generations': 4,
      'sample_size': 800,
      'seed': 42,
      'target_drift_rate': 0.01,
      'static_params': {},
  }
  if not os.path.exists(NSGA2_PREQ_CONFIG_PATH):
    return cfg
  with open(NSGA2_PREQ_CONFIG_PATH) as f:
    loaded = yaml.safe_load(f) or {}
  if not isinstance(loaded, dict):
    return cfg
  section = loaded.get('nsga2_prequential', loaded)
  if not isinstance(section, dict):
    return cfg
  cfg.update(section)
  cfg['fairness_type'] = str(cfg.get('fairness_type', 'dp')).lower()
  if cfg['fairness_type'] not in ['dp', 'eo']:
    cfg['fairness_type'] = 'dp'

  static_params = cfg.get('static_params', {})
  if not isinstance(static_params, dict):
    static_params = {}

  int_keys = ['lr_decay_steps', 'fairness_window', 'cooldown', 'min_samples_per_stream']
  normalized_static_params = {}
  for key, value in static_params.items():
    if key in int_keys:
      normalized_static_params[key] = int(round(float(value)))
    else:
      normalized_static_params[key] = float(value)
  cfg['static_params'] = normalized_static_params
  return cfg


def _load_tree_nsga2_config():
  cfg = {
      'enabled': False,
      'population_size': 8,
      'generations': 4,
      'seed': 42,
      'train_sample_size': 2000,
      'val_sample_size': 1200,
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
  cfg['train_sample_size'] = int(cfg.get('train_sample_size', 2000))
  cfg['val_sample_size'] = int(cfg.get('val_sample_size', 1200))
  cfg['validation_ratio'] = float(cfg.get('validation_ratio', 0.2))

  depth_bounds = cfg.get('depth_bounds', [2, 6])
  num_trees_bounds = cfg.get('num_trees_bounds', [1, 7])
  lambda_bounds = cfg.get('lambda_bounds', [0.0, 5.0])
  if not isinstance(depth_bounds, list) or len(depth_bounds) != 2:
    depth_bounds = [2, 6]
  if not isinstance(num_trees_bounds, list) or len(num_trees_bounds) != 2:
    num_trees_bounds = [1, 7]
  if not isinstance(lambda_bounds, list) or len(lambda_bounds) != 2:
    lambda_bounds = [0.0, 5.0]

  cfg['depth_bounds'] = [int(depth_bounds[0]), int(depth_bounds[1])]
  cfg['num_trees_bounds'] = [int(num_trees_bounds[0]), int(num_trees_bounds[1])]
  cfg['lambda_bounds'] = [float(lambda_bounds[0]), float(lambda_bounds[1])]
  return cfg


def _build_forest_model(dataset, data_dim, num_class, depth, num_trees, activation,
                        compute_mode):
  if dataset in ['celeba']:
    return clip_forest.FairCLIPDecisionForest(
        num_trees=num_trees,
        data_dim=data_dim,
        tree_depth=depth,
        num_classes=num_class,
        activation=activation,
        compute_mode=compute_mode,
    )
  return forest.FairDecisionForest(
      num_trees=num_trees,
      data_dim=data_dim,
      tree_depth=depth,
      num_classes=num_class,
      activation=activation,
      compute_mode=compute_mode,
  )


def _split_train_validation(x_train, y_train, a_train, validation_ratio, seed):
  n = len(x_train)
  if n <= 5:
    return x_train, y_train, a_train, [], [], []
  val_n = max(1, int(round(float(validation_ratio) * n)))
  val_n = min(val_n, n - 1)
  x_arr = np.asarray(x_train)
  y_arr = np.asarray(y_train)
  a_arr = np.asarray(a_train)
  x_tr, x_val, y_tr, y_val, a_tr, a_val = train_test_split(
      x_arr, y_arr, a_arr,
      test_size=val_n,
      random_state=int(seed),
      shuffle=True,
  )
  return x_tr, y_tr, a_tr, x_val, y_val, a_val


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

  tune_train_n = min(len(x_train), int(tree_cfg['train_sample_size']))
  tune_val_n = min(len(x_val), int(tree_cfg['val_sample_size']))

  x_tune_train = np.asarray(x_train)[:tune_train_n]
  y_tune_train = np.asarray(y_train)[:tune_train_n]
  a_tune_train = np.asarray(a_train)[:tune_train_n]
  x_tune_val = np.asarray(x_val)[:tune_val_n]
  y_tune_val = np.asarray(y_val)[:tune_val_n]
  a_tune_val = np.asarray(a_val)[:tune_val_n]

  from src.helpers.nsga2_tuner import run_nsga2

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
    )
    return (1.0 - float(val_metrics['accuracy']), float(val_metrics['dp']), float(val_metrics['eo']))

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
  from src.helpers.nsga2_tuner import run_nsga2
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

def train(
    dataset='civil',
    lambda_const=1,
    max_iter=1,
    depth=4,
    num_trees=3,
    compute_fairness=True,
    batch_size=1,
    activation='sigmoid',
    compute_mode='default',
    base_gamma=None,
    constraint_type='node',
    gradient_type='vanilla',
    encoder_model='instructor',
    offline_loss_type='mmd',
    local_run=False,
    drift=False,
    drift_scenario=None,
    save_model=False,
    load_model=False,
    model_path=None,
    prequential=False,
    folktables_sensitive_attribute='sex',
    folktables_states='CA',
    folktables_train_year=2015,
    folktables_test_years='2016,2017,2018',
    folktables_horizon='1-Year',
):
  all_dps = []
  all_accuracies = []
  all_equalized_odds = []
  effective_base_gamma = float(base_gamma) if base_gamma is not None else 0.9

  data_dim, num_class = None, None
  x_test, y_test, a_test, number_of_attributes = None, None, None, None  # For test set evaluation
  folktables_states_list = [
      state.strip() for state in str(folktables_states).split(',') if state.strip()
  ] or ['CA']
  folktables_test_years_tuple = tuple(
      int(year.strip()) for year in str(folktables_test_years).split(',') if year.strip()
  )
  
  # Check if we should load an existing model
  if load_model:
    if model_path is None:
      model_path = f'src/models/{dataset}_model'
    
    model_file = f'{model_path}.pkl'
    if os.path.exists(model_file):
      print(f"Loading existing model from {model_file}...")
      # Determine if it's a CLIP model
      is_clip = dataset in ['celeba']
      try:
        if is_clip:
          loaded_model = clip_forest.FairCLIPDecisionForest.load(model_path)
        else:
          loaded_model = forest.FairDecisionForest.load(model_path)
      except (KeyError, TypeError, ValueError) as exc:
        print(
            f"Warning: Could not load {model_file} as an Aranyani forest checkpoint "
            f"({exc}). Proceeding with training instead."
        )
        loaded_model = None
      except AttributeError as exc:
        print(
            f"Warning: Incompatible checkpoint format in {model_file} ({exc}). "
            "Proceeding with training instead."
        )
        loaded_model = None

      if loaded_model is None:
        print(
            "Hint: this file may be a different model type (e.g., sklearn Pipeline). "
            "Save an Aranyani model with --save_model to enable --load_model."
        )
      else:
      
        # Still need to load data for evaluation
        if dataset == 'adult':
          data_dim = 14
          num_class = 2
          x_train, x_test, y_train, y_test, a_train, a_test = data.read_adult(
              drift, drift_scenario=drift_scenario
          )
          assert len(set(a_train)) == 2, "Expected binary sensitive attribute for adult dataset"
          print(f"Sensitive attribute values in training set: {set(a_train)}")
        elif dataset == 'census':
          data_dim = 40
          num_class = 2
          x_train, x_test, y_train, y_test, a_train, a_test = data.read_census()
        elif dataset == 'compas':
          data_dim = 10
          x_train, y_train, a_train = data.read_compas()
          num_class = 2
        elif dataset == 'jigsaw':
          data_dim = 768
          x_train, y_train, a_train = data.read_jigsaw()
          num_class = 2
        elif dataset == 'celeba':
          data_dim = 768
          x_train, y_train, a_train = data.read_celeba()
          num_class = 2
        elif dataset == 'folktables':
          x_train, x_test, y_train, y_test, a_train, a_test, number_of_attributes = data.read_folktables(
              train_year=folktables_train_year,
              test_years=folktables_test_years_tuple,
              state=folktables_states_list,
              horizon=folktables_horizon,
              sensitive_attribute=folktables_sensitive_attribute,
          )
          data_dim = x_train.shape[1]
          num_class = 2
          a_train = [a-1 for a in a_train]
          a_test = [a-1 for a in a_test]
        
        # Evaluate the loaded model
        if x_test is not None and len(x_test) > 0:
          test_metrics = None
          if not prequential:
            test_metrics = utils.get_test_performance(
                loaded_model, x_test, y_test, a_test,
                data_dim=data_dim,
                show_confusion_matrix=True,
            )
            print(f"\n{'='*80}")
            print("Loaded Model Test Results:")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Test DP: {test_metrics['dp']:.4f}")
            print(f"  Test EO: {test_metrics['eo']:.4f}")
            print(f"  Test AUC: {test_metrics['auc']:.4f}")
            print(f"  Test F1-Score: {test_metrics['f1']:.4f}")
            print(f"{'='*80}\n")
          if prequential:
              preq_cfg = _load_prequential_nsga2_config()
              tuned_static_params = _tune_prequential_static_params(
                  loaded_model, x_train, y_train, a_train, data_dim,
                  compute_fairness, lambda_const, depth, num_trees,
                  constraint_type, gradient_type, effective_base_gamma, preq_cfg,
              )
              static_params_to_use = tuned_static_params or preq_cfg.get('static_params') or None
              print("\nRunning prequential (test-then-train) evaluation...")
              # Reload a fresh copy so baseline and prequential start from
              # the same weights and we can compare fairly.
              if is_clip:
                preq_model = clip_forest.FairCLIPDecisionForest.load(model_path)
              else:
                preq_model = forest.FairDecisionForest.load(model_path)
              preq_results = utils.evaluate_over_timesteps(
                  preq_model, x_test, y_test, a_test, data_dim=data_dim,
                  test_then_train=True,
                  compute_fairness=compute_fairness,
                  fairness_type=preq_cfg['fairness_type'],
                  lambda_const=lambda_const,
                  tree_depth=depth,
                  num_trees=num_trees,
                  constraint_type=constraint_type,
                  gradient_type=gradient_type,
                  base_gamma=effective_base_gamma,
                  static_params=static_params_to_use,
              )
              plot_metrics_over_timesteps(preq_results,
                                          save_path='files/metrics_prequential.png')
              final_acc = float(preq_results['accuracy'][-1]) if preq_results['accuracy'] else 0.0
              final_dp = float(preq_results['dp'][-1]) if preq_results['dp'] else 0.0
              print(
                  f"[Prequential] Final streamed test accuracy: {100.0 * final_acc:.2f}% | "
                  f"Final DP: {100.0 * final_dp:.2f}%"
              )
              # Return prequential results as the primary timestep_results
              timestep_results = preq_results
          else:
              timestep_results = None

          return [], [], [], timestep_results, test_metrics, loaded_model, data_dim
        else:
          return [], [], [], None, None, loaded_model, data_dim
    else:
      print(f"Warning: Model file {model_file} not found. Proceeding with training...")
  
  if dataset == 'adult':
    data_dim = 14
    num_class = 2
    x_train, x_test, y_train, y_test, a_train, a_test = data.read_adult(
        drift, drift_scenario=drift_scenario
    )
    print(f"Sensitive attribute values in training set: {set(a_train)}")
    assert len(set(a_train)) == 2, "Expected binary sensitive attribute for adult dataset"
  elif dataset == 'census':
    data_dim = 40
    num_class = 2
    x_train, x_test, y_train, y_test, a_train, a_test = data.read_census()
  elif dataset == 'compas':
    data_dim = 10
    x_train, y_train, a_train = data.read_compas()
    num_class = 2
  elif dataset == 'jigsaw':
    data_dim = 768
    x_train, y_train, a_train = data.read_jigsaw()
    num_class = 2
  elif dataset == 'celeba':
    data_dim = 768
    x_train, y_train, a_train = data.read_celeba()
    num_class = 2
  elif dataset == 'folktables':
    x_train, x_test, y_train, y_test, a_train, a_test, number_of_attributes = data.read_folktables(
        train_year=folktables_train_year,
        test_years=folktables_test_years_tuple,
        state=folktables_states_list,
        horizon=folktables_horizon,
        sensitive_attribute=folktables_sensitive_attribute,
    )
    data_dim = x_train.shape[1]
    num_class = 2
    sensitive = str(folktables_sensitive_attribute).lower()
    print(f"Sensitive attribute values in training set: {set(a_train)}")
    if sensitive == 'race':
      assert len(set(a_train)) == 2, (
          "Expected binary sensitive attribute for folktables race setup "
          "(white vs non-white)"
      )
    else:
      assert len(set(a_train)) == 2, "Expected binary sensitive attribute for folktables sex setup"
  else:
    x_train, y_train, a_train = [], [], []

  if data_dim is None or num_class is None:
    raise ValueError(f'Unsupported or misconfigured dataset: {dataset}')

  tree_cfg = _load_tree_nsga2_config()
  if bool(tree_cfg['enabled']) and len(x_train) > 1:
    split_seed = int(tree_cfg['seed'])
    x_inner_train, y_inner_train, a_inner_train, x_val, y_val, a_val = _split_train_validation(
        x_train, y_train, a_train, tree_cfg['validation_ratio'], split_seed
    )
    tuned_tree = _tune_tree_hyperparameters_nsga2(
        dataset=dataset,
        x_train=x_inner_train,
        y_train=y_inner_train,
        a_train=a_inner_train,
        x_val=x_val,
        y_val=y_val,
        a_val=a_val,
        data_dim=data_dim,
        num_class=num_class,
        base_depth=depth,
        base_num_trees=num_trees,
        compute_fairness=compute_fairness,
        base_lambda_const=lambda_const,
        batch_size=batch_size,
        activation=activation,
        compute_mode=compute_mode,
        base_gamma=effective_base_gamma,
        constraint_type=constraint_type,
        gradient_type=gradient_type,
        local_run=local_run,
        tree_cfg=tree_cfg,
    )
    if tuned_tree is not None:
      depth = int(tuned_tree['depth'])
      num_trees = int(tuned_tree['num_trees'])
      lambda_const = float(tuned_tree['lambda_const'])
      print(
          f"[NSGA2-TREE] Applying tuned settings for final training: "
          f"depth={depth}, num_trees={num_trees}, lambda_const={lambda_const:.4f}"
      )


  print(f'DP in the original dataset: {utils.get_demographic_parity(y_train, a_train)[0]}')
  print(f"EO in the original dataset: {utils.get_equalized_odds(y_train, a_train, y_train)[0]}")
  
  trained_model = None
  for _ in range(max_iter):
    model = _build_forest_model(
      dataset=dataset,
      data_dim=data_dim,
      num_class=num_class,
      depth=depth,
      num_trees=num_trees,
      activation=activation,
      compute_mode=compute_mode,
    )

    dp, eo, accuracies, average_w_fair_grad, average_b_fair_grad = aranyani.train_online(
        model,
        x_train,
        y_train,
        a_train,
        data_dim=data_dim,
        batch_size=batch_size,
        tree_depth=depth,
        compute_fairness=compute_fairness,
        lambda_const=lambda_const,
        num_trees=num_trees,
        base_gamma=effective_base_gamma,
        constraint_type=constraint_type,
        gradient_type=gradient_type,
        local_run=local_run,
    )

    all_dps.append(dp)
    all_accuracies.append(accuracies)
    all_equalized_odds.append(eo)
    trained_model = model

  has_test_set = x_test is not None and len(x_test) > 0
  test_metrics = None
  if has_test_set and not prequential and trained_model is not None:
    test_metrics = utils.get_test_performance(
        trained_model, x_test, y_test, a_test,
        data_dim=data_dim, show_confusion_matrix=True,
    )
    print(f"\n{'='*80}")
    print("Test Set Results:")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test DP: {test_metrics['dp']:.4f}")
    print(f"  Test AUC: {test_metrics['auc']:.4f}")
    print(f"  Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"  Test EO: {test_metrics['eo']:.4f}")
    print(f"{'='*80}\n")
  elif has_test_set and prequential:
    print("\nPrequential mode enabled with test set: skipping one-shot test metric aggregation.")


  # Evaluate metrics over timesteps on drifted test set (single-scenario mode)
  timestep_results = None
  if drift and x_test is not None and len(x_test) > 0:
    if prequential:
      preq_cfg = _load_prequential_nsga2_config()
      tuned_static_params = _tune_prequential_static_params(
          trained_model, x_train, y_train, a_train, data_dim,
          compute_fairness, lambda_const, depth, num_trees,
            constraint_type, gradient_type, effective_base_gamma, preq_cfg,
      )
      static_params_to_use = tuned_static_params or preq_cfg.get('static_params') or None
      print("\nRunning prequential (test-then-train) evaluation...")
      preq_model = copy.deepcopy(trained_model)
      preq_results = utils.evaluate_over_timesteps(
          preq_model, x_test, y_test, a_test, data_dim=data_dim,
          test_then_train=True,
          compute_fairness=compute_fairness,
          fairness_type=preq_cfg['fairness_type'],
          lambda_const=lambda_const,
          tree_depth=depth,
          num_trees=num_trees,
          constraint_type=constraint_type,
          gradient_type=gradient_type,
          base_gamma=effective_base_gamma,
          static_params=static_params_to_use,
      )
      plot_metrics_over_timesteps(preq_results,
                                  save_path='files/metrics_prequential.png')
      final_acc = float(preq_results['accuracy'][-1]) if preq_results['accuracy'] else 0.0
      final_dp = float(preq_results['dp'][-1]) if preq_results['dp'] else 0.0
      print(
          f"[Prequential] Final streamed test accuracy: {100.0 * final_acc:.2f}% | "
          f"Final DP: {100.0 * final_dp:.2f}%"
      )
      timestep_results = preq_results

  # Save model if requested
  if save_model and trained_model is not None:
    if model_path is None:
      model_path = f'models/{dataset}_model'
    trained_model.save(model_path)

  return all_dps, all_accuracies, all_equalized_odds, timestep_results, test_metrics, trained_model, data_dim
