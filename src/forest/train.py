"""Driver code."""


import os
import yaml

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


FLAGS = flags.FLAGS
NSGA2_PREQ_CONFIG_PATH = 'files/nsga2_prequential_config.yaml'


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
  import copy
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
):
  all_dps = []
  all_accuracies = []
  all_equalized_odds = []

  data_dim, num_class = None, None
  x_test, y_test, a_test, number_of_attributes = None, None, None, None  # For test set evaluation
  
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
          x_train, x_test, y_train, y_test, a_train, a_test, number_of_attributes = data.read_folktables()
          data_dim = x_train.shape[1]
          num_class = 2
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
                  constraint_type, gradient_type, base_gamma, preq_cfg,
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
                  base_gamma=base_gamma,
                  static_params=static_params_to_use,
              )
              plot_metrics_over_timesteps(preq_results,
                                          save_path='files/metrics_prequential.png')
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
    x_train, x_test, y_train, y_test, a_train, a_test, number_of_attributes = data.read_folktables()
    # for each a in a train subtract 1 to get 0-indexed groups
    a_train = [a - 1 for a in a_train]
    a_test = [a - 1 for a in a_test]
    data_dim = x_train.shape[1]
    num_class = 2
    print(f"Sensitive attribute values in training set: {set(a_train)}")
    assert len(set(a_train)) == 9, "Expected 9 unique values in the sensitive attribute for folktables dataset"
  else:
    x_train, y_train, a_train = [], [], []


  print(f'DP in the original dataset: {utils.get_demographic_parity(y_train, a_train)[0]}')
  print(f"EO in the original dataset: {utils.get_equalized_odds(y_train, a_train, y_train)[0]}")
  
  trained_model = None
  for _ in range(max_iter):
    model = forest.FairDecisionForest(
        num_trees=num_trees,
        data_dim=data_dim,
        tree_depth=depth,
        num_classes=num_class,
        activation=activation,
        compute_mode=compute_mode,
    )
    if dataset in ['celeba']:
      model = clip_forest.FairCLIPDecisionForest(
          num_trees=num_trees,
          data_dim=data_dim,
          tree_depth=depth,
          num_classes=num_class,
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
        base_gamma=base_gamma,
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
          constraint_type, gradient_type, base_gamma, preq_cfg,
      )
      static_params_to_use = tuned_static_params or preq_cfg.get('static_params') or None
      print("\nRunning prequential (test-then-train) evaluation...")
      import copy
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
          base_gamma=base_gamma,
            static_params=static_params_to_use,
      )
      plot_metrics_over_timesteps(preq_results,
                                  save_path='files/metrics_prequential.png')
      timestep_results = preq_results

  # Save model if requested
  if save_model and trained_model is not None:
    if model_path is None:
      model_path = f'models/{dataset}_model'
    trained_model.save(model_path)

  return all_dps, all_accuracies, all_equalized_odds, timestep_results, test_metrics, trained_model, data_dim
