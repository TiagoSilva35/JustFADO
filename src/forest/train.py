"""Driver code."""


import os
import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from absl import flags

import src.helpers.data as data
import src.forest.forest as forest
import src.forest.aranyani as aranyani

from sklearn.model_selection import TimeSeriesSplit
from src.helpers.plots import plot_metric_over_iterations, plot_metrics_over_timesteps

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
flags.DEFINE_bool('use_test_set', False,
                  'Whether to use actual test set instead of cross-validation (for adult dataset).')
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
  return cfg

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
    use_test_set=True,
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
  x_test, y_test, a_test = None, None, None  # For test set evaluation
  
  # Check if we should load an existing model
  if load_model:
    if model_path is None:
      model_path = f'src/models/{dataset}_model'
    
    model_file = f'{model_path}.pkl'
    if os.path.exists(model_file):
      print(f"Loading existing model from {model_file}...")
      # Determine if it's a CLIP model
      is_clip = dataset in ['celeba']
      if is_clip:
        loaded_model = clip_forest.FairCLIPDecisionForest.load(model_path)
      else:
        loaded_model = forest.FairDecisionForest.load(model_path)
      
      # Still need to load data for evaluation
      if dataset == 'adult':
        data_dim = 14
        num_class = 2
        x_train, x_test, y_train, y_test, a_train, a_test = data.read_adult(
            drift, drift_scenario=drift_scenario
        )
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
      
      # Evaluate the loaded model
      if x_test is not None and len(x_test) > 0:
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
            print("\nRunning prequential (test-then-train) evaluation...")
            # Reload a fresh copy so baseline and prequential start from
            # the same weights and we can compare fairly.
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
                enable_nsga2_tuning=bool(preq_cfg['enabled']),
                nsga2_config={
                    'population_size': int(preq_cfg['population_size']),
                    'generations': int(preq_cfg['generations']),
                    'sample_size': int(preq_cfg['sample_size']),
                    'seed': int(preq_cfg['seed']),
                    'target_drift_rate': float(preq_cfg['target_drift_rate']),
                },
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
  else:
    x_train, y_train, a_train = [], [], []


  print(f'DP in the original dataset: {utils.get_demographic_parity(y_train, a_train)[0]}')
  print(f"EO in the original dataset: {utils.get_equalized_odds(y_train, a_train, y_train)[0]}")
  
  # Decide whether to use test set or cross-validation
  use_actual_test_set = use_test_set and dataset == 'adult' and len(x_test) > 0
  
  if use_actual_test_set:
    splits = [(range(len(x_train)), None)] 
    fold_results = []
  else:
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(x_train))
    fold_results = []
  
  for fold_idx, (train_idx, val_idx) in enumerate(splits):
    if use_actual_test_set:
      x_train_fold = x_train
      y_train_fold = y_train
      a_train_fold = a_train
      x_val_fold = x_test
      y_val_fold = y_test
      a_val_fold = a_test
    else:
      x_train_fold = x_train[train_idx]
      y_train_fold = y_train[train_idx]
      a_train_fold = a_train[train_idx]
      x_val_fold = x_train[val_idx]
      y_val_fold = y_train[val_idx]
      a_val_fold = a_train[val_idx]
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
          x_train_fold,
          y_train_fold,
          a_train_fold,
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

    val_metrics = utils.get_test_performance(
        model, x_val_fold, y_val_fold, a_val_fold,
        data_dim=data_dim, show_confusion_matrix=True,
    )

    fold_results.append({
        'fold': fold_idx + 1,
        'train_dp': dp,
        'train_accuracies': accuracies,
        'train_eo': eo,
        'val_metrics': val_metrics,
        'model': model,
    })

  if use_actual_test_set:
    # For test set evaluation, just report the single result
    print(f"\n{'='*80}")
    print("Test Set Results:")
    test_result = fold_results[0]['val_metrics']
    print(f"  Test Accuracy: {test_result['accuracy']:.4f}")
    print(f"  Test DP: {test_result['dp']:.4f}")
    print(f"  Test AUC: {test_result['auc']:.4f}")
    print(f"  Test Sensitivity: {test_result['sensitivity']:.4f}")
    print(f"  Test F1-Score: {test_result['f1']:.4f}")
    print(f"  Test EO: {test_result['eo']:.4f}")
    print(f"{'='*80}\n")
  else:
    avg_metrics = utils.aggregate_fold_results(fold_results)

    print(f"\n{'='*80}")
    print("Cross-Validation Results:")
    print(f"  Mean Val Accuracy: {avg_metrics['mean_val_accuracy']:.4f} +/- {avg_metrics['std_val_accuracy']:.4f}")
    print(f"  Mean Val DP: {avg_metrics['mean_val_dp']:.4f} +/- {avg_metrics['std_val_dp']:.4f}")
    print(f"  Mean Val AUC: {avg_metrics['mean_val_auc']:.4f} +/- {avg_metrics['std_val_auc']:.4f}")
    print(f"  Mean Val Sensitivity: {avg_metrics['mean_val_sensitivity']:.4f} +/- {avg_metrics['std_val_sensitivity']:.4f}")
    print(f"  Mean Val F1-Score: {avg_metrics['mean_val_f1']:.4f} +/- {avg_metrics['std_val_f1']:.4f}")
    print(f"  Mean Val EO: {avg_metrics['mean_val_eo']:.4f} +/- {avg_metrics['std_val_eo']:.4f}")
    print(f"{'='*80}\n")


  # Evaluate metrics over timesteps on drifted test set (single-scenario mode)
  timestep_results = None
  test_metrics = None
  if drift and x_test is not None and len(x_test) > 0:
    if prequential:
      preq_cfg = _load_prequential_nsga2_config()
      print("\nRunning prequential (test-then-train) evaluation...")
      import copy
      preq_model = copy.deepcopy(fold_results[0]['model'])
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
          enable_nsga2_tuning=bool(preq_cfg['enabled']),
          nsga2_config={
              'population_size': int(preq_cfg['population_size']),
              'generations': int(preq_cfg['generations']),
              'sample_size': int(preq_cfg['sample_size']),
              'seed': int(preq_cfg['seed']),
              'target_drift_rate': float(preq_cfg['target_drift_rate']),
          },
      )
      plot_metrics_over_timesteps(preq_results,
                                  save_path='files/metrics_prequential.png')
      timestep_results = preq_results

  if fold_results and fold_results[0].get('val_metrics'):
    test_metrics = fold_results[0]['val_metrics']

  # Return the trained model so callers can reuse it for multiple evaluations
  trained_model = fold_results[0]['model'] if fold_results else None
  
  # Save model if requested
  if save_model and trained_model is not None:
    if model_path is None:
      model_path = f'models/{dataset}_model'
    trained_model.save(model_path)

  return all_dps, all_accuracies, all_equalized_odds, timestep_results, test_metrics, trained_model, data_dim
