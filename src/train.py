"""Driver code."""


import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from absl import flags

import data
import mlp

import forest
import majority
import mlp_trainer
import aranyani

import reservoir
import hoeffding_tree

from skmultiflow.trees import HoeffdingTree
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from plots import plot_metric_over_iterations

import utils
import clip_forest

from drift.create_drifted_ds import generate_drifted_dataset


flags.DEFINE_string('sweep_id', '-1', 'Wandb sweep ID.')
flags.DEFINE_float('lambda_const', 0.1, 'Cell to run.')
flags.DEFINE_string('dataset', 'civil', 'Dataset name.')
flags.DEFINE_string('mode', 'node', 'Loss mode.')
flags.DEFINE_integer('max_iter', 1, 'Total number of iterations.')
flags.DEFINE_integer('depth', 4, 'Tree depth.')
flags.DEFINE_integer('num_trees', 3, 'Number of trees.')
flags.DEFINE_bool(
    'compute_fairness', True, 'Whether to apply fairness constraints.'
)
flags.DEFINE_integer('batch_size', 1, 'Samples in an online batch.')
flags.DEFINE_string('activation', 'sigmoid', 'Activation function.')
flags.DEFINE_string('model_type', 'forest', 'Type of f(x).')
flags.DEFINE_string('compute_mode', 'default', 'log or default.')
flags.DEFINE_string('base_gamma', None, 'gamma for the gradients.')
flags.DEFINE_string('constraint_type', 'node', '[node, leaf]')
flags.DEFINE_string('gradient_type', 'vanilla', '[vanilla, momentum, ema]')
flags.DEFINE_float('probability', 0.5,
                   'Probability of selection in majority baseline.')
flags.DEFINE_string('encoder_model', 'instructor', '[bert, instructor]')
flags.DEFINE_integer('reservoir_size', 100, 'size of reservoir.')
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


FLAGS = flags.FLAGS

def train(
    mode='node',
    dataset='civil',
    lambda_const=1,
    model_type='forest',
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
    probability=0.5,
    encoder_model='instructor',
    reservoir_size=100,
    offline_loss_type='mmd',
    local_run=False,
    use_test_set=True,
    drift=False,
):

  all_dps = []
  all_accuracies = []
  all_equalized_odds = []

  data_dim, num_class = None, None
  x_test, y_test, a_test = None, None, None  # For test set evaluation
  
  if dataset == 'adult':
    data_dim = 14
    num_class = 2
    x_train, x_test, y_train, y_test, a_train, a_test = data.read_adult(drift)
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
      model = None
      if model_type == 'mlp':
        model = mlp.FairReLUNetwork(
            data_dim=data_dim,
            tree_depth=depth,
            num_classes=num_class,
            activation='relu',
            use_layer_norm=True,
            dropout_rate=0.0,
            num_layers=2,
            hidden_multiplier=2,
        )
      elif model_type == 'ht':
        model = HoeffdingTree()
      elif model_type == 'aht':
        model = HoeffdingAdaptiveTreeClassifier()
      elif model_type == 'forest':
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

      if mode == 'node' and model_type == 'forest':
        train_func = aranyani.train_online
        use_correlation_penalty = False
        dp, eo, accuracies, average_w_fair_grad, average_b_fair_grad = train_func(
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
        print(f"fold {fold_idx + 1} here are the average w fairness gradients: {average_w_fair_grad}")
        print(f"fold {fold_idx + 1} here are the average b fairness gradients: {average_b_fair_grad}")
      elif mode == 'majority':
        train_func = majority.train_online
        dp, accuracies = train_func(
            model,
            x_train_fold,
            y_train_fold,
            a_train_fold,
            batch_size=batch_size,
            probability=probability,
            local_run=local_run,
        )
      elif model_type == 'mlp':
        dp, accuracies = mlp_trainer.train_online(
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
      elif model_type in ['ht', 'aht']:
        dp, accuracies = hoeffding_tree.train_online(
            model,
            x_train_fold,
            y_train_fold,
            a_train_fold,
            batch_size=batch_size,
            local_run=local_run,
            label_type='categorical',
        )
      elif mode == 'reservoir':
        dp, accuracies = reservoir.train_online(
            model,
            x_train_fold,
            y_train_fold,
            a_train_fold,
            batch_size=batch_size,
            tree_depth=depth,
            compute_fairness=compute_fairness,
            lambda_const=lambda_const,
            reservoir_size=reservoir_size,
            local_run=local_run,
        )
      else:
        dp, accuracies, eo = [], [], []

      
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
    })

    del model

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

  plot_metric_over_iterations(all_accuracies, 'Accuracy', model_type, 'dp' if compute_fairness else 'none', constraint_type)
  plot_metric_over_iterations(all_dps, 'Demographic Parity', model_type, 'dp' if compute_fairness else 'none', constraint_type)
  plot_metric_over_iterations(all_equalized_odds, 'Equalized Odds', model_type, 'eo' if compute_fairness else 'none', constraint_type)
  
  return all_dps, all_accuracies, all_equalized_odds
