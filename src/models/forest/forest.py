"""Fair classification decision tree."""

import tensorflow as tf
import pickle
import os

import src.models.forest.fdt as fdt


class FairDecisionForest(tf.Module):
  """Fair classification decision tree."""

  def __init__(self,
               num_trees,
               data_dim,
               tree_depth,
               num_classes,
               activation='sigmoid',
               compute_mode='default',
               use_recursive_updater=None):
    super(FairDecisionForest, self).__init__()
    assert tree_depth > 1
    assert num_trees >= 1
    self.use_recursive_updater = (
        fdt.RECURSIVE_UPDATER if use_recursive_updater is None
        else bool(use_recursive_updater)
    )
    self.layers = []
    for _ in range(num_trees):
      self.layers.append(fdt.FairDecisionTree(
          data_dim, tree_depth, num_classes, activation, compute_mode,
          use_recursive_updater=self.use_recursive_updater))

  def __call__(self, inputs, training=False):
    all_predictions = []
    all_node_decisions = []
    for layer in self.layers:
      if training:
        # During training, we want to collect predictions and node decisions
        prediction, node_decisions, _ = layer(inputs, training=training)
        all_predictions.append(prediction)
        all_node_decisions.append(node_decisions)
      else:
        # During inference, we only want the final prediction
        prediction = layer(inputs, training=training)
        all_predictions.append(prediction)

    stacked_predictions = tf.stack(all_predictions, axis=0)  # [num_trees, batch_size, num_classes]
    final_prediction = tf.reduce_mean(stacked_predictions, axis=0)

    if training:
      all_node_decisions = tf.stack(all_node_decisions, axis=0)
      return final_prediction, all_node_decisions, stacked_predictions
    
    return final_prediction
  
  def predict_per_tree(self, inputs):
    all_predictions = []
    for layer in self.layers:
      prediction = layer(inputs, training=False)
      all_predictions.append(prediction)
    stacked_predictions = tf.stack(all_predictions, axis=0)
    final_prediction = tf.reduce_mean(stacked_predictions, axis=0)
    return stacked_predictions, final_prediction

  def reset_tree(self, tree_id):
    tree = self.layers[tree_id]
    num_leaves  = tree.theta.shape[0]
    num_classes = tree.theta.shape[1]

    tree.theta.assign(tf.random.uniform([num_leaves, num_classes]))

  def save(self, filepath):
    """Save the model weights and configuration to a file.
    
    Args:
      filepath: Path where to save the model (without extension).
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save configuration and weights
    model_data = {
        'num_trees': len(self.layers),
        'weights': [],
    }
    
    for tree in self.layers:
      tree_data = {
          'weight': tree.weight.numpy(),
          'bias': tree.bias.numpy(),
          'theta': tree.theta.numpy(),
          'data_dim': tree.weight.shape[0],
          'tree_depth': int(tf.math.log(float(tree.num_leaves)) / tf.math.log(2.0)),
          'num_classes': tree.theta.shape[1],
          'activation': tree.activation.__name__,
          'compute_mode': tree.compute_mode,
          'use_recursive_updater': bool(tree.use_recursive_updater),
      }
      model_data['weights'].append(tree_data)
    
    with open(f'{filepath}.pkl', 'wb') as f:
      pickle.dump(model_data, f)
    
    print(f"Model saved to {filepath}.pkl")
  
  @classmethod
  def load(cls, filepath):
    """Load a saved model from a file.
    
    Args:
      filepath: Path to the saved model file (without .pkl extension).
      
    Returns:
      A FairDecisionForest instance with loaded weights.
    """
    with open(f'{filepath}.pkl', 'rb') as f:
      model_data = pickle.load(f)
    
    # Get configuration from first tree
    first_tree = model_data['weights'][0]
    
    # Create new model instance
    model = cls(
        num_trees=model_data['num_trees'],
        data_dim=first_tree['data_dim'],
        tree_depth=first_tree['tree_depth'],
        num_classes=first_tree['num_classes'],
        activation=first_tree['activation'],
        compute_mode=first_tree['compute_mode'],
        # Older checkpoints predate the per-model code-path switch; they were
        # produced on the recursive path, which is also the module default.
        use_recursive_updater=first_tree.get('use_recursive_updater', None),
    )
    
    # Load weights for each tree
    for i, tree_data in enumerate(model_data['weights']):
      model.layers[i].weight.assign(tree_data['weight'])
      model.layers[i].bias.assign(tree_data['bias'])
      model.layers[i].theta.assign(tree_data['theta'])
    
    print(f"Model loaded from {filepath}.pkl")
    return model