"""Fair classification decision tree."""

import tensorflow as tf
import pickle
import os

import src.forest.fdt as fdt


class FairDecisionForest(tf.Module):
  """Fair classification decision tree."""

  def __init__(self,
               num_trees,
               data_dim,
               tree_depth,
               num_classes,
               activation='sigmoid',
               compute_mode='default'):
    """Constructor.

    Args:
      num_trees: number of trees in the forest.
      data_dim: dimension of the data. 
      tree_depth: depth of the binary tree. 
      num_classes: number of target task classes.
      activation: activation function.
      compute_mode: log or default.
    """

    super(FairDecisionForest, self).__init__()
    assert tree_depth > 1
    assert num_trees >= 1
    self.layers = []
    for _ in range(num_trees):
      self.layers.append(fdt.FairDecisionTree(
          data_dim, tree_depth, num_classes, activation, compute_mode))

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

    final_prediction = tf.reduce_mean(tf.stack(all_predictions), axis=0)

    if training:
      all_node_decisions = tf.stack(all_node_decisions, axis=0)
      return final_prediction, all_node_decisions
    
    return final_prediction
  
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
    )
    
    # Load weights for each tree
    for i, tree_data in enumerate(model_data['weights']):
      model.layers[i].weight.assign(tree_data['weight'])
      model.layers[i].bias.assign(tree_data['bias'])
      model.layers[i].theta.assign(tree_data['theta'])
    
    print(f"Model loaded from {filepath}.pkl")
    return model