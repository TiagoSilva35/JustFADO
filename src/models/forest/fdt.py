"""Fair classification decision tree."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import logging

logging.basicConfig(level=logging.INFO)

LEAF_PROBABILITY_MODES = ('auto', 'recursive', 'mask')
DEFAULT_LEAF_PROBABILITY = 'auto'

# Depth at or above which the recursive path (O(B x 2^n)) beats the mask path
# (O(B x 4^n)). The asymptotics are about FLOPs, but in eager mode at batch 1
# the runtime is dominated by the NUMBER of dispatched ops: the mask path
# issues a constant handful regardless of depth, the recursive path ~4 per
# level. Measured at batch 1 (TESTS/rec_leaf_prob.py): mask wins up to depth 8,
# recursive takes over around 9-10. Hardware-dependent -- re-measure and update.
RECURSIVE_UPDATER_MIN_DEPTH = 9

# Legacy boolean alias for leaf_probability (True -> 'recursive').
RECURSIVE_UPDATER = True


def resolve_leaf_probability(mode, tree_depth):
  """Resolve a requested mode to 'recursive' or 'mask'."""
  if mode is None:
    mode = DEFAULT_LEAF_PROBABILITY
  if isinstance(mode, bool):
    return 'recursive' if mode else 'mask'
  mode = str(mode).lower()
  if mode not in LEAF_PROBABILITY_MODES:
    raise ValueError(f"leaf_probability must be one of {LEAF_PROBABILITY_MODES}, got {mode!r}")
  if mode == 'auto':
    return 'recursive' if int(tree_depth) >= RECURSIVE_UPDATER_MIN_DEPTH else 'mask'
  return mode

def construct_mask_matrix(tree_depth=3):
  """Construct mask matrix for efficient DT training."""

  num_internal_nodes = 2**tree_depth - 1
  num_leaves = 2**tree_depth

  mask_matrix = np.zeros([num_internal_nodes, num_leaves])

  # iterate for the leaves
  for idx, leaf_index in enumerate(
      range(2**tree_depth, 2 ** (tree_depth + 1))
  ):
    # iterate over ancestors
    ancestor = leaf_index
    last_ancestor = ancestor
    while ancestor > 1:
      ancestor = ancestor // 2
      mask_matrix[ancestor - 1, idx] = (
          1 if 2 * ancestor == last_ancestor else -1
      )
      last_ancestor = ancestor
  return mask_matrix


class FairDecisionTree(tf.Module):
  """Fair classification decision tree."""

  def __init__(self,
               data_dim,
               tree_depth,
               num_classes,
               activation='sigmoid',
               compute_mode='log',
               leaf_probability=None,
               use_recursive_updater=None):

    super(FairDecisionTree, self).__init__()
    assert tree_depth > 1
    self.num_internal_nodes = 2**tree_depth - 1
    self.tree_depth = tree_depth

    # internal node parameters
    self.weight = tf.Variable(
        tf.random.normal([data_dim, self.num_internal_nodes]), 
        name='W',
        trainable=True,
    )

    self.bias = tf.Variable(
        tf.random.normal([self.num_internal_nodes]),
        name='B',
        trainable=True,
    )

    if activation == 'sigmoid':
      self.activation = tf.keras.activations.sigmoid
    elif activation == 'smoother':
      self.activation = tfp.math.smootherstep
    elif activation == 'relu':
      self.activation = tf.keras.activations.relu
    elif activation == 'gelu':
      self.activation = tf.keras.activations.gelu

    # leaf parameters
    self.num_leaves = 2**tree_depth
    self.theta = tf.Variable(
        tf.random.uniform([self.num_leaves, num_classes]), 
        name='theta',
        trainable=True,
    )

    # mask parameters
    mask_matrix = construct_mask_matrix(tree_depth=tree_depth)
    self.mask_matrix = tf.constant(mask_matrix, 
                                   dtype=tf.float32,
                                   name='mask_matrix')
    self.ones_nodes = tf.constant(
        tf.nn.relu(-self.mask_matrix), 
        dtype=tf.float32,
        name='ones'
    )
    self.mask = tf.constant(
        tf.ones_like(self.mask_matrix) - tf.math.abs(self.mask_matrix),
        dtype=tf.float32,
        name='mask'
    )
    self.temperature = tf.Variable(
        1.0, 
        name='temperature',
        trainable=False,
        dtype=tf.float32
    )
    self.compute_mode = compute_mode    
    requested = leaf_probability if leaf_probability is not None else use_recursive_updater
    self.leaf_probability_mode = (
        DEFAULT_LEAF_PROBABILITY if requested is None else requested
    )
    self.leaf_probability = resolve_leaf_probability(requested, tree_depth)
    # Resolved boolean, kept for call sites and checkpoints that speak in
    # terms of use_recursive_updater.
    self.use_recursive_updater = (self.leaf_probability == 'recursive')

  def set_leaf_probability(self, mode):
    """Switch the leaf-probability path on an existing tree."""
    self.leaf_probability_mode = mode
    self.leaf_probability = resolve_leaf_probability(mode, self.tree_depth)
    self.use_recursive_updater = (self.leaf_probability == 'recursive')

  def _recursive_updater_(self, node_decisions):
    """Leaf probabilities by level-wise propagation.

    Walks one LEVEL at a time rather than one leaf at a time, so each partial
    path probability is computed once and reused by both children. Replaces a
    version that built a Python list of 2^depth scalar products per call (~100x
    slower at depth 10) and indexed node_decisions[0], silently dropping every
    sample of a batch but the first.

    Args:
      node_decisions: [batch, num_internal_nodes].
    Returns:
      [batch, num_leaves], each row summing to 1.
    """
    batch = tf.shape(node_decisions)[0]
    probs = tf.ones([batch, 1], dtype=node_decisions.dtype)
    for level in range(self.tree_depth):
      start = 2 ** level - 1
      end = 2 ** (level + 1) - 1
      decisions = node_decisions[:, start:end]
      probs = tf.reshape(
          tf.stack([probs * decisions, probs * (1.0 - decisions)], axis=-1),
          [batch, -1],
      )
    return probs
  
  def __call__(self, inputs, training=False, pred_type='categorical'):
    logits = (tf.matmul(inputs, self.weight) + self.bias) / self.temperature
    raw_node_decisions = self.activation(logits)
    if self.use_recursive_updater:
      leaf_probs = self._recursive_updater_(raw_node_decisions)
      if self.compute_mode == 'log':
        # mask path adds 1e-8 per node probability; equivalent guard here is
        # on the product, so an underflowed leaf does not give log(0).
        leaf_probs = tf.math.log(leaf_probs + 1e-8)
        theta = tf.expand_dims(self.theta, axis=0)
        prediction = tf.math.exp(tf.expand_dims(leaf_probs, axis=2) + theta)
        prediction = tf.reduce_sum(prediction, axis=1)
        leaf_probs = tf.math.exp(leaf_probs)
      else:
        prediction = tf.matmul(leaf_probs, self.theta)  # raw logits; softmax applied at output only
    else:
      # add numerical stability
      y = tf.expand_dims(raw_node_decisions, axis=2)
      y_repeated = tf.repeat(y, self.num_leaves, axis=2)
      z = tf.multiply(y_repeated, self.mask_matrix)

      # P \in [batch_size, num_internal_nodes, num_leaves]
      probs = tf.nn.relu(z) + (self.ones_nodes - tf.nn.relu(-z)) + self.mask 
      probs += 1e-8
      if self.compute_mode == 'log':
        probs = tf.math.log(probs)

        # axis=1 because it corresponds to internal nodes
        leaf_probs = tf.math.reduce_sum(probs, axis=1)
        theta = tf.expand_dims(self.theta, axis=0)

        prediction = tf.math.exp(tf.expand_dims(leaf_probs, axis=2) + theta)
        prediction = tf.reduce_sum(prediction, axis=1)

        leaf_probs = tf.math.exp(leaf_probs)
      else:
        leaf_probs = tf.math.reduce_prod(probs, axis=1)
        if pred_type == 'categorical':
            prediction = tf.matmul(leaf_probs, self.theta)  # raw logits; softmax applied at output only
        else:
            prediction = tf.matmul(leaf_probs, self.theta)



    if training:
      return prediction, raw_node_decisions, leaf_probs
    return tf.nn.softmax(prediction, axis=-1)
