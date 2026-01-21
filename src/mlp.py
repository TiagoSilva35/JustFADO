"""Fair MLP tree."""

import tensorflow as tf
import tensorflow_probability as tfp


class FairReLUNetwork(tf.Module):
  """Fair MLP with improved architecture and regularization."""
  def __init__(self,
               data_dim,
               tree_depth,
               num_classes,
               activation='relu',
               use_layer_norm=True,
               dropout_rate=0.0):
    """Constructor.
    
    Args:
      data_dim: dimension of the data.
      tree_depth: depth of the binary tree.
      num_classes: number of target task classes.
      activation: activation function.
      use_layer_norm: whether to use layer normalization.
      dropout_rate: dropout probability (0.0 means no dropout).
    """
    super(FairReLUNetwork, self).__init__()
    assert tree_depth > 1
    self.num_internal_nodes = 2**tree_depth - 1
    self.use_layer_norm = use_layer_norm
    self.dropout_rate = dropout_rate
    
    # internal node parameters with Xavier initialization
    initializer = tf.keras.initializers.GlorotUniform()
    self.weight = tf.Variable(
        initializer([data_dim, self.num_internal_nodes]), name='W'
    )
    self.bias = tf.Variable(
        tf.zeros([self.num_internal_nodes]), name='B'
    )
    
    # activation function
    if activation == 'sigmoid':
      self.activation = tf.keras.activations.sigmoid
    elif activation == 'smoother':
      self.activation = tfp.math.smootherstep
    elif activation == 'relu':
      self.activation = tf.keras.activations.relu
    elif activation == 'gelu':
      self.activation = tf.keras.activations.gelu
    else:
      raise ValueError(f'Unknown activation: {activation}')
    
    # optional layer norm
    if self.use_layer_norm:
      self.layer_norm = tf.keras.layers.LayerNormalization(axis=1)
    
    # optional dropout
    if self.dropout_rate > 0.0:
      self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
    
    # dense layer with proper initialization
    self.dense = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )

  def __call__(self, inputs, training=False):
    """Forward pass with optional layer norm and dropout.
    
    Args:
      inputs: input tensor of shape [batch_size, data_dim].
      training: whether in training mode (affects dropout).
      
    Returns:
      predictions: softmax probabilities of shape [batch_size, num_classes].
    """
    # linear transformation + activation
    hidden = tf.matmul(inputs, self.weight) + self.bias
    
    # optional layer normalization
    if self.use_layer_norm:
      hidden = self.layer_norm(hidden)
    
    hidden = self.activation(hidden)
    
    # optional dropout (only active during training)
    if self.dropout_rate > 0.0 and training:
      hidden = self.dropout(hidden, training=training)
    
    # output layer
    prediction = self.dense(hidden)
    return tf.nn.softmax(prediction, axis=-1)
