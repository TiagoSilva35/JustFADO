"""Fair MLP - Strong baseline for online fair learning."""

import tensorflow as tf
import tensorflow_probability as tfp


class FairReLUNetwork(tf.Module):
  """Strong MLP baseline with multiple layers and modern techniques."""
  def __init__(self,
               data_dim,
               tree_depth,
               num_classes,
               activation='relu',
               use_layer_norm=True,
               dropout_rate=0.1,
               num_layers=2,
               hidden_multiplier=4):
    """Constructor.
    
    Args:
      data_dim: dimension of the data.
      tree_depth: depth parameter (hidden_size = 2^depth - 1).
      num_classes: number of target task classes.
      activation: activation function ('relu', 'gelu', 'sigmoid', 'smoother').
      use_layer_norm: whether to use layer normalization.
      dropout_rate: dropout probability.
      num_layers: number of hidden layers.
      hidden_multiplier: multiply base hidden size by this factor.
    """
    super(FairReLUNetwork, self).__init__()
    assert tree_depth > 1
    
    # Larger hidden size for more capacity
    base_hidden = 2**tree_depth - 1
    self.hidden_size = base_hidden * hidden_multiplier  # e.g., 15 * 4 = 60
    self.num_layers = num_layers
    self.use_layer_norm = use_layer_norm
    self.dropout_rate = dropout_rate
    
    # activation function
    if activation == 'sigmoid':
      self.activation = tf.keras.activations.sigmoid
    elif activation == 'smoother':
      self.activation = tfp.math.smootherstep
    elif activation == 'relu':
      self.activation = tf.keras.activations.relu
    elif activation == 'gelu':
      self.activation = tf.keras.activations.gelu
    elif activation == 'swish':
      self.activation = tf.keras.activations.swish
    else:
      raise ValueError(f'Unknown activation: {activation}')
    
    # Build layers
    self.hidden_layers = []
    self.layer_norms = []
    self.dropouts = []
    
    for i in range(num_layers):
      in_dim = data_dim if i == 0 else self.hidden_size
      
      # Dense layer with proper initialization
      layer = tf.keras.layers.Dense(
          self.hidden_size,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=tf.keras.regularizers.l2(1e-4),
          name=f'hidden_{i}'
      )
      self.hidden_layers.append(layer)
      
      if use_layer_norm:
        self.layer_norms.append(tf.keras.layers.LayerNormalization(axis=-1))
      
      if dropout_rate > 0:
        self.dropouts.append(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    self.output_layer = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        name='output'
    )

  def __call__(self, inputs, training=False):
    """Forward pass.
    
    Args:
      inputs: input tensor of shape [batch_size, data_dim].
      training: whether in training mode (affects dropout).
      
    Returns:
      predictions: softmax probabilities of shape [batch_size, num_classes].
    """
    x = inputs
    
    for i in range(self.num_layers):
      x = self.hidden_layers[i](x)
      
      if self.use_layer_norm:
        x = self.layer_norms[i](x)
      
      x = self.activation(x)
      
      if self.dropout_rate > 0 and training:
        x = self.dropouts[i](x, training=training)
    
    logits = self.output_layer(x)
    return tf.nn.softmax(logits, axis=-1)
