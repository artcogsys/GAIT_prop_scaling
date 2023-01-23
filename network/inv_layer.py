import tensorflow as tf

class InvDense(tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim, activation, adaptive_gamma, square, seed):

    super().__init__()
    self.output_dim = output_dim
    self.square = square
    self.seed = seed

    init = tf.keras.initializers.orthogonal(seed=seed)
    if self.square:  # if square, choose ortho init
        self.weight_matrix = tf.Variable(init([input_dim, input_dim], dtype=tf.float64), trainable=True)
        self.bias = tf.Variable(tf.zeros([input_dim], dtype=tf.float64), dtype=tf.float64, trainable=True)
        self.inverse_matrix = None

    else:
        self.weight_matrix = tf.Variable(init([input_dim, output_dim], dtype=tf.float64), trainable=True)
        self.bias = tf.Variable(tf.zeros([self.output_dim], dtype=tf.float64), dtype=tf.float64, trainable=True)

    if activation == 'leaky_relu':
        self.activation_fn = self.leaky_relu
        self.activation_fn_inv = self.leaky_relu_inverse
        self.activation_fn_derivative = self.leaky_relu_derivative
    else:
        self.activation_fn = lambda x:x
        self.activation_fn_inv = lambda x:x
        self.activation_fn_derivative = lambda x:1

    if adaptive_gamma:
        self.compute_gamma_fn = self.compute_adaptive_gamma
    else:
        self.compute_gamma_fn = self.compute_nonadaptive_gamma

  def call(self, x):
      x = tf.keras.layers.Flatten()(x)
      return self.activation_fn(tf.matmul(x, self.weight_matrix) + self.bias)

  def update_inverse(self):
      w = self.weight_matrix
      self.w_inv = tf.linalg.inv(w)

  def inverse(self, x):
      return tf.einsum('ji, nj -> ni', self.w_inv, self.activation_fn_inv(x) - self.bias)

  def leaky_relu(self, data, alpha=.1):
      return tf.tensor_scatter_nd_update(data, tf.where(data < 0), data[data < 0] * alpha)

  def leaky_relu_derivative(self, data, alpha=.1):
      derivatives = tf.ones(data.shape, dtype=tf.float64)
      return tf.tensor_scatter_nd_update(derivatives, tf.where(data < 0), derivatives[data < 0] * alpha)

  def leaky_relu_inverse(self, data, alpha=.1):
      return tf.tensor_scatter_nd_update(data, tf.where(data < 0), data[data < 0] / alpha)

  def get_ortho_gradient(self):
      with tf.GradientTape() as tape:
          w = self.weight_matrix
          tape.watch(w)
          ortho_loss = self.get_ortho_loss(w)
      return tape.gradient([ortho_loss], w)

  def get_ortho_loss(self, w):
      reg = tf.einsum('ik, jk -> ij', w, w)
      target = tf.eye(w.shape[0], dtype=tf.float64)
      return tf.reduce_sum(tf.square(reg - target))

  def compute_adaptive_gamma(self, targets, activations, gamma):
      error = (targets - activations)
      distance_to_target = tf.reshape(error, [len(error), -1])
      l2_norm = tf.linalg.norm(distance_to_target, 2, axis=1)      
      layer_gamma = gamma / l2_norm
      layer_gamma *= tf.sqrt(tf.cast(distance_to_target.shape[1], tf.float64))
      return layer_gamma

  def compute_nonadaptive_gamma(self, targets, activations, gamma):
      return gamma


  def propagate_target(self, target, layer_output, layer_gamma, mult_factor):

      layer_gamma = self.compute_gamma_fn(target, layer_output, layer_gamma)
      mult_factor /= layer_gamma

      layer_derivatives = self.activation_fn_derivative(
          self.activation_fn_inv(layer_output))

      grad_adj_incr_factor = tf.expand_dims(layer_gamma, 1) * layer_derivatives * layer_derivatives

      nudged_activation = ((grad_adj_incr_factor * target)) + (
              (1.0 - grad_adj_incr_factor) * layer_output)

      return self.inverse(nudged_activation), mult_factor


class InvConv(tf.keras.layers.Layer):
  def __init__(self, chan_in, filters, kernel_size, stride, activation, adaptive_gamma, square, input_dims, seed):

    super().__init__()
    self.filters = filters
    self.stride = stride
    self.kernel_size = kernel_size
    self.chan_in = chan_in
    self.square = square
    self.x = 0
    self.seed = seed

    init = tf.keras.initializers.orthogonal(seed=seed)
    if self.square:  # if square, choose ortho init
        self.output_dim = self.chan_in * self.kernel_size * self.kernel_size
        self.weight_matrix = init([self.output_dim, self.output_dim], dtype=tf.float64)
        self.weight_matrix = tf.Variable(tf.reshape(self.weight_matrix,
                                        (self.kernel_size, self.kernel_size, self.chan_in, self.output_dim)), dtype=tf.float64, trainable=True)
        self.inverse_matrix = None

    else:  # else choose Glorot init
        self.output_dim = self.filters
        self.weight_matrix = tf.Variable(init([self.kernel_size, self.kernel_size, self.chan_in, self.output_dim], dtype=tf.float64), trainable=True)

    # Initialize biases
    self.bias = tf.Variable(tf.zeros([self.output_dim], dtype=tf.float64), dtype=tf.float64, trainable=True)

    if activation == 'leaky_relu':
        self.activation_fn = self.leaky_relu
        self.activation_fn_inv = self.leaky_relu_inverse
        self.activation_fn_derivative = self.leaky_relu_derivative
    else:
        self.activation_fn = lambda x: x
        self.activation_fn_inv = lambda x: x
        self.activation_fn_derivative = lambda x: 1

    self.input_dims = input_dims
    self.example_input = tf.ones(input_dims, dtype=tf.float64)
    self.example_output = self(self.example_input)

    # Create reconstruction correction for then stride != kernel
    if self.square:
        self.update_inverse()
        self.reconstruction_correction = self.example_input
        self.reconstruction_correction = self.inverse(self.example_output)
    self.output_dims = self.example_output.shape

    if adaptive_gamma:
        self.compute_gamma_fn = self.compute_adaptive_gamma
    else:
        self.compute_gamma_fn = self.compute_nonadaptive_gamma


  def call(self, x):
      if self.square:
          self.x = x
      y = tf.nn.conv2d(input=x,
                       filters=self.weight_matrix,
                       strides=self.stride,
                       padding='SAME',
                       data_format='NCHW')

      return self.activation_fn(y + self.bias[tf.newaxis, :, tf.newaxis, tf.newaxis])


  def update_inverse(self):
      if self.square:
          w_square_transposed = tf.transpose(tf.reshape(self.weight_matrix, [self.output_dim, self.output_dim]))
          w_square_inv = tf.linalg.inv(w_square_transposed)
          self.inverse_matrix = tf.reshape(w_square_inv,
                                           [self.kernel_size, self.kernel_size, self.chan_in, self.output_dim])
      else:
          raise NotImplementedError('A non-square layer cannot be inverted')

  def inverse(self, data):
      if self.square:
          inverse = tf.nn.conv2d_transpose(
              input=self.activation_fn_inv(data) - self.bias[tf.newaxis, :, tf.newaxis, tf.newaxis],
              filters=self.inverse_matrix,
              strides=self.stride,
              output_shape=[len(data)] + self.input_dims[1:],
              padding='SAME',
              data_format='NCHW')
          return inverse - ((self.reconstruction_correction-1)*self.x) # Correct reconstruction when there are overlapping receptive fields
      else:
          raise NotImplementedError('A non-square layer cannot be inverted')


  def leaky_relu(self, data, alpha=.1):
      return tf.tensor_scatter_nd_update(data, tf.where(data < 0), data[data < 0] * alpha)

  def leaky_relu_derivative(self, data, alpha=.1):
      derivatives = tf.ones(data.shape, dtype=tf.float64)
      return tf.tensor_scatter_nd_update(derivatives, tf.where(data < 0), derivatives[data < 0] * alpha)

  def leaky_relu_inverse(self, data, alpha=.1):
      return tf.tensor_scatter_nd_update(data, tf.where(data < 0), data[data < 0] / alpha)

  def get_ortho_gradient(self):
      with tf.GradientTape() as tape:
          w = tf.reshape(self.weight_matrix, [self.weight_matrix.shape[-1], self.weight_matrix.shape[-1]])
          tape.watch(w)
          ortho_loss = self.get_ortho_loss(w)
      grad = tape.gradient([ortho_loss], w)
      return tf.reshape(grad, self.weight_matrix.shape)

  def get_ortho_loss(self, w):
      reg = tf.einsum('ik, jk -> ij', w, w)
      target = tf.eye(w.shape[0], dtype=tf.float64)
      return tf.reduce_sum(tf.square(reg - target))

  def compute_adaptive_gamma(self, targets, activations, gamma):
      error = (targets - activations)
      distance_to_target = tf.reshape(error, [len(error), -1])
      l2_norm = tf.linalg.norm(distance_to_target, 2, axis=1)
      layer_gamma = gamma / l2_norm
      layer_gamma *= tf.sqrt(tf.cast(distance_to_target.shape[1], tf.float64))
      return layer_gamma

  def compute_nonadaptive_gamma(self, targets, activations, gamma):
      return gamma

  def propagate_target(self, target, layer_output, layer_gamma, mult_factor):

      layer_gamma = self.compute_gamma_fn(target, layer_output, layer_gamma)
      mult_factor /= layer_gamma

      layer_derivatives = self.activation_fn_derivative(
          self.activation_fn_inv(layer_output))

      grad_adj_incr_factor = tf.expand_dims(tf.expand_dims(tf.expand_dims(layer_gamma, 1), 1), 1) * layer_derivatives * layer_derivatives

      nudged_activation = ((grad_adj_incr_factor * target)) + (
              (1.0 - grad_adj_incr_factor) * layer_output)

      return self.inverse(nudged_activation), mult_factor