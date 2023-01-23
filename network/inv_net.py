import tensorflow as tf
from network.inv_layer import InvDense, InvConv

def add_ortho_gradients(network, gradients, ortho_weight):
    for index, layer in enumerate(network.layers):
        if layer.square:
            ortho_gradient = layer.get_ortho_gradient()
            gradients[(2*index)] += ortho_gradient*ortho_weight
    return gradients

def get_local_updates(net, layer_outputs, layer_targets, mult_factors, tape):

    grads = []
    for index, layer in enumerate(net.layers):
        with tape:
            error = tf.square(layer_targets[index] - layer_outputs[index])
            loss = tf.reduce_sum(error, axis=-1)
            loss = tf.reshape(loss, (loss.shape[0], -1)) * mult_factors[index][:, tf.newaxis]
        grads.extend(tape.gradient(loss, layer.trainable_weights))
    return grads

def update_inverse_model(net):
    for layer in net.layers:
        if hasattr(layer, 'square') and layer.square:
            layer.update_inverse()

class FA_Updater:

    def __init__(self):
        self.random_weight_matrices = []

    def initialize_random_weights(self, net):
        for layer in net.layers:
            init = tf.keras.initializers.orthogonal(seed=layer.seed+1)
            random_weights = init(layer.weight_matrix.shape, dtype=tf.float64)
            self.random_weight_matrices.append(random_weights)

    def get_updates(self, net, outputs, error, tape):

        updates = []

        # Loop over layers to compute updates and propagate errors
        for layer, R, output in zip(net.layers[::-1], self.random_weight_matrices[::-1], outputs[::-1]):
            # # Get update
            if len(output.shape) > 2: # For conv outputs we reshape the error and zero pad
                error = tf.reshape(error, output.shape)

            # Compute local loss for automatic weight update
            with tape:
                local_target = tf.Variable(output - error)
                loss = tf.reduce_sum(tf.square(output - local_target))

            w_update, b_update = tape.gradient(loss, layer.trainable_weights)
            updates.extend([b_update, w_update])

            # Multiply error by layer derivatives evaluated at pre-activation level
            error = layer.activation_fn_derivative(layer.activation_fn_inv(output)) * error

            # Propagate error
            if isinstance(layer, InvDense):
                error = tf.matmul(error, tf.transpose(R))
            if isinstance(layer, InvConv):
                error = tf.nn.conv2d_transpose(
                        input=error,
                        filters=R,
                        strides=layer.stride,
                        output_shape=[error.shape[0]] + layer.input_dims[1:],
                        padding='SAME',
                        data_format='NCHW')

        return updates[::-1]



class CNN_imagenet(tf.keras.Model):
    def __init__(self, input_shape=None, activation=None, adaptive_gamma=None, seed=None, use_inverses=False):

        super(CNN_imagenet, self).__init__()

        example_input = tf.random.normal(shape=input_shape, dtype=tf.float64)
        print(f'Example input shape: {example_input.shape}')

        # Conv1
        self.conv1 = InvConv(chan_in=3, filters=24, kernel_size=9, stride=4, activation=activation, adaptive_gamma=adaptive_gamma,
                             input_dims=input_shape, square=False, seed=seed + 1)
        example_input = self.conv1(example_input)[:, :self.conv1.filters, :, :]
        print(f'Conv 1 output shape: {example_input.shape}')

        # Conv2
        self.conv2 = InvConv(chan_in=self.conv1.filters, filters=24, kernel_size=3, stride=2, activation=activation, adaptive_gamma=adaptive_gamma,
                             input_dims=example_input.shape, square=use_inverses, seed=seed + 2)
        example_input = self.conv2(example_input)[:, :self.conv2.filters, :, :]
        print(f'Conv 2 output shape: {example_input.shape}')

        # Conv3
        self.conv3 = InvConv(chan_in=self.conv2.filters, filters=48, kernel_size=5, stride=2, activation=activation, adaptive_gamma=adaptive_gamma,
                             input_dims=example_input.shape, square=use_inverses, seed=seed + 3)
        example_input = self.conv3(example_input)[:, :self.conv3.filters, :, :]
        print(f'Conv 3 output shape: {example_input.shape}')

        # Conv4
        self.conv4 = InvConv(chan_in=self.conv3.filters, filters=48, kernel_size=3, stride=2, activation=activation, adaptive_gamma=adaptive_gamma,
                             input_dims=example_input.shape, square=use_inverses, seed=seed + 4)
        example_input = self.conv4(example_input)[:, :self.conv4.filters, :, :]
        print(f'Conv 4 output shape: {example_input.shape}')

        # Conv5
        self.conv5 = InvConv(chan_in=self.conv4.filters, filters=96, kernel_size=3, stride=1, activation=activation,
                             adaptive_gamma=adaptive_gamma,
                             input_dims=example_input.shape, square=use_inverses, seed=seed + 5)
        example_input = self.conv5(example_input)[:, :self.conv5.filters, :, :]
        print(f'Conv 5 output shape: {example_input.shape}')

        # Conv6
        self.conv6 = InvConv(chan_in=self.conv5.filters, filters=96, kernel_size=3, stride=2, activation=activation,
                             adaptive_gamma=adaptive_gamma,
                             input_dims=example_input.shape, square=use_inverses, seed=seed + 6)
        example_input = self.conv6(example_input)[:, :self.conv6.filters, :, :]
        print(f'Conv 6 output shape: {example_input.shape}')

        # Conv7
        self.conv7 = InvConv(chan_in=self.conv6.filters, filters=192, kernel_size=3, stride=1, activation=activation,
                             adaptive_gamma=adaptive_gamma,
                             input_dims=example_input.shape, square=use_inverses, seed=seed + 7)
        example_input = self.conv7(example_input).numpy()[:, :self.conv7.filters, :, :]
        print(f'Conv 7 output shape: {example_input.shape}')

        self.output_layer = InvDense(input_dim=len(example_input[0].flatten()),
                                     output_dim=1000,
                                     activation='linear',
                                     adaptive_gamma=adaptive_gamma,
                                     square=use_inverses,
                                     seed=seed+8)


    def call(self, x):

        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2[:, :self.conv2.filters, :, :])
        h4 = self.conv4(h3[:, :self.conv3.filters, :, :])
        h5 = self.conv5(h4[:, :self.conv4.filters, :, :])
        h6 = self.conv6(h5[:, :self.conv5.filters, :, :])
        h7 = self.conv7(h6[:, :self.conv6.filters, :, :])
        y = self.output_layer(h7[:, :self.conv7.filters, :, :])

        return [h1, h2, h3, h4, h5, h6, h7, y]

    def inverse_pass(self, target, hidden_states, gamma):
        gamma = tf.zeros(target.shape[0], dtype=tf.float64) + gamma
        y_mf = tf.ones(target.shape[0], dtype=tf.float64)

        y_target = tf.concat([target[:, :1000], hidden_states[-1][:, 1000:]], axis=1)  # add aux units to output targets for inversion
        conv7_target, conv7_mf = self.output_layer.propagate_target(y_target, hidden_states[-1], gamma, y_mf)

        conv7_target = tf.reshape(conv7_target, [-1, self.conv7.filters, 3, 3])
        conv7_target = tf.concat([conv7_target[:, :self.conv7.filters, :, :], hidden_states[-2][:, self.conv7.filters:, :, :]], axis=1)

        conv6_target, conv6_mf = self.conv7.propagate_target(conv7_target, hidden_states[-2], gamma, conv7_mf)
        conv6_target = tf.concat([conv6_target[:, :self.conv6.filters, :, :], hidden_states[-3][:, self.conv6.filters:, :, :]], axis=1)

        conv5_target, conv5_mf = self.conv6.propagate_target(conv6_target, hidden_states[-3], gamma, conv6_mf)
        conv5_target = tf.concat([conv5_target[:, :self.conv5.filters, :, :], hidden_states[-4][:, self.conv5.filters:, :, :]], axis=1)

        conv4_target, conv4_mf = self.conv5.propagate_target(conv5_target, hidden_states[-4], gamma, conv5_mf)
        conv4_target = tf.concat([conv4_target[:, :self.conv4.filters, :, :], hidden_states[-5][:, self.conv4.filters:, :, :]], axis=1)

        conv3_target, conv3_mf = self.conv4.propagate_target(conv4_target, hidden_states[-5], gamma, conv4_mf)
        conv3_target = tf.concat([conv3_target[:, :self.conv3.filters, :, :], hidden_states[-6][:, self.conv3.filters:, :, :]], axis=1)

        conv2_target, conv2_mf = self.conv3.propagate_target(conv3_target, hidden_states[-6], gamma, conv3_mf)
        conv2_target = tf.concat([conv2_target[:, :self.conv2.filters, :, :], hidden_states[-7][:, self.conv2.filters:, :, :]], axis=1)

        conv1_target, conv1_mf = self.conv2.propagate_target(conv2_target, hidden_states[-7], gamma, conv2_mf)
        conv1_target = tf.concat([conv1_target[:, :self.conv1.filters, :, :], hidden_states[-8][:, self.conv1.filters:, :, :]], axis=1)

        return [conv1_target, conv2_target, conv3_target, conv4_target, conv5_target, conv6_target, conv7_target, y_target], \
               [conv1_mf, conv2_mf, conv3_mf, conv4_mf, conv5_mf, conv6_mf, conv7_mf, y_mf]


class FCNET_cifar10(tf.keras.Model):

    def __init__(self, input_shape=None, activation=None, adaptive_gamma=None, seed=None, use_inverses=False):
        super(FCNET_cifar10, self).__init__()

        example_input = tf.random.normal(shape=input_shape, dtype=tf.float64)
        print(f'Example input shape: {example_input.shape}')

        self.dense1 = InvDense(input_dim=len(example_input[0].numpy().flatten()),
                               output_dim=1024,
                               activation=activation,
                               adaptive_gamma=adaptive_gamma,
                               square=False,
                               seed=seed+1)
        self.dense2 = InvDense(input_dim=1024,
                               output_dim=1024,
                               activation=activation,
                               adaptive_gamma=adaptive_gamma,
                               square=use_inverses,
                               seed=seed+2)
        self.dense3 = InvDense(input_dim=1024,
                               output_dim=1024,
                               activation=activation,
                               adaptive_gamma=adaptive_gamma,
                               square=use_inverses,
                               seed=seed+3)
        self.dense4 = InvDense(input_dim=1024,
                               output_dim=1024,
                               activation=activation,
                               adaptive_gamma=adaptive_gamma,
                               square=use_inverses,
                               seed=seed+4)
        self.dense5 = InvDense(input_dim=1024,
                               output_dim=1024,
                               activation=activation,
                               adaptive_gamma=adaptive_gamma,
                               square=use_inverses,
                               seed=seed+5)
        self.dense6 = InvDense(input_dim=1024,
                               output_dim=1024,
                               activation=activation,
                               adaptive_gamma=adaptive_gamma,
                               square=use_inverses,
                               seed=seed+6)
        self.output_layer = InvDense(input_dim=1024,
                                     output_dim=10,
                                     activation=None,
                                     adaptive_gamma=adaptive_gamma,
                                     square=use_inverses,
                                     seed=seed+7)


    def call(self, inputs):

        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        h4 = self.dense4(h3)
        h5 = self.dense5(h4)
        h6 = self.dense6(h5)
        y = self.output_layer(h6)
        return [h1, h2, h3, h4, h5, h6, y]

    def inverse_pass(self, target, hidden_states, gamma):

        gamma = tf.zeros(target.shape[0], dtype=tf.float64) + gamma
        y_mf = tf.ones(target.shape[0], dtype=tf.float64)

        y_target = tf.concat([target[:, :10], hidden_states[-1][:, 10:]], axis=1) # add aux units to output targets for inversion
        h6_target, h6_mf = self.output_layer.propagate_target(y_target, hidden_states[-1], gamma, y_mf)
        h5_target, h5_mf = self.dense6.propagate_target(h6_target, hidden_states[-2], gamma, h6_mf)
        h4_target, h4_mf = self.dense5.propagate_target(h5_target, hidden_states[-3], gamma, h5_mf)
        h3_target, h3_mf = self.dense4.propagate_target(h4_target, hidden_states[-4], gamma, h4_mf)
        h2_target, h2_mf = self.dense3.propagate_target(h3_target, hidden_states[-5], gamma, h3_mf)
        h1_target, h1_mf = self.dense2.propagate_target(h2_target, hidden_states[-6], gamma, h2_mf)

        return [h1_target, h2_target, h3_target, h4_target, h5_target, h6_target, y_target],\
               [h1_mf, h2_mf, h3_mf, h4_mf, h5_mf, h6_mf, y_mf]




class FCNET_cifar10_shallow(tf.keras.Model):

    def __init__(self, input_shape=None, activation=None, adaptive_gamma=None, seed=None, use_inverses=False):
        super(FCNET_cifar10_shallow, self).__init__()

        example_input = tf.random.normal(shape=input_shape, dtype=tf.float64)
        print(f'Example input shape: {example_input.shape}')

        self.dense1 = InvDense(input_dim=len(example_input[0].numpy().flatten()),
                               output_dim=1024,
                               activation=activation,
                               adaptive_gamma=adaptive_gamma,
                               square=False,
                               seed=seed+1)
        self.dense2 = InvDense(input_dim=1024,
                               output_dim=1024,
                               activation=activation,
                               adaptive_gamma=adaptive_gamma,
                               square=use_inverses,
                               seed=seed+2)
        self.dense3 = InvDense(input_dim=1024,
                               output_dim=1024,
                               activation=activation,
                               adaptive_gamma=adaptive_gamma,
                               square=use_inverses,
                               seed=seed+3)
        self.output_layer = InvDense(input_dim=1024,
                                     output_dim=10,
                                     activation=None,
                                     adaptive_gamma=adaptive_gamma,
                                     square=use_inverses,
                                     seed=seed+7)


    def call(self, inputs):

        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        y = self.output_layer(h3)
        return [h1, h2, h3, y]

    def inverse_pass(self, target, hidden_states, gamma):

        gamma = tf.zeros(target.shape[0], dtype=tf.float64) + gamma
        y_mf = tf.ones(target.shape[0], dtype=tf.float64)

        y_target = tf.concat([target[:, :10], hidden_states[-1][:, 10:]], axis=1) # add aux units to output targets for inversion
        h3_target, h3_mf = self.output_layer.propagate_target(y_target, hidden_states[-1], gamma, y_mf)
        h2_target, h2_mf = self.dense3.propagate_target(h3_target, hidden_states[-2], gamma, h3_mf)
        h1_target, h1_mf = self.dense2.propagate_target(h2_target, hidden_states[-3], gamma, h2_mf)

        return [h1_target, h2_target, h3_target, y_target],\
               [h1_mf, h2_mf, h3_mf, y_mf]