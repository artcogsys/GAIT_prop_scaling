import tensorflow as tf
import numpy as np
import os

def performance(preds, targets, output_size):
    top1_acc = np.mean(tf.keras.metrics.top_k_categorical_accuracy(
                            y_true=targets, y_pred=preds[:, :output_size]
                            , k=1).numpy())
    top5_acc = np.mean(tf.keras.metrics.top_k_categorical_accuracy(
                            y_true=targets, y_pred=preds[:, :output_size]
                            , k=5).numpy())

    loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true=targets, y_pred=preds[:, :output_size],
                                                                   from_logits=True))


    return top1_acc, top5_acc, loss

def progress(epoch, percent_complete, acc, acc_top5):
    out = f'Epoch: {epoch} Percent Complete: {percent_complete} Test acc: {acc} Test acc top5: {acc_top5}'
    bs = '\r'            # The backspace
    print(bs, end="")
    print(out, end="", flush=True)


def save_weights(save_path, net, optimizer):
    optimizer_path = os.path.join(save_path, 'optimizer')
    net_path = os.path.join(save_path, 'net')
    os.makedirs(save_path, exist_ok=True)

    # Save optimizer weights
    os.makedirs(optimizer_path, exist_ok=True)
    np.save(optimizer_path+'/optimizer_weights.npy', optimizer.get_weights())

    # Save network weights
    os.makedirs(net_path, exist_ok=True)
    for index, layer in enumerate(net.layers):
        layer_path = net_path+f"/{str(index)}"
        os.makedirs(layer_path, exist_ok=True)
        np.save(layer_path+'/weight_matrix.npy', layer.weight_matrix)
        np.save(layer_path+'/bias.npy', layer.bias)


def load_weights(load_path, net, optimizer, input_shape):

    # Load optimizer weights
    optimizer_path = os.path.join(load_path, 'optimizer')
    optimizer_weights = np.load(optimizer_path+'/optimizer_weights.npy', allow_pickle=True)

    # Run test batch to initialize optimizer
    test_batch = tf.random.normal(input_shape, dtype=tf.float64)
    with tf.GradientTape() as tape:
        y = net(test_batch)[-1]
        loss = tf.random.normal(y.shape, dtype=tf.float64)-y
    grads = tape.gradient(loss, net.trainable_weights)
    optimizer.apply_gradients(zip(grads, net.trainable_weights)) # this will be overwritten when loading network weights

    # Set optimizer weights
    optimizer.set_weights(optimizer_weights)

    # Load network weights
    net_path = os.path.join(load_path, 'net')
    for index, layer in enumerate(net.layers):
        layer_path = net_path+f"/{str(index)}"
        w = np.load(layer_path+'/weight_matrix.npy', allow_pickle=True)
        b = np.load(layer_path+'/bias.npy', allow_pickle=True)
        layer.set_weights([w, b])



def load_metrics(load_path):

    with open (load_path+'/cors_per_layer.txt', 'r') as f:
        cors_per_layer = []
        for line in f.readlines():
            cors_per_layer.append([float(x) for x in line.split(' ')])

    with open(load_path + '/test_acc.txt', 'r') as f:
        test_accuracies = []
        for line in f.readlines():
            test_accuracies.append(float(line))

    with open(load_path + '/test_acc_top5.txt', 'r') as f:
        test_accuracies_top5 = []
        for line in f.readlines():
            test_accuracies_top5.append(float(line))

    with open(load_path + '/test_loss.txt', 'r') as f:
        test_losses = []
        for line in f.readlines():
            test_losses.append(float(line))

    with open(load_path + '/train_acc.txt', 'r') as f:
        train_accuracies = []
        for line in f.readlines():
            train_accuracies.append(float(line))

    with open(load_path + '/train_acc_top5.txt', 'r') as f:
        train_accuracies_top5 = []
        for line in f.readlines():
            train_accuracies_top5.append(float(line))

    with open(load_path + '/train_loss.txt', 'r') as f:
        train_losses = []
        for line in f.readlines():
            train_losses.append(float(line))

    return cors_per_layer, test_accuracies, test_accuracies_top5, test_losses, train_accuracies, train_accuracies_top5, train_losses