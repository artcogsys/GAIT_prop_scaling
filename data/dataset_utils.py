from glob import glob
from os.path import join

import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np

def indices_to_onehot(data, nb_categories=10):
    onehot = np.zeros((len(data), nb_categories))
    onehot[range(len(data)), data] = 1.0
    return onehot

def rgb2gray(rgb):
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def get_imagenet_iterator(imagenet_path, subsample):
    def parse_record(record):
        parse_dict = {"features": tf.io.FixedLenFeature([], tf.string),
                      "label": tf.io.FixedLenFeature(shape=[1000], dtype=tf.int64)}

        parsed_features = tf.io.parse_single_example(record, parse_dict)

        imgs, labels = parsed_features["features"], parsed_features["label"]
        imgs = tf.io.decode_raw(imgs, np.float16)
        imgs.set_shape([224 * 224 * 3])
        imgs = tf.cast(tf.reshape(imgs, [3, 224, 224]), tf.float64)
        return imgs, labels

    def create_iterator(filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        return dataset.map(parse_record).apply(tf.data.experimental.ignore_errors()) # drops record if it produces error (very rare)

    train_files = glob(join(imagenet_path, 'train/*'))
    val_files = glob(join(imagenet_path, 'val/*'))
    if subsample:
        train_files = train_files[:3]
        val_files = val_files[:3]
    train_data = create_iterator(train_files)
    test_data = create_iterator(val_files)

    data_params = {'chan_in': 3,
                   'h_in': 224,
                   'w_in': 224,
                   'output_size': 1000}

    return train_data, test_data, data_params

def get_data(dataset, validation=False, greyscale=False, imagenet_path=None, subsample=None):
    # MNIST/fashion - Loading
    x_train, y_train, x_test, y_test = None, None, None, None
    if dataset == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if dataset == 'FMNIST':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if dataset == 'KMNIST':
        try:
            x_train = np.load('./kmnist-train-imgs.npz')['arr_0']
        except FileNotFoundError:
            print("ERROR: KMNIST files not found. You must download the kmnist dataset to this folder in the npz format.")
            exit()
        x_test = np.load('./kmnist-test-imgs.npz')['arr_0']
        y_train = np.load('./kmnist-train-labels.npz')['arr_0']
        y_test = np.load('./kmnist-test-labels.npz')['arr_0']
    if dataset == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = [x[0] for x in y_train]
        y_test = [x[0] for x in y_test]
        if greyscale:
            x_train, x_test = rgb2gray(x_train), rgb2gray(x_test)
    if dataset == 'CIFAR100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        y_train = [x[0] for x in y_train]
        y_test = [x[0] for x in y_test]
        if greyscale:
            x_train, x_test = rgb2gray(x_train), rgb2gray(x_test)
    if dataset == 'ImageNet':
        return get_imagenet_iterator(imagenet_path=imagenet_path, subsample=subsample)

    # Normalizing to 0->1
    maxval = np.max(np.abs(x_train))
    x_train = x_train / maxval
    x_test = x_test / maxval
   
    if validation:
      # Taking 10,000 of training set as a validation set
      np.random.seed(42)
      valid_choice = np.random.choice(len(y_train), size=10000, replace=False)
      mask = np.zeros(len(y_train))
      mask[valid_choice] = 1
      x_test, y_test = x_train[mask == 1], y_train[mask == 1]
      x_train, y_train = x_train[mask != 1], y_train[mask != 1]

    # Reshape greyscale image to channel*height*width
    dim = len(x_train.shape)
    if dim == 3:
        x_train = x_train[:, np.newaxis, :, :]
        x_test = x_test[:, np.newaxis, :, :]

    # Reshape color image to channel*height*width
    if dim == 4:
        x_train = x_train.swapaxes(3, 1)
        x_test = x_test.swapaxes(3, 1)

    # Moving my datasets to the device
    y_train = indices_to_onehot(y_train)
    y_test = indices_to_onehot(y_test)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    chan_in, h_in, w_in = x_train.shape[1:]

    data_params = {'chan_in': chan_in,
                   'h_in': h_in,
                   'w_in': w_in,
                   'output_size': y_train.shape[1]}

    return train_data, test_data, data_params