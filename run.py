import json
import os
import yaml
import time
import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from network.inv_net import CNN_imagenet, FCNET_cifar10, FCNET_cifar10_shallow, add_ortho_gradients, get_local_updates,\
    update_inverse_model, FA_Updater
from data.dataset_utils import get_data
from utils.compare_updates import get_cors_per_layer
from utils.utils import performance, progress, load_metrics, save_weights, load_weights

parser = argparse.ArgumentParser()
parser.add_argument("-config", help="Experiment configuration.", default='')
parser.add_argument("-gpu", help="GPU number to use.", default='')
parser.add_argument("-subsample", help="Subsample in case of ImageNet for debugging.", default=False)
args = parser.parse_args()

config = args.config
gpu = args.gpu
subsample = args.subsample

with open(f'{config}', 'r') as f:
    config = yaml.load(f)

tf.keras.backend.set_floatx('float64')

# Select GPU
# os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpu}"
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
   gpus[0],
   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

param_list = [
              'train_network', 'max_steps', 'nb_epochs',  'network', 'seed', 'learning_rate', 'batch_size', 'dataset_size',
              'gamma', 'adam_optimizer', 'activation', 'experiment_name', 'algorithm', 'adaptive_gamma',
              'ortho_reg', 'validation', 'greyscale', 'load_epoch', 'imagenet_path', 'beta1', 'beta2', 'epsilon'
             ]

def save_metrics(metric_path):
    metrics = {}

    metrics.update({
        'test_acc': test_accuracies,
        'test_acc_top5': test_accuracies_top5,
        'test_loss': test_losses,
        'train_acc': train_accuracies,
        'train_acc_top5': train_accuracies_top5,
        'train_loss': train_losses,
        'cors_per_layer': cors_per_layer,
        'cors_per_batch': cors_per_batch,
         })

    # Create plots
    os.makedirs(metric_path, exist_ok=True)
    for filename, metric in metrics.items():
        plt.plot(metric)
        plt.savefig(metric_path + filename + '.png')
        plt.clf()
        np.savetxt(metric_path + filename + '.txt', metric)

    with open(os.path.join(outpath, 'params.txt'), 'w') as file:
        file.write(json.dumps(params))


def get_accuracy_on_dataset(data):
    accs_per_batch = []
    accs_top5_per_batch = []
    losses_per_batch = []

    for batch, targets in data.batch(params['batch_size']):
        preds = net(batch)[-1]
        targets = tf.cast(targets, tf.float64)
        acc, acc_top5, loss = performance(preds=preds.numpy(),
                                          targets=targets.numpy(),
                                          output_size=data_params['output_size'])
        accs_per_batch.append(acc)
        accs_top5_per_batch.append(acc_top5)
        losses_per_batch.append(loss)

    accuracy = np.mean(accs_per_batch)
    accuracy_top5 = np.mean(accs_top5_per_batch)
    loss = np.mean(losses_per_batch)
    return accuracy, accuracy_top5, loss

print(f'Running experiments with config file {args.config}.')

for exp in config:
    if exp != 'globals':
        # Set params for the experiment.
        params = {param: config['globals'][param] if param in config['globals'].keys() else config[exp][param] for param in param_list}

        outpath = f"./results/{params['experiment_name']}/"
        os.makedirs(outpath, exist_ok=True)

        # Create optimizer
        if params['adam_optimizer']:
            optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=float(params['beta1']),
                                                 beta_2=params['beta2'], epsilon=float(params['epsilon']))
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])

        # Create network
        if params['network'] == 'ImageNet':
            input_shape = [1, 3, 224, 224]
            net = CNN_imagenet(input_shape=input_shape, activation=params['activation'], adaptive_gamma=params['adaptive_gamma'], seed=params['seed'],
                               use_inverses=True if params['algorithm']=='GAIT' else False)
            train_data, test_data, data_params = get_data('ImageNet', params['validation'], params['greyscale'],
                                                          params['imagenet_path'], subsample=subsample)
            nb_batches_per_epoch = np.ceil(params['dataset_size'] / params['batch_size'])
        if params['network'] == 'CIFAR':
            input_shape = [1, 3, 32, 32]
            net = FCNET_cifar10(input_shape=input_shape, activation=params['activation'], adaptive_gamma=params['adaptive_gamma'], seed=params['seed'],
                                use_inverses=True if params['algorithm']=='GAIT' else False)
            train_data, test_data, data_params = get_data('CIFAR10', params['validation'], params['greyscale'],
                                                          params['imagenet_path'])
            nb_batches_per_epoch = np.ceil(params['dataset_size'] / params['batch_size'])
        if params['network'] == 'CIFAR_shallow':
            input_shape = [1, 3, 32, 32]
            net = FCNET_cifar10_shallow(input_shape=input_shape, activation=params['activation'], adaptive_gamma=params['adaptive_gamma'], seed=params['seed'],
                                        use_inverses=True if params['algorithm']=='GAIT' else False)
            train_data, test_data, data_params = get_data('CIFAR10', params['validation'], params['greyscale'],
                                                          params['imagenet_path'])
            nb_batches_per_epoch = np.ceil(params['dataset_size'] / params['batch_size'])

        fa_updater = None

        # Initialize metrics
        train_accuracies = []
        train_accuracies_top5 = []
        train_losses = []

        test_accuracies = []
        test_accuracies_top5 = []
        test_losses = []

        # Get train/test acc before training
        if not params['load_epoch'] or params['load_epoch'] == 0:
            train_accuracy, train_accuracy_top5, train_loss = get_accuracy_on_dataset(train_data)
            test_accuracy, test_accuracy_top5, test_loss = get_accuracy_on_dataset(test_data)

            train_accuracies.append(train_accuracy)
            train_accuracies_top5.append(train_accuracy_top5)
            train_losses.append(train_loss)

            test_accuracies.append(test_accuracy)
            test_accuracies_top5.append(test_accuracy_top5)
            test_losses.append(test_loss)

            print('*'*20)
            print(train_accuracy, test_accuracy)
            print('*' * 20)

        cors_per_layer = []
        cors_per_batch = []

        if params['load_epoch'] > 0:
            load_path = os.path.join('results', params['experiment_name'], 'epoch' + str(params['load_epoch']))
            cors_per_layer, test_accuracies, test_accuracies_top5, test_losses, \
            train_accuracies, train_accuracies_top5, train_losses = load_metrics(os.path.join(load_path, 'metrics'))
            load_weights(load_path=os.path.join(load_path, 'weights'),
                         net=net,
                         optimizer=optimizer,
                         input_shape=input_shape)
            train_accuracy, train_accuracy_top5, train_loss, test_accuracy, test_accuracy_top5, test_loss = 0,0,0,0,0,0
            print(f"Weights loaded from {load_path}")

        # Training
        batch_size = params['batch_size']
        print_interval = 100  # After how many batches should we print an update and collect stats
        if params['algorithm'] == 'GAIT':
            update_inverse_model(net)

        # Beginning training
        print(outpath)
        for e_index in range(int(params['load_epoch'])+1, params['nb_epochs']+1):
            start_time = time.time()
            progress(e_index, 0.0, test_accuracy, test_accuracy_top5)
            train_step = 0

            for batch, targets in train_data.shuffle(seed=e_index, buffer_size=1000).batch(params['batch_size']):
                if params['max_steps'] and train_step == params['max_steps']:
                    break

                # Run a forward and inverse pass
                with tf.GradientTape(persistent=True) as tape:
                    outputs = net(batch)
                    targets = tf.cast(targets, tf.float64)
                    y_ = outputs[-1]
                    loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true=targets, y_pred=y_[:, :data_params['output_size']],
                                                                                   from_logits=True))

                # Computing the layer-wise errors and updating weights
                if params['algorithm'] == 'GAIT':
                    update_inverse_model(net)
                    error = tape.gradient(loss, y_)/2
                    targets_gait = y_ - error
                    local_targets, mult_factors = net.inverse_pass(targets_gait, outputs, float(params['gamma']))
                    grads = get_local_updates(net, outputs, local_targets, mult_factors, tape)

                if params['algorithm'] == 'FA':
                    if fa_updater is None:
                        fa_updater = FA_Updater()
                        fa_updater.initialize_random_weights(net)
                    error = tape.gradient(loss, y_)/2
                    grads = fa_updater.get_updates(net=net, outputs=outputs, error=error, tape=tape)

                elif params['algorithm'] == 'BP':
                    grads = tape.gradient(loss, net.trainable_weights)

                # Correlation with BP
                if  train_step % print_interval == 0:
                    grads_bp = tape.gradient(loss, net.trainable_weights)
                    cors_per_layer.append(get_cors_per_layer(grads, grads_bp))
                    cors_per_batch.append(np.mean(cors_per_layer))

                if params['ortho_reg'] > 0:
                    grads = add_ortho_gradients(net, grads, params['ortho_reg'])

                if params['train_network']:
                    optimizer.apply_gradients(zip(grads, net.trainable_weights))

                train_step += 1

            train_accuracy, train_accuracy_top5, train_loss = get_accuracy_on_dataset(train_data)
            test_accuracy, test_accuracy_top5, test_loss = get_accuracy_on_dataset(test_data)

            test_accuracies.append(test_accuracy)
            test_accuracies_top5.append(test_accuracy_top5)
            test_losses.append(test_loss)

            train_accuracies.append(train_accuracy)
            train_accuracies_top5.append(train_accuracy_top5)
            train_losses.append(train_loss)

            print('')
            print('-'*40)
            print("Time taken: :" + str(time.time() - start_time))
            print('')
            print(f'Average train loss: {train_loss}')
            print(f'Average train accuracy: {train_accuracy}')
            print(f'Average train accuracy top5: {train_accuracy_top5}')
            print('')
            print(f'Test loss: {test_loss}')
            print(f'Test accuracy: {test_accuracy}')
            print(f'Test accuracy top5: {test_accuracy_top5}')
            print('-' * 40)

            # Save metrics every epoch
            save_metrics(os.path.join(outpath + f'epoch{e_index}/metrics/'))

            if e_index % 10 == 0:
                # Save weights every 5 epochs (because of disk space)
                save_weights(save_path=os.path.join(outpath + f'epoch{e_index}/weights/'),
                             net=net,
                             optimizer=optimizer)

        save_metrics(outpath)
        save_weights(save_path=os.path.join(outpath + 'final/weights/'),
                     net=net,
                     optimizer=optimizer)
        print("\nTraining Complete!")

