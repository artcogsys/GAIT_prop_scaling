# GAIT Propagation Code Instructions
**run.py** will read config/path_to_config_file.yaml and run the experiments defined in it.
Parameters in the config file are detailed below.

Note that due to GAIT-prop's sensitivity to small errors, experiments are run in tf.float64,
significantly slowing down run time.

Long Arguments:

|option|explanation|
|---|---|
|`train_network`|If True a network will be trained, if False, a random network will be initialized every batch|
|`max_steps`|Stop training after X training steps if you just want to test the code quickly|
|`adaptive_gamma`|If True, gamma will be set to normalization_constant / l2norm(error) for every layer for every batch|
|`algorithm`|The training algorithm used. BP,GAIT, and FA supported|
|`seed`|The seed for the random generators which produce network weights|
|`ortho_reg`|The strength of the orthogonal regularizer (lambda)|
|`nb_epochs`|The number of training epochs to run (after which accuracies and network parameters are saved)|
|`learning_rate`|The learning rate to use for the simulation|
|`network`| Network architecturer and dataset. 'CIFAR', 'CIFAR_shallow' and 'ImageNet' supported.
|`activation`| Activation function for all layers except the output layer. 'leaky_relu' and 'linear' supported.
|`imagenet_path`| Path to ImageNet tfrecords.

To run the code simply type 'python run.py -config config/<config_file> -gpu <gpu_number>'.
To add comparative plots over multiple experiments, run 'python plot_multiple_experiments.py'