import os
from os.path import join
from time import time

import tensorflow as tf
import numpy as np
import argparse

from PIL import Image
from glob import glob

def preprocess_image(img):

    img = np.array(img)

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    # Resize
    img = tf.image.resize(img, [224, 224])

    # Normalize, save as float16 to save diskspace
    img = img.numpy().astype(np.float16)
    img /= 255

    img[:, :, 0] = (img[:, :, 0] - .485) / .229
    img[:, :, 1] = (img[:, :, 1] - .456) / .224
    img[:, :, 2] = (img[:, :, 2] - .406) / .225

    return img.tobytes()

def serialize_example(img, label):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    example = tf.train.Example(features=tf.train.Features(feature={"features": _bytes_feature(img),
                                                                   "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                                                                   }))
    return example.SerializeToString()

def load_class_mapping(path):
    class_mapping = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(' ')[:2]
            class_mapping[key] = int(value)
    return class_mapping

def load_val_ground_truth(path):
    val_ground_truth = []
    with open(path, 'r') as f:
        for line in f.readlines():
            val_ground_truth.append(int(line.split('\n')[0]))
    return val_ground_truth

parser = argparse.ArgumentParser()
parser.add_argument("-datapath", help="Imagenet raw data folder.", default='')
parser.add_argument("-outpath", help="Destination folder for processed data.", default='')
args = parser.parse_args()

data_path = args.datapath
outpath = args.outpath

os.environ["CUDA_VISIBLE_DEVICES"]='-1'

train_data_path = join(data_path, 'Data/CLS-LOC/train')
val_data_path = join(data_path, 'Data/CLS-LOC/val')
val_ground_truth_path = join(data_path, 'devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt')
class_mapping_path = join(data_path, 'devkit/data/map_clsloc.txt')

val_ground_truth = load_val_ground_truth(val_ground_truth_path)
class_mapping = load_class_mapping(class_mapping_path)

num_shards = 500
train_data = glob(train_data_path+'/*')
val_data = glob(val_data_path+'/*')
num_classes = 1000


train_record_writers = []
val_record_writers = []

for i in range(num_shards):
    train_record_writers.append(
        tf.io.TFRecordWriter(join(outpath, 'train', f"tfrecord_{i}")))

for i in range(num_shards):
    val_record_writers.append(
        tf.io.TFRecordWriter(join(outpath, 'val', f"tfrecord_{i}")))

# Write train data
failed = 0
success = 0
start = time()
for index, folder in enumerate(sorted(train_data)):
    folder_name = os.path.basename(os.path.normpath(folder))
    class_number = class_mapping[folder_name]
    print(index, folder_name, class_number)
    file_paths = glob(folder+'/*')

    for file_path in file_paths:

        # Load and preprocess image
        try:
            img = Image.open(file_path)
            img = preprocess_image(img)
        except:
            print(f'Could not process {file_path}')
            img = None

        if img is None:
            failed += 1
            continue

        success += 1

        # Append label
        label = np.zeros(num_classes, dtype=np.int64)
        label[class_number-1] = 1

        # Serialize example
        example = serialize_example(img, label)
        train_record_writers[np.random.randint(0, num_shards - 1)].write(example)


[w.close() for w in train_record_writers]
print(f'Succesfully processed {success} train images. {failed} images failed.')
print(f'Time taken: {time()-start} seconds.')

# Write val data
failed = 0
success = 0
start = time()
for index, file_path in enumerate(sorted(val_data)):
    if index % 1000 == 0:
        print(index)
    # Load and preprocess image
    try:
        img = Image.open(file_path)
        img = preprocess_image(img)
    except:
        print(f'Could not process file {file_path}')
        img = None

    if img is None:
        failed += 1
        continue
    success += 1

    # Append label
    label = np.zeros(num_classes, dtype=np.int64)
    label[val_ground_truth[index]-1] = 1

    # Serialize example
    example = serialize_example(img, label)
    val_record_writers[np.random.randint(0, num_shards - 1)].write(example)

[w.close() for w in val_record_writers]
print(f'Succesfully processed {success} val images. {failed} images failed.')
print(f'Time taken: {time()-start} seconds.')