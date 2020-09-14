from tensorflow.python.keras import backend as K
from keras.utils import np_utils
import numpy as np
import random
import pickle
import sys
import cv2
import os

# Functions from keras to load the dataset.
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/datasets/cifar10.py


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    Arguments:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    Returns:
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_full_cifar():
    """Loads [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
    This is a dataset of 50,000 32x32 color training images and 10,000 test
    images, labeled over 10 categories. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        **x_train, x_test**: uint8 arrays of RGB image data with shape
          (num_samples, 3, 32, 32) if the `tf.keras.backend.image_data_format` is
          'channels_first', or (num_samples, 32, 32, 3) if the data format
          is 'channels_last'.
        **y_train, y_test**: uint8 arrays of category labels
          (integers in range 0-9) each with shape (num_samples, 1).
    """
    path = "../Datasets/CIFAR10"
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return x_train, y_train, x_test, y_test


def load_cifar(items_per_class=10, seed=0):
    training_images, training_labels, x_test, y_test = load_full_cifar()
    print(int(training_labels[0]))

    x_train, y_train, indexed = [], [], []
    class_counter = np.zeros(10)
    done = False
    random.seed(seed)
    while not done:
        index = random.randint(0, len(training_labels)-1)
        if class_counter[int(training_labels[index])] < items_per_class and index not in indexed:
            class_counter[int(training_labels[index])] += 1
            y_train.append(int(training_labels[index]))
            x_train.append(training_images[index])
            indexed.append(index)
        if sum(class_counter) == items_per_class * 10:
            done = True

    # Convert lists into numpy arrays
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train, y_train, x_test, y_test


def cifar_preprocess(x_train, y_train, x_test, y_test):

    # Pre-processing
    x_train_grayscale = np.zeros(x_train.shape[:-1])
    for i in range(x_train.shape[0]):
        x_train_grayscale[i] = cv2.cvtColor(x_train[i], cv2.COLOR_BGR2GRAY)

    x_test_grayscale = np.zeros(x_test.shape[:-1])
    for i in range(x_test.shape[0]):
        x_test_grayscale[i] = cv2.cvtColor(x_test[i], cv2.COLOR_BGR2GRAY)

    # Set pixel value to [0,1] (MinMax Normalization)
    x_train_grayscale = x_train_grayscale.astype('float32')
    x_train_grayscale /= 255.0
    x_test_grayscale = x_test_grayscale.astype('float32')
    x_test_grayscale /= 255.0

    # One hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    x_train_grayscale = x_train_grayscale.reshape(
        (len(x_train_grayscale), 32, 32, 1))
    x_test_grayscale = x_test_grayscale.reshape(
        (len(x_test_grayscale), 32, 32, 1))

    return x_train_grayscale, y_train, x_test_grayscale, y_test


if __name__ == "__main__":
    # Load 100 sample training set with 10 images per class.
    x_train, y_train, x_test, y_test = load_cifar()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train, x_test, y_test = cifar_preprocess(x_train, y_train, x_test, y_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
