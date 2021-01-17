import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tf warnings about cuda
from tensorflow.python.keras import backend as K
from keras.utils import np_utils
import numpy as np
import random
import pickle
import sys
import cv2

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    Arguments:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    Returns:
        A tuple `(data, labels)`.
    Function from keras to load the dataset.
    https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/datasets/cifar10.py
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

    Function from keras to load the dataset.
    https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/datasets/cifar10.py
    """
    path = "Dataset/"
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


def load_cifar(items_per_class=10):
    '''
    Randomly picks "items_per_class" items from each class to form the training set.
    '''
    seed = 123456789
    random.seed(seed)
    training_images, training_labels, x_test, y_test = load_full_cifar()

    x_train, y_train, indexed = [], [], []
    class_counter = np.zeros(10)
    done = False
    
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
    # Set pixel value to [0,1] (MinMax Normalization)
    x_train = x_train.astype('float32')
    x_train /= 255.0
    x_test = x_test.astype('float32')
    x_test /= 255.0

    # One hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test
    


if __name__ == "__main__":
    import pickle
    items = [10, 50, 250, 500]
    for item in items:
        x_train, y_train, x_test, y_test = load_cifar(items_per_class=item)
        x_train, y_train, x_test, y_test = cifar_preprocess(x_train, y_train, x_test, y_test)
        pickle_out = open(f"Dataset/CIFAR10-{item}.pickle", "wb")
        pickle.dump((x_train, y_train, x_test, y_test), pickle_out)
        pickle_out.close()