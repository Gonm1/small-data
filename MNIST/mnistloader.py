import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tf warnings about cuda
from keras.utils import np_utils
import numpy as np
import random


def load_full_mnist(prefix):
    "Function from https://stackoverflow.com/a/53226079"
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(prefix + '-images-idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(
        data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(
        dtype='float32').reshape([nImages, width, height])

    labels = np.fromfile(prefix + '-labels-idx1-ubyte',
                         dtype='ubyte')[2 * intType.itemsize:]

    return data, labels


def load_mnist(items_per_class=10, seed=0):
    '''
    Randomly picks

    '''

    # Data loading
    training_images, training_labels = load_full_mnist(
        "../Datasets/MNIST/train")
    x_test, y_test = load_full_mnist("../Datasets/MNIST/t10k")

    # Make subset for training by picking random examples for each class
    x_train, y_train, indexed = [], [], []
    class_counter = np.zeros(10)
    done = False
    random.seed(seed)
    while not done:
        index = random.randint(0, len(training_labels)-1)
        if class_counter[training_labels[index]] < items_per_class and index not in indexed:
            class_counter[training_labels[index]] += 1
            y_train.append(training_labels[index])
            x_train.append(training_images[index])
            indexed.append(index)
        if sum(class_counter) == items_per_class * 10:
            done = True

    # Convert lists into numpy arrays
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train, y_train, x_test, y_test


def mnist_preprocess(x_train, y_train, x_test, y_test):
    '''
    - Convert images from x_train and x_test into minmax normalized images.
    - Convert y_train and y_test targets into one hot encoded arrays.
    '''

    # Pre-processing
    # Set pixel value to [0,1]
    x_train = x_train.astype('float32')
    x_train /= 255.0
    x_test = x_test.astype('float32')
    x_test /= 255.0

    # One hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    # Load 100 sample training set with 10 images per class.
    x, y, xt, yt = load_mnist()
    print(x.shape, y.shape, xt.shape, yt.shape)
    x, y, xt, yt = mnist_preprocess(x, y, xt, yt)
    print(x.shape, y.shape, xt.shape, yt.shape)
