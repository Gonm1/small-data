from mnistloader import load_mnist, mnist_preprocess
import tensorflow as tf
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tf warnings about cuda

from svm import svm_f
from dnn import dnn
from dropout import dropout
from gap import gap
from bnorm import bnorm
from closs import closs
from dconv import dconv

seed_value = 0
def set_seed(s=0):
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

set_seed(seed_value)
# Load the dataset
x_train, y_train, x_test, y_test = load_mnist(items_per_class=10, seed=seed_value) # 10 items per class means a dataset size of 100
print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Pre process images
x_train, y_train, x_test, y_test = mnist_preprocess(x_train, y_train, x_test, y_test)
print("Shape after pre processing: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

print(f"Training set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}", end='\n\n')

set_seed(seed_value)
print("default svm run")
results = svm_f(x_train, y_train, x_test, y_test)
print(results)

epochs = 25
batch_size = 64

set_seed(seed_value)
print("normal dnn run")
# epochs = 10
# batch_size = 32
results = dnn(x_train, y_train, x_test, y_test, epochs, batch_size)
print(results)

set_seed(seed_value)
print("dropout dnn run")
# epochs = 25
# batch_size = 32
results = dropout(x_train, y_train, x_test, y_test, epochs, batch_size)
print(results)

set_seed(seed_value)
print("gap dnn run")
# epochs = 25
# batch_size = 32
results = gap(x_train, y_train, x_test, y_test, epochs, batch_size)
print(results)

set_seed(seed_value)
print("bnorm dnn run")
# epochs = 25
# batch_size = 16
results = bnorm(x_train, y_train, x_test, y_test, epochs, batch_size)
print(results)

set_seed(seed_value)
print("closs dnn run")
# epochs = 25
# batch_size = 64
results = closs(x_train, y_train, x_test, y_test, epochs, batch_size)
print(results)

set_seed(seed_value)
print("dconv dnn run")
# epochs = 25
# batch_size = 32
results = dconv(x_train, y_train, x_test, y_test, epochs, batch_size)
print(results)

set_seed(seed_value)
print("CLR dnn run" + " pending")
# epochs = 25
# batch_size = 32
#results = dconv(x_train, y_train, x_test, y_test, epochs, batch_size)
#print(results)

