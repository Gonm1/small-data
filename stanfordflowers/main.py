from flowersloader import load_flowers, flowers_preprocess
import tensorflow as tf
import numpy as np
import random
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tf warnings about cuda

seed_value = 0
def set_seed(s=0):
    "https://stackoverflow.com/a/52897216/9082357"
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)


SIZE = 50
set_seed(seed_value)
# Load the dataset
print("Loading dataset")
try:
    pickle_in = open("Dataset/LoadedDataset.pickle", "rb")
    x_train, y_train, x_test, y_test = pickle.load(pickle_in)
    pickle_in.close()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
except (OSError, IOError) as e:
    x_train, y_train, x_test, y_test = load_flowers(img_size=SIZE)
    x_train, y_train, x_test, y_test = flowers_preprocess(x_train, y_train, x_test, y_test, img_size=SIZE)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    pickle_out = open("Dataset/LoadedDataset.pickle", "wb")
    pickle.dump((x_train, y_train, x_test, y_test), pickle_out)
    pickle_out.close()

print(f"Training set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}", end='\n\n')

try:
    from svm import svm_f
    set_seed(seed_value)
    print("default svm run")
    results = svm_f(x_train, y_train, x_test, y_test, img_size=SIZE)
    print(results)
except:
    print("svm module not loaded")

epochs = 25
batch_size = 64

try:
    from dnn import dnn
    set_seed(seed_value)
    print("normal dnn run")
    # epochs = 10
    # batch_size = 32
    results = dnn(x_train, y_train, x_test, y_test, epochs, batch_size, img_size=SIZE)
    print(results)
except:
    print("standard dnn module not loaded")


try:
    from dropout import dropout
    set_seed(seed_value)
    print("dropout dnn run")
    # epochs = 25
    # batch_size = 32
    results = dropout(x_train, y_train, x_test, y_test, epochs, batch_size, img_size=SIZE)
    print(results)
except:
    print("dropout module not loaded.")

try:
    from gap import gap
    set_seed(seed_value)
    print("gap dnn run")
    # epochs = 25
    # batch_size = 32
    results = gap(x_train, y_train, x_test, y_test, epochs, batch_size, img_size=SIZE)
    print(results)
except:
    print("gap module not loaded.")

try:
    from bnorm import bnorm
    set_seed(seed_value)
    print("bnorm dnn run")
    # epochs = 25
    # batch_size = 16
    results = bnorm(x_train, y_train, x_test, y_test, epochs, batch_size, img_size=SIZE)
    print(results)
except:
    print("bnorm module not loaded.")

try:
    from closs import closs
    set_seed(seed_value)
    print("closs dnn run")
    # epochs = 25
    # batch_size = 64
    results = closs(x_train, y_train, x_test, y_test, epochs, batch_size, img_size=SIZE)
    print(results)
except:
    print("closs module not loaded.")

try:
    from dconv import dconv
    set_seed(seed_value)
    print("dconv dnn run")
    # epochs = 25
    # batch_size = 32
    results = dconv(x_train, y_train, x_test, y_test, epochs, batch_size, img_size=SIZE)
    print(results)
except:
    print("dconv module not loaded.")

try:
    from clr import clr
    set_seed(seed_value)
    print("clr dnn run")
    # epochs = 25
    # batch_size = 32
    results = clr(x_train, y_train, x_test, y_test, epochs, batch_size, img_size=SIZE)
    print(results)
except:
    print("clr module not loaded.")

