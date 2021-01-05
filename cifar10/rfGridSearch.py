from sklearn.metrics import classification_report, matthews_corrcoef
from utils import load_cifar_pickle
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
import random
import sys
import os

VERBOSE = 1
items = [10, 50, 250, 500]
for item in items:

    # Load the dataset
    x_train, y_train, x_test, y_test = load_cifar_pickle(id=item)
    if VERBOSE: print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Reshape to vector form
    x_train = x_train.reshape(len(x_train), 32*32)
    x_test = x_test.reshape(len(x_test), 32*32)
    if VERBOSE: print("Shape after converting to vector", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Revert categorical arrays on labels
    y_train = [np.argmax(label) for label in y_train]
    y_test = [np.argmax(label) for label in y_test]
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if VERBOSE: print("Shape after converting labels: ", y_train.shape, y_train.shape)

    seed_value = 0
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    from sklearn.model_selection import GridSearchCV 

    param_grid = {'criterion' : ['gini', 'entropy'],
                'min_samples_split' : [2,3,4,5,6],
                'min_samples_leaf': [1,2,3,5,7],
                'n_estimators' : [1,5,25,50,100,250,500, 2500, 5000],
                'random_state': [seed_value]}

    grid = GridSearchCV(RandomForestClassifier(), param_grid, verbose = VERBOSE, n_jobs=-1) 

    # fitting the model for grid search 
    grid.fit(x_train, y_train)

    print("Grid search finished")

    original_stdout = sys.stdout
    with open(f'results/rfGridSearch.txt', 'a') as f:
        sys.stdout = f
        # print best parameter after tuning 
        print(f"{item} - {grid.best_params_}") 
    sys.stdout = original_stdout
    f.close()