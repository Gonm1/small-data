from sklearn.metrics import classification_report, matthews_corrcoef
from mnistloader import load_mnist, mnist_preprocess
from sklearn import svm
import tensorflow as tf
import numpy as np
import random
import os


VERBOSE = 0
items = [10, 50, 250, 500]
for item in items:
    seed_value = 0
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    # Load the dataset
    x_train, y_train, x_test, y_test = load_mnist(items_per_class=item, seed=seed_value) # 10 items per class means a dataset size of 100
    if VERBOSE: print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Pre process images
    x_train, y_train, x_test, y_test = mnist_preprocess(x_train, y_train, x_test, y_test)
    if VERBOSE: print("Shape after pre processing: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    if VERBOSE: print(f"Training set size: {len(x_train)}")
    if VERBOSE: rint(f"Test set size: {len(x_test)}")

    # Reshape to vector form
    x_train = x_train.reshape(len(x_train), 28*28)
    x_test = x_test.reshape(len(x_test), 28*28)
    if VERBOSE: print("Shape after converting to vector", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Revert categorical arrays on labels
    y_train = [np.argmax(label) for label in y_train]
    y_test = [np.argmax(label) for label in y_test]
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if VERBOSE: print("Shape after converting labels: ", y_train.shape, y_train.shape)

    # Model definition
    clf = svm.SVC(C=5.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)

    if VERBOSE: print("Training")
    # Model training
    clf.fit(x_train, y_train)

    # Model testing
    predictions = clf.predict(x_test)

    if VERBOSE: print("Training complete", end='\n\n')
    print("items/class: ", item)
    print(classification_report(y_true=y_test, y_pred=predictions, digits=3))
    print("MCC: ", matthews_corrcoef(y_true=y_test, y_pred=predictions))
    print('--------------------------------------------------')
