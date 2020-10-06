from sklearn import svm, metrics
import numpy as np


def svm_f(x_train, y_train, x_test, y_test, verb=0, img_size=100):
    # Reshape to vector form
    x_train = x_train.reshape(len(x_train), img_size*img_size)
    x_test = x_test.reshape(len(x_test), img_size*img_size)
    print("Shape after converting to vector", x_train.shape,
          y_train.shape, x_test.shape, y_test.shape)

    # Revert categorical arrays on labels
    y_train = [np.argmax(label) for label in y_train]
    y_test = [np.argmax(label) for label in y_test]
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    print("Shape after converting labels: ", y_train.shape, y_train.shape)

    # Model definition
    clf = svm.SVC()

    # Model training
    clf.fit(x_train, y_train)
    print("Trainig complete.")

    # Model testing
    predictions = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)

    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    precision = metrics.precision_score(
        y_true=y_test, y_pred=predictions, average='micro')

    recall = metrics.recall_score(
        y_true=y_test, y_pred=predictions, average='micro')

    f1 = metrics.f1_score(y_true=y_test, y_pred=predictions, average='micro')

    return f"val_accuracy: {round(accuracy,4)}\nval_precision: {round(precision,4)}\nval_recall{round(recall,4)}\nval_f1: {round(f1,4)}\n"


if __name__ == "__main__":
    from flowersloader import flowers_preprocess, load_flowers
    import tensorflow as tf
    import numpy as np
    import random
    import os

    seed_value = 0
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    SIZE = 50
    # Load the dataset
    x_train, y_train, x_test, y_test = load_flowers(img_size=SIZE)
    print("Shape after loading: ", x_train.shape,
          y_train.shape, x_test.shape, y_test.shape)

    # Pre process images
    x_train, y_train, x_test, y_test = flowers_preprocess(
        x_train, y_train, x_test, y_test, img_size=SIZE)
    print("Shape after pre processing: ", x_train.shape,
          y_train.shape, x_test.shape, y_test.shape)

    print(f"Training set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}", end='\n\n')

    print(svm_f(x_train, y_train, x_test, y_test, img_size=SIZE))
