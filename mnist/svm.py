from sklearn import svm, metrics
import numpy as np

def svm_f(x_train, y_train, x_test, y_test, verb=0):
    # Reshape to vector form
    x_train = x_train.reshape(len(x_train), 28*28)
    x_test = x_test.reshape(len(x_test), 28*28)
    if verb: print("Shape after converting to vector", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Revert categorical arrays on labels
    y_train = [np.argmax(label) for label in y_train]
    y_test = [np.argmax(label) for label in y_test]
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if verb: print("Shape after converting labels: ", y_train.shape, y_train.shape)

    # Model definition
    clf = svm.SVC(C=5.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)


    # Model training
    clf.fit(x_train, y_train)

    # Model testing
    predictions = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)

    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    precision = metrics.precision_score(y_true=y_test, y_pred=predictions, average='macro')

    recall = metrics.recall_score(y_true=y_test, y_pred=predictions, average='macro')

    f1 = metrics.f1_score(y_true=y_test, y_pred=predictions, average='macro')

    return round(accuracy,4), round(precision,4), round(recall,4), round(f1,4)


if __name__ == "__main__":
    from mnistloader import load_mnist, mnist_preprocess
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

    items = [10, 50, 250, 500]
    results = np.zeros((4,5))
    for index, item in enumerate(items):
        # Load the dataset
        x_train, y_train, x_test, y_test = load_mnist(items_per_class=item, seed=seed_value) # 10 items per class means a dataset size of 100
        print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        # Pre process images
        x_train, y_train, x_test, y_test = mnist_preprocess(x_train, y_train, x_test, y_test)
        print("Shape after pre processing: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        print(f"Training set size: {len(x_train)}")
        print(f"Test set size: {len(x_test)}")

        print("Training")
        acc, pr, rec, f1 = svm_f(x_train, y_train, x_test, y_test)
        results[index] = item*10, acc, pr, rec, f1
        print("Training complete", end='\n\n')
    
    print('  Size\t     accuracy precision recall  f1')
    np.set_printoptions(suppress=True)
    print(results)
