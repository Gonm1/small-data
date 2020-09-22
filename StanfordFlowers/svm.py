from flowersloader import flowers_preprocess, load_flowers
from sklearn import svm, metrics
import numpy as np

# Load the dataset
x_train, y_train, x_test, y_test = load_flowers()
print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Pre process images
x_train, y_train, x_test, y_test = flowers_preprocess(x_train, y_train, x_test, y_test)
print("Shape after pre processing: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Reshape to vector form
x_train = x_train.reshape(len(x_train), 100*100)
x_test = x_test.reshape(len(x_test), 100*100)
print("Shape after converting to vector", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

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
print("Accuracy: ", accuracy)

# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
precision = metrics.precision_score(y_true=y_test, y_pred=predictions, average=None, zero_division=0)
print("Precision: ", precision)

recall = metrics.recall_score(y_true=y_test, y_pred=predictions, average=None)
print("Recall: ", recall)

f1 = metrics.f1_score(y_true=y_test, y_pred=predictions, average=None)
print("F1: ", f1)
