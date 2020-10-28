import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tf warnings about cuda
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.metrics import Precision, Recall, CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score

def dnn(x_train, y_train, x_test, y_test, ep, bs, verb=0):
    num_classes = y_test.shape[1]
    # build model
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(7,7), input_shape=(28, 28, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4,4), padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(5,5), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(7,7), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(5,5), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(units = 64, activation='relu'))
    model.add(Dense(units = 32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(lr=0.001), metrics=[CategoricalAccuracy(), Precision(), Recall(), F1Score(num_classes=num_classes, average='macro')])

    if verb: model.summary()

    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=verb)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=ep, batch_size=bs, verbose=verb, callbacks=[earlyStop])

    loss, accuracy, precision, recall, f1 = model.evaluate(x_test, y_test, batch_size=bs, verbose=verb)
    return history, round(loss,4), round(accuracy,4), round(precision,4), round(recall,4), round(f1,4)

if __name__ == "__main__":
    from mnistloader import load_mnist, mnist_preprocess
    from utils import make_graphs
    import tensorflow as tf
    import numpy as np
    import random

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
    results = np.zeros((4,6))
    for index, item in enumerate(items):

        # Load the dataset
        x_train, y_train, x_test, y_test = load_mnist(items_per_class=item, seed=seed_value) # 10 items per class means a dataset size of 100
        print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        # Pre process images
        x_train, y_train, x_test, y_test = mnist_preprocess(x_train, y_train, x_test, y_test)
        print("Shape after pre processing: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        
        print(f"Training set size: {len(x_train)}")
        print(f"Test set size: {len(x_test)}", end='\n\n')

        epochs = 30
        batch_size = 32
        history, loss, accuracy, precision, recall, f1 = dnn(x_train, y_train, x_test, y_test, epochs, batch_size, verb=1)
        results[index] = item*10, loss, accuracy, precision, recall, f1

        make_graphs(history, item, 'dnn')

    np.set_printoptions(suppress=True)
    print(results)