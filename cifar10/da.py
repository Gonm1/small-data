import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tf warnings about cuda
from utils import make_graphs, print_to_file
from utils import load_cifar_pickle, print_to_file, make_graphs
import sys
import random
import numpy as np
import tensorflow as tf
from pandas import DataFrame
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import classification_report, matthews_corrcoef
from keras.metrics import CategoricalAccuracy
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.optimizer.set_jit(True)

VERBOSE = 1
if not VERBOSE:
    print("Change verbose to 1 to see messages.")

last_epochs = list()
mccs = list()
dicts = list()
histories = list()
items = [10, 50, 250, 500]
patiences = [55, 30, 10, 10]
batch_sizes = [20, 32, 32, 32]
for index, item in enumerate(items):
    # Load the dataset
    x_train, y_train, x_test, y_test = load_cifar_pickle(id=item)
    if VERBOSE: print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    seed_value = 0
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    epochs = 80
    learning_rate = 0.00005
    patience = patiences[index]
    num_classes = y_test.shape[1]
    # build model
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(7, 7), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(filters=128, kernel_size=(7, 7), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(48, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss=CategoricalCrossentropy(), 
        optimizer=Adam(lr=learning_rate), 
        metrics=[CategoricalAccuracy()])

    if VERBOSE:
        model.summary()

    earlyStop = EarlyStopping(
        monitor='val_loss', mode='min', patience=patience, verbose=VERBOSE)

    datagen = ImageDataGenerator(rotation_range=45, zoom_range=[0.85,1.0], horizontal_flip=True, fill_mode='reflect')
    
    history = model.fit(datagen.flow(x=x_train, y=y_train), 
                    validation_data=(x_test, y_test), 
                    epochs=epochs,
                    batch_size=batch_sizes[index], verbose=VERBOSE,
                    callbacks=[earlyStop],
                    validation_batch_size=1000)
    histories.append(history)
    model.save(f"models/da-{item}.h5")

    predictions = model.predict(x_test)
    y_test = np.argmax(y_test, axis=1)
    predictions = np.argmax(predictions, axis=1)

    dicts.append(classification_report(
        y_true=y_test, y_pred=predictions, digits=3, output_dict=True))
    mccs.append(matthews_corrcoef(y_true=y_test, y_pred=predictions))
    last_epochs.append(len(history.history['loss']))

print_to_file(dicts, mccs, items, epochs,
              batch_sizes, learning_rate, patiences, last_epochs, model, 'da')
make_graphs(histories, items, 'da')
