import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tf warnings about cuda
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Input, UpSampling2D
from keras.metrics import CategoricalAccuracy
from sklearn.metrics import classification_report, matthews_corrcoef
from keras.losses import CosineSimilarity, CategoricalCrossentropy
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.models import Sequential
from keras.optimizers import Adam
from pandas import DataFrame
import tensorflow as tf
import numpy as np
import random
import sys

from utils import load_cifar_pickle, print_to_file, make_graphs
from utils import make_graphs, print_to_file

GLOBAL_EPOCHS = 350

def scheduler(epoch, lr):
    lrmin = 0.0005
    lrmax = 0.001
    step_size = 20
    max_iter = GLOBAL_EPOCHS
    delta = 10
    clr = lrmin + ((lrmax-lrmin)*(1.-np.fabs((epoch/step_size)-(2*np.floor(epoch/(2*step_size)))-1.)))
    clr_decay = clr/(1.+((delta-1.)*(epoch/max_iter)))
    return clr_decay

VERBOSE = 1
if not VERBOSE: print("Change verbose to 1 to see messages.")

last_epochs = list()
mccs = list()
dicts = list()
histories = list()

items = [10, 50, 250, 500]
patiences = [20, 20, 20, 20]
batch_sizes = [20, 32, 32, 32]
for index, item in enumerate(items):

    # Load the dataset
    x_train, y_train, x_test, y_test = load_cifar_pickle(id=item)
    if VERBOSE: print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    seed_value = 0
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    x_train *= 255
    x_test *= 255

    epochs = GLOBAL_EPOCHS
    learning_rate = 0.0001
    patience = patiences[index]
    num_classes = y_test.shape[1]
    # build model
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    model.add(UpSampling2D(size=(7,7), interpolation='nearest'))

    transfer_learning_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    for layer in transfer_learning_model.layers:
        layer.trainable = False

    model.add(transfer_learning_model)
    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=CosineSimilarity(axis=1), optimizer=Adam(lr=learning_rate), metrics=[CategoricalAccuracy()])

    if VERBOSE: model.summary()
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=patience, verbose=VERBOSE)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_sizes[index], verbose=VERBOSE, callbacks=[earlyStop, LearningRateScheduler(scheduler)], validation_batch_size=1000)
    histories.append(history)
    model.save(f"models/combined-{item}.h5")


    predictions = model.predict(x_test)
    y_test = np.argmax(y_test, axis=1)
    predictions = np.argmax(predictions, axis=1)
    
    dicts.append(classification_report(y_true=y_test, y_pred=predictions, digits=3, output_dict=True))
    mccs.append(matthews_corrcoef(y_true=y_test, y_pred=predictions))
    last_epochs.append(len(history.history['loss']))

print_to_file(dicts, mccs, items, epochs, batch_sizes, learning_rate, patiences, last_epochs, model, 'combined')
make_graphs(histories, items, 'combined')