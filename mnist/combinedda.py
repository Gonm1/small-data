import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tf warnings about cuda
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Input, UpSampling2D
from keras.metrics import CategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
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

from utils import make_graphs, print_to_file, load_mnist_pickle

GLOBAL_EPOCHS = 200

def scheduler(epoch, lr):
    lrmin = 0.00005
    lrmax = 0.0035
    step_size = 10
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
patiences = [60, 40, 40, 40]
batch_sizes = [20, 32, 32, 32]
for index, item in enumerate(items):

    # Load the dataset
    x_train, y_train, x_test, y_test = load_mnist_pickle(id=item)
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

    def add_channel(image):
        img = array_to_img(image, scale=False) #returns PIL Image
        img = img.convert(mode='RGB') #makes 3 channels
        arr = img_to_array(img) #convert back to array
        return arr
    
    x_train *= 255
    x_test *= 255

    x_train = [add_channel(img) for img in x_train]
    x_test = [add_channel(img) for img in x_test]
    x_train = np.asarray(x_train, dtype='float32')
    x_test = np.asarray(x_test, dtype='float32')
    print(x_train.shape)
    print(y_train.shape)

    epochs = GLOBAL_EPOCHS
    learning_rate = 0.001
    patience = patiences[index]
    num_classes = y_test.shape[1]
    # build model
    model = Sequential()
    model.add(Input(shape=(28, 28, 3)))
    model.add(UpSampling2D(size=(8,8), interpolation='nearest'))

    # Load updated weights from official repository
    transfer_learning_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='../cifar10/efficientnet-b0/efficientnetb0_notop.h5')
    for layer in transfer_learning_model.layers:
        layer.trainable = False

    model.add(transfer_learning_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=CosineSimilarity(axis=1), optimizer=Adam(), metrics=[CategoricalAccuracy()])

    if VERBOSE: model.summary()

    datagen = ImageDataGenerator(rotation_range=45, zoom_range=[0.85,1.0], horizontal_flip=False, fill_mode='reflect')

    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=patience, verbose=VERBOSE)

    history = model.fit(datagen.flow(x=x_train, y=y_train, batch_size=batch_sizes[index], seed=seed_value), validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_sizes[index], verbose=VERBOSE, callbacks=[earlyStop, LearningRateScheduler(scheduler)])
    histories.append(history)

    model.save(f"models/combinedda-{item}.h5")

    predictions = model.predict(x_test)
    y_test = np.argmax(y_test, axis=1)
    predictions = np.argmax(predictions, axis=1)
    
    dicts.append(classification_report(y_true=y_test, y_pred=predictions, digits=3, output_dict=True))
    mccs.append(matthews_corrcoef(y_true=y_test, y_pred=predictions))
    last_epochs.append(len(history.history['loss']))

print_to_file(dicts, mccs, items, epochs, batch_sizes, learning_rate, patiences, last_epochs, model, 'combinedda')
make_graphs(histories, items, 'combinedda')