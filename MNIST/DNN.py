from mnistloader import load_mnist, mnist_preprocess
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.metrics import Precision, Recall, CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

def F1Measure(y_true, y_pred): #taken from old keras source code
    "Function from https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d"
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Load the dataset
x_train, y_train, x_test, y_test = load_mnist()
print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Pre process images
x_train, y_train, x_test, y_test = mnist_preprocess(x_train, y_train, x_test, y_test)
print("Shape after pre processing: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

num_classes = y_test.shape[1]

print(f"Training set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}", end='\n\n')

# build model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(7,7), input_shape=(28, 28, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(128,(7,7), activation='relu', padding='same'))
model.add(Conv2D(64,(5,5), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(lr=0.001), metrics=[CategoricalAccuracy(), Precision(), Recall(), F1Measure])

model.summary()

model.fit(x_train, y_train, epochs=8, batch_size=32)

loss, accuracy, precision, recall, f1 = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print(f"val_loss: {loss}, val_accuracy: {accuracy}, val_precision: {precision}, val_recall: {recall}, val_f1: {f1}")