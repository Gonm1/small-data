from mnistloader import load_mnist, mnist_preprocess
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, Accuracy

# Load the dataset
x_train, y_train, x_test, y_test = load_mnist()
print("Shape after loading: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Pre process images
x_train, y_train, x_test, y_test = mnist_preprocess(x_train, y_train, x_test, y_test)
print("Shape after pre processing: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

num_classes = y_test.shape[1]

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

opt = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[Accuracy(), Precision(), Recall()])

model.summary()

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=4)

loss, accuracy, precision, recall = model.evaluate(x_test,y_test)