import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tf warnings about cuda
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from sklearn.metrics import classification_report, matthews_corrcoef
from keras.models import load_model
from utils import load_cifar_pickle
from tensorflow import keras
import tensorflow as tf
import numpy as np
import sys
    
x_train, y_train, x_test, y_test = load_cifar_pickle(id=10)

y_test = np.argmax(y_test, axis=1)
x_test_not_normalized = x_test * 255

all_predictions = []
items = [10, 50, 250, 500]
for item in items:
    print(f'Loading {item} items per class models.')

    base = 'models/'
    dnn = load_model(f'{base}dnn-{item}.h5')
    dropout = load_model(f'{base}dropout-{item}.h5')
    gap = load_model(f'{base}gap-{item}.h5')
    bnorm = load_model(f'{base}bnorm-{item}.h5')
    closs = load_model(f'{base}closs-{item}.h5')
    dconv = load_model(f'{base}dconv-{item}.h5')
    clr = load_model(f'{base}clr-{item}.h5')
    combined = load_model(f'{base}combined-{item}.h5')
    combinedda = load_model(f'{base}combinedda-{item}.h5')

    print('Models loaded. Making predictions.')

    models = [dnn, dropout, gap, bnorm, closs, dconv, clr, combined, combinedda]
    predictions = []
    for count, model in enumerate(models):
        if model == combinedda:
            predictions.append(np.argmax(model.predict(x_test_not_normalized), axis=1))
        else:
            predictions.append(np.argmax(model.predict(x_test), axis=1))
        print(f'{count+1}/9')
    predictions = np.array(predictions)
    all_predictions.append(predictions)
    print('Done.')
    print()


print('Writing results to file.')
original_stdout = sys.stdout
with open(f'results/ensemble.txt', 'a') as f:
    sys.stdout = f
    for predictions in all_predictions:
        parcial_predictions = []
        for image in range(10_000):
            aux_list = []
            for model in range(9):
                aux_list.append(predictions[model][image])
            parcial_predictions.append(max(set(aux_list), key=aux_list.count))
        parcial_predictions = np.array(parcial_predictions)
        parcial_predictions.shape
        print(classification_report(y_pred=parcial_predictions, y_true=y_test, digits=3))
        print(f'mcc: {round(matthews_corrcoef(y_pred=parcial_predictions, y_true=y_test), 3)}')
        print('-------------------------------')
        print()
sys.stdout = original_stdout
f.close()