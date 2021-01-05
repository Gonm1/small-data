import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tf warnings about cuda
import cv2
import numpy as np
from scipy.io import loadmat
from keras.utils import np_utils

def make_graphs(histories, items, prefix=''):
    from matplotlib import pyplot
    keys = list()
    for key in histories[0].history.keys():
        keys.append(key)
    fig, axs = pyplot.subplots(1, 2, constrained_layout=True, figsize=(16,7))
    fontsize=12
    for index, history in enumerate(histories):
        axs[0].set_title(f"Items por clase: {items[0]}")
        axs[0].plot(history.history[keys[0]], '--', label='train')
        axs[0].plot(history.history[keys[2]], label='test')
        axs[0].set_xlabel('Iteraciones', fontsize=fontsize)
        axs[0].set_ylabel('Loss', fontsize=fontsize)

        axs[1].plot(history.history[keys[1]], '--', label='train')
        axs[1].plot(history.history[keys[3]], label='test')
        axs[1].set_xlabel('Iteraciones', fontsize=fontsize)
        axs[1].set_ylabel('Accuracy', fontsize=fontsize)
    axs[0].legend()
    pyplot.savefig(f"results/{prefix}-metrics.pdf")
    pyplot.close()

def print_to_file(dictionaries, mccs, items, epochs, batch_size, learning_rate, patiences, last_epochs, model, prefix=''):
    import sys
    from pandas import DataFrame
    original_stdout = sys.stdout
    with open(f'results/{prefix}.txt', 'a') as f:
        sys.stdout = f
        print(model.summary())
        print(f'batch size: {batch_size}\tlearning rate: {learning_rate}\tmax epochs: {epochs}')
        print()
        for index, dictionary in enumerate(dictionaries):
            print()
            print(f"items/class:  {items[index]}  patience:  {patiences[index]}")
            dataFrame = DataFrame.from_dict(dictionary).T.round(3)
            dataFrame['support'] = dataFrame['support'].astype(int)
            dataFrame.loc['accuracy', 'support'] = 6149
            dataFrame.loc['accuracy','recall'] = '-'
            dataFrame.loc['accuracy','precision'] = '-'
            print(dataFrame)
            print("mcc: ", round(mccs[index],3))
            print(f'last epoch:  {last_epochs[index]}')
            print()
    sys.stdout = original_stdout
    f.close()

def load_class_names():
    # Load class names for the labels
    class_names = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}
    return class_names


def load_flowers(img_size=32):
    # Load images into python numpy arrays
    setid = loadmat('Dataset/setid.mat')

    trainingID = setid['trnid'][0]
    testID = setid['tstid'][0]

    image_list = os.listdir('Dataset/jpg/')

    x_train = np.zeros((len(trainingID), img_size, img_size, 3))
    x_test = np.zeros((len(testID), img_size, img_size, 3))

    for index, id in enumerate(trainingID):
        for img in image_list:
            if str(id).zfill(4) in img:
                x_train[index] = cv2.resize((cv2.imread(
                    f'Dataset/jpg/{img}', cv2.IMREAD_UNCHANGED)), (img_size, img_size), interpolation=cv2.INTER_AREA)

    for index, id in enumerate(testID):
        for img in image_list:
            if str(id).zfill(4) in img:
                x_test[index] = cv2.resize((cv2.imread(
                    f'Dataset/jpg/{img}', cv2.IMREAD_UNCHANGED)), (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Load image labels
    annots = loadmat('Dataset/imagelabels.mat')
    labels = annots['labels'][0]

    y_train = np.zeros(len(trainingID), dtype=int)
    for i, id in enumerate(trainingID):
        y_train[i] = labels[id-1]

    y_test = np.zeros(len(testID), dtype=int)
    for i, id in enumerate(testID):
        y_test[i] = labels[id-1]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, y_train, x_test, y_test


def flowers_preprocess(x_train, y_train, x_test, y_test, img_size=32):
    # Pre-processing
    x_train_grayscale = np.zeros(x_train.shape[:-1])
    for i in range(x_train.shape[0]):
        x_train_grayscale[i] = cv2.cvtColor(x_train[i], cv2.COLOR_BGR2GRAY)

    x_test_grayscale = np.zeros(x_test.shape[:-1])
    for i in range(x_test.shape[0]):
        x_test_grayscale[i] = cv2.cvtColor(x_test[i], cv2.COLOR_BGR2GRAY)

    # Set pixel value to [0,1] (MinMax Normalization)
    x_train_grayscale = x_train_grayscale.astype('float32')
    x_test_grayscale = x_test_grayscale.astype('float32')
    x_train_grayscale /= 255.0
    x_test_grayscale /= 255.0

    # Substract 1 to get 102 categories from categorical type
    # Remeber to add 1 to predictions to get correct label
    y_train = y_train -1
    y_test = y_test -1

    # One hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    x_train_grayscale = x_train_grayscale.reshape(
        (len(x_train_grayscale), img_size, img_size, 1))
    x_test_grayscale = x_test_grayscale.reshape(
        (len(x_test_grayscale), img_size, img_size, 1))

    return x_train_grayscale, y_train, x_test_grayscale, y_test


if __name__ == "__main__":
    SIZE = 32
    x_train, y_train, x_test, y_test = load_flowers(img_size=SIZE)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train, x_test, y_test = flowers_preprocess(x_train, y_train, x_test, y_test, img_size=SIZE)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)