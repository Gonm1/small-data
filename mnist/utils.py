def make_graphs(history, item, prefix=''):
    from matplotlib import pyplot
    keys = list()
    for key in history.history.keys():
        keys.append(key)

    _, axs = pyplot.subplots(1, 5, constrained_layout=True, figsize=(20,5))
    fontsize=14

    axs[0].plot(history.history[keys[0]], label='train')
    axs[0].plot(history.history[keys[5]], label='test')
    axs[0].set_xlabel('Epochs', fontsize=fontsize-2)
    axs[0].set_ylabel('Loss', fontsize=fontsize-2)
    axs[0].set_title(f"Categorical loss with {item*10} examples", fontsize=fontsize)
    
    axs[1].plot(history.history[keys[1]], label='train')
    axs[1].plot(history.history[keys[6]], label='test')
    axs[1].set_xlabel('Epochs', fontsize=fontsize-2)
    axs[1].set_ylabel('Accuracy', fontsize=fontsize-2)
    axs[1].set_title(f"Categorical Accuracy with {item*10} examples", fontsize=fontsize)

    axs[2].plot(history.history[keys[2]], label='train')
    axs[2].plot(history.history[keys[7]], label='test')
    axs[2].set_xlabel('Epochs', fontsize=fontsize-2)
    axs[2].set_ylabel('Precision', fontsize=fontsize-2)
    axs[2].set_title(f"Precision with {item*10} examples", fontsize=fontsize)
    axs[2].legend()

    axs[3].plot(history.history[keys[3]], label='train')
    axs[3].plot(history.history[keys[8]], label='test')
    axs[3].set_xlabel('Epochs', fontsize=fontsize-2)
    axs[3].set_ylabel('Recall', fontsize=fontsize-2)
    axs[3].set_title(f"Recall with {item*10} examples", fontsize=fontsize)

    axs[4].plot(history.history[keys[4]], label='train')
    axs[4].plot(history.history[keys[9]], label='test')
    axs[4].set_xlabel('Epochs', fontsize=fontsize-2)
    axs[4].set_ylabel('F1Measure', fontsize=fontsize-2)
    axs[4].set_title(f"F1Measure with {item*10} examples", fontsize=fontsize)

    pyplot.savefig(f"{prefix}-{item*10}-metrics.pdf")
    pyplot.close()

def F1Measure(y_true, y_pred): #taken from old keras source code
    from keras import backend as K
    "Function from https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d"
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val