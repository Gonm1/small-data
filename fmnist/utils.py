def make_graphs(histories, items, prefix=''):
    from matplotlib import pyplot
    keys = list()
    for key in histories[0].history.keys():
        keys.append(key)
    fig, axs = pyplot.subplots(2, 4, constrained_layout=True, figsize=(18,7))
    fontsize=12
    for index, history in enumerate(histories):
        axs[0][index].set_title(f"Items por clase: {items[index]}")
        axs[0][index].plot(history.history[keys[0]], '--', label='Entrenamiento')
        axs[0][index].plot(history.history[keys[2]], label='Prueba')
        axs[0][0].set_ylabel('Perdida/Costo', fontsize=fontsize)

        axs[1][index].plot(history.history[keys[1]], '--', label='Entrenamiento')
        axs[1][index].plot(history.history[keys[3]], label='Prueba')
        axs[1][index].set_xlabel('Iteraciones', fontsize=fontsize)
        axs[1][0].set_ylabel('Exactitud', fontsize=fontsize)
    axs[0][0].legend()
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
            dataFrame.loc['accuracy', 'support'] = 10000
            dataFrame.loc['accuracy','recall'] = '-'
            dataFrame.loc['accuracy','precision'] = '-'
            print(dataFrame)
            print("mcc: ", round(mccs[index],3))
            print(f'last epoch:  {last_epochs[index]}')
            print()
    sys.stdout = original_stdout
    f.close()


def load_fmnist_pickle(id):
    import pickle
    pickle_in = open(f"Dataset/fMNIST-{id}.pickle", "rb")
    x_train, y_train, x_test, y_test = pickle.load(pickle_in)
    pickle_in.close()
    return x_train, y_train, x_test, y_test