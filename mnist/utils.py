def make_graphs(histories, items, prefix=''):
    from matplotlib import pyplot
    keys = list()
    for key in histories[0].history.keys():
        keys.append(key)
    fig, axs = pyplot.subplots(2, 4, constrained_layout=True, figsize=(18,7))
    fontsize=12
    for index, history in enumerate(histories):
        axs[0][index].set_title(f"Items por clase: {items[index]}")
        axs[0][index].plot(history.history[keys[0]], '--', label='train')
        axs[0][index].plot(history.history[keys[2]], label='test')
        axs[0][0].set_ylabel('Loss', fontsize=fontsize)

        axs[1][index].plot(history.history[keys[1]], '--', label='train')
        axs[1][index].plot(history.history[keys[3]], label='test')
        axs[1][index].set_xlabel('Iteraciones', fontsize=fontsize)
        axs[1][0].set_ylabel('Accuracy', fontsize=fontsize)
    axs[0][0].legend()
    pyplot.savefig(f"{prefix}-metrics.pdf")
    pyplot.close()