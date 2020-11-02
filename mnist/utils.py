def make_graphs(history, item, prefix=''):
    from matplotlib import pyplot
    keys = list()
    for key in history.history.keys():
        keys.append(key)

    fig, axs = pyplot.subplots(1, 5, constrained_layout=True, figsize=(18,4))
    fontsize=12

    fig.suptitle(f"Metricas sobre un conjunto de entrenamiento de tamaño {item*10}", fontsize=fontsize+2)

    axs[0].plot(history.history[keys[0]], label='train')
    axs[0].plot(history.history[keys[5]], label='test')
    axs[0].set_xlabel('Iteraciones', fontsize=fontsize)
    axs[0].set_ylabel('Loss', fontsize=fontsize)
    
    axs[1].plot(history.history[keys[1]], label='train')
    axs[1].plot(history.history[keys[6]], label='test')
    axs[1].set_xlabel('Iteraciones', fontsize=fontsize)
    axs[1].set_ylabel('Accuracy', fontsize=fontsize)

    axs[2].plot(history.history[keys[2]], label='train')
    axs[2].plot(history.history[keys[7]], label='test')
    axs[2].set_xlabel('Iteraciones', fontsize=fontsize)
    axs[2].set_ylabel('Precision', fontsize=fontsize)
    axs[2].legend()

    axs[3].plot(history.history[keys[3]], label='train')
    axs[3].plot(history.history[keys[8]], label='test')
    axs[3].set_xlabel('Iteraciones', fontsize=fontsize)
    axs[3].set_ylabel('Recall', fontsize=fontsize)

    axs[4].plot(history.history[keys[4]], label='train')
    axs[4].plot(history.history[keys[9]], label='test')
    axs[4].set_xlabel('Iteraciones', fontsize=fontsize)
    axs[4].set_ylabel('F1 Score', fontsize=fontsize)

    pyplot.savefig(f"{prefix}-{item*10}-metrics.pdf")
    pyplot.close()