from matplotlib import pyplot
import numpy as np

exactitud = 0
precision = 1
recall = 2
f1 = 3
mcc = 4

dt = [[16.4,	19.2,	22.9,	25.2], [18.4,	19.6,	23.8,	25.4], [16.4,	19.2,	22.9,	25.2], [16.5,	19.3,	21.7,	24.6], [7.1,	10.2,	14.6,	17]]
rf = [[28.4,	33.4,	40.3,	42.5], [29.4,	33.1,	39.6,	41.8], [28.4,	33.4,	40.3,	42.5], [27.2,	32.9,	39.6,	41.7], [20.7,	26.1,	33.7,	36.2]]
SVM = [[26.2,	33.9,	42.2,	45.6], [27.9,	34.8,	42,	45.6], [26.2,	33.9,	42.2,	45.6], [26.2,	34,	41.9,	45.5], [18.1,	26.6,	35.7,	39.6]]
DNN = [[28.5,	35.7,	48.9,	57.5], [28.5,	37.7,	50.4,	57.1], [28.5,	35.7,	48.9,	57.5], [27.8,	35.7,	48.3,	56.9], [20.6,	28.8,	43.5,	52.8]]
Dropout = [[29.4,	38.9,	52.4,	60.4], [30,	41.1,	52.6,	60.4], [29.4,	38.9,	52.4,	60.4], [29.2,	39.3,	51.9,	60.1], [21.6,	32.2,	47.3,	56.1]]
GAP = [[29.3,	39,	49.8,	59], [28.3,	39.1,	48.7,	60.9], [29.3,	39,	49.8,	59], [27.7,	37.1,	47.5,	59], [21.7,	32.6,	44.5,	54.6]]
BNORM = [[29.5,	41,	51.2,	56.9], [30.2,	41.7,	51.9,	58.5], [29.5,	41,	51.2,	56.9], [29.1,	41,	50.7,	57.1], [21.7,	34.5,	46,	52.2]]
CosineLoss = [[26.9,	37.5,	50.8,	57.3], [28.8,	39.5,	50.3,	58], [26.9,	37.5,	50.8,	57.3], [26.2,	37.3,	50.4,	56.9], [19.1,	30.7,	45.4,	52.8]]
DilatedConv = [[26.2,	35,	47.3,	53.1], [27.2,	35.3,	47.8,	53.1], [26.2,	35,	47.3,	53.1], [25.4,	34.1,	46.9,	52.4], [18.4,	28,	41.5,	48]]
CLR = [[28.8,	36,	50.7,	56.2], [28.2,	35.3,	49.8,	56.3], [28.8,	36,	50.7,	56.2], [27,	34.9,	50,	55.9], [21.2,	29.1,	45.3,	51.4]]
Combined = [[29.1,	38.5,	46.9,	54.4], [30.2,	39.4,	47.5,	54.1], [29.1,	38.5,	46.9,	54.4], [29.3,	38.6,	46.6,	54], [21.2,	31.7,	41.2,	49.4]]
Combinedda = [[37.8,	58.4,	72.4,	76.2], [39,	61.7,	72.8,	76.8], [37.8,	58.4,	72.4,	76.2], [35.1,	57.4,	71.7,	75.8], [31.5,	54.4,	69.5,	73.8,]]
Ensemble = [[31.9,	43,	56,	64.2], [31.5,	43.2,	55.8,	63.7], [31.9,	43,	56.7,	64.2], [31,	42.6,	56,	63.9], [24.5,	36.8,	51.9,	60.3,]]

all_data = [dt, rf, SVM, DNN, Dropout, GAP, BNORM, CosineLoss, DilatedConv, CLR, Combined, Combinedda, Ensemble]
#names = ['Árbol de decisión', 'Bosque aleatorio', 'Máquina de soporte de vectores', 'Red neuronal profunda',
         #'Dropout', 'Agrupación de promedio global', 'Normalización por lotes', 'Similitud de coseno', 'Convolución dilatada',
         #'Decadencia cíclica de tasa de aprendizaje', 'Combinación de técnicas (C1)', 'Combinación de técnicas (C2)', 'Ensamblaje de múltiples modelos']

names = ['Decision tree', 'Random forest', 'Support vector machine', 'Deep neural network', 'Dropout', 'Global average pooling', 'Batch Normalization', 'Cosine similarity', 'Dilated convolution', 'Cyclic learning rate decay', 'Combination of techniques (C1)', 'Combination of techniques (C2)', 'Multi-model ensemble']

colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4', '#469990', '#dcbeff', '#9A6324',  '#800000', '#000075']
samples = [0, 15, 30, 45]

# Grafico de exactitud
pyplot.rcParams.update({'font.size': 15})

fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Accuracy for CIFAR-10', xlabel='Examples per class', ylabel='Accuracy')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [ x/100 for x in data[exactitud]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.15, 0.80])
pyplot.yticks(np.arange(0.15, 0.85, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
#pyplot.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=3)
pyplot.savefig('results/english-cifar-exactitud.pdf', bbox_inches = 'tight')

# Grafico de precision
fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Precision for CIFAR-10', xlabel='Examples per class', ylabel='Precision')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [ x/100 for x in data[precision]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.15, 0.80])
pyplot.yticks(np.arange(0.15, 0.85, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
#pyplot.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=3)
pyplot.savefig('results/english-cifar-precision.pdf', bbox_inches = 'tight')

# Grafico de recall
fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Recall for CIFAR-10', xlabel='Examples per class', ylabel='Recall')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [ x/100 for x in data[recall]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.15, 0.80])
pyplot.yticks(np.arange(0.15, 0.85, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
#pyplot.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=3)
pyplot.savefig('results/english-cifar-recall.pdf', bbox_inches = 'tight')

# Grafico de f1
fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='F1 for CIFAR-10', xlabel='Examples per class', ylabel='F1')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [ x/100 for x in data[f1]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.15, 0.80])
pyplot.yticks(np.arange(0.15, 0.85, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
#pyplot.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=3)
pyplot.savefig('results/english-cifar-f1.pdf', bbox_inches = 'tight')

# Grafico de mcc
fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Matthews Correlation Coefficient for CIFAR-10', xlabel='Examples per class', ylabel='MCC')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [ x/100 for x in data[mcc]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.05, 0.75])
pyplot.yticks(np.arange(0.05, 0.80, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=3)
pyplot.savefig('results/english-cifar-mcc.pdf', bbox_inches = 'tight')
