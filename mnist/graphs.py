from matplotlib import pyplot
import numpy as np

exactitud = 0
precision = 1
recall = 2
f1 = 3
mcc = 4

dt = [[41.8, 60.1, 74, 78.9], [42.5,	60.2,	73.7,	78.6], [41.2,	59.7,	73.7,	78.7], [40.9,	59.6,	73.6,	78.6], [35.5,	55.7,	71.1,	76.6]]
rf = [[74.9, 89, 93.3, 94.6], [74.8,	88.9,	93.4,	94.5], [74.5,	88.8,	93.4,	94.5], [74.1,	88.8,	93.3,	94.5], [72.2,	87.8,	92.7,	94]]
SVM = [[79.2, 90.3, 94.6, 96], [79.2,	90.2,	94.6,	95.9], [78.8,	90.2,	94.6,	95.9], [78.7,	90.2,	94.6,	95.9], [76.9,	89.2,	94,	95.5]]
DNN = [[68.0, 92.0, 96.8, 98.2], [67.7,	92.1,	96.8,	98.2], [67.1,	91.9,	96.8,	98.1], [66.6,	91.9,	96.8,	98.1], [64.6,	91.1,	96.4,	98]]
Dropout = [[74.6, 90.3, 94.4, 97.9], [77.5,	91.6,	97.4,	98], [74.1,	90.1,	97.4,	97.9], [73.7,	90.3,	97.4,	97.9], [72.2,	89.4,	97.1,	97.7]]
GAP = [[80.3, 90.2, 98.2, 98], [81.3,	91.6,	98.2,	98], [80.1,	89.7,	98.2,	98], [80.2,	89.8,	98.2,	98], [78.3,	89.2,	98,	97.8]]
BNORM = [[67.4, 91.4, 94, 95.4], [71.2,	91.6,	94.3,	95.6], [67.1,	91.2,	93.9,	95.5], [67.6,	91.3,	93.9,	95.5], [64.2,	90.4,	93.4,	94.9]]
CosineLoss = [[70.3, 89.4, 96.1, 96.1], [71.5,	89.8,	96.2,	96.2], [69.8,	89.2,	96.1,	96.1], [69.9,	89.3,	96.1,	96.1], [67.1,	88.3,	95.7,	95.7]]
DilatedConv = [[69.3, 88.3, 95.7, 97.2], [69.3,	88.2,	95.7,	97.2], [68.6,	88,	95.6,	97.2], [68.3,	88.1,	95.7,	97.2], [66.1,	87,	95.2,	96.9]]
CLR = [[69.2, 89, 96.7, 97.8], [70.1,	89,	96.7,	97.8], [68.7,	88.9,	96.7,	97.8], [68.3,	88.9,	96.7,	97.8], [66.1,	87.8,	96.4,	97.6]]
Combined = [[80.4, 93.8, 97.3, 97.6], [81.1,	93.9,	97.3,	97.6], [79.9,	93.7,	97.3,	97.6], [79.8,	93.7,	97.3,	97.6], [78.3,	93.1,	97,	97.3]]
Combinedda = [[79.6, 93, 96.7, 98.1], [81.1,	93,	96.7,	98.1], [79.1,	92.9,	96.7,	98.1], [79.1,	92.9,	96.7,	98.1], [77.5,	92.2,	96.3,	97.9]]
Ensemble = [[86.8, 96, 98.6, 99], [87.5,	96,	98.6,	99], [86.3,	95.9,	98.6,	99], [86.2,	95.9,	98.6,	99], [85.4,	95.5,	98.5,	98.9]]

all_data = [dt, rf, SVM, DNN, Dropout, GAP, BNORM, CosineLoss, DilatedConv, CLR, Combined, Combinedda, Ensemble]
names = ['Árbol de decisión', 'Bosque aleatorio', 'Máquina de soporte de vectores', 'Red neuronal Profunda',
         'Dropout', 'Agrupación Promedio Global', 'Normalización por lotes', 'Similitud de coseno', 'Convolución dilatada',
         'Decadencia cíclica de tasa de aprendizaje', 'Combinación', 'Combinación con transferencia de aprendizaje', 'Ensamblaje']

colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4', '#469990', '#dcbeff', '#9A6324',  '#800000', '#000075']
samples = [0, 15, 30, 45]

# Grafico de exactitud
pyplot.rcParams.update({'font.size': 14})

fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Exactitud en MNIST', xlabel='Ejemplos por clase', ylabel='Exactitud')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[exactitud]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.40, 1])
pyplot.yticks(np.arange(0.40, 1.05, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/mnist-exactitud.pdf', bbox_inches = 'tight')

# Grafico de precision
fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Precisión en MNIST', xlabel='Ejemplos por clase', ylabel='Precisión')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[precision]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.40, 1])
pyplot.yticks(np.arange(0.40, 1.05, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/mnist-precision.pdf', bbox_inches = 'tight')

# Grafico de recall
fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Recall en MNIST', xlabel='Ejemplos por clase', ylabel='Recall')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[recall]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.40, 1])
pyplot.yticks(np.arange(0.40, 1.05, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/mnist-recall.pdf', bbox_inches = 'tight')

# Grafico de f1
fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='F1 en MNIST', xlabel='Ejemplos por clase', ylabel='F1')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[f1]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.40, 1])
pyplot.yticks(np.arange(0.40, 1.05, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/mnist-f1.pdf', bbox_inches = 'tight')

# Grafico de mcc
fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Coeficiente de Correlacion de Matthews en MNIST', xlabel='Ejemplos por clase', ylabel='MCC')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[mcc]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.35, 1])
pyplot.yticks(np.arange(0.35, 1.05, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/mnist-mcc.pdf', bbox_inches = 'tight')
