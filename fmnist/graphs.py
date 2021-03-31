from matplotlib import pyplot
import numpy as np

exactitud = 0
precision = 1
recall = 2
f1 = 3
mcc = 4

dt = [[46.3,	64.9,	72.9,	75.6], [48.4,	61.8,	73.1,	75.5], [46.3,	64.9,	72.9,	75.6], [46.2,	61.8,	72.8,	75.4], [40.6,	61.7,	69.9,	72.9]]
rf = [[67.5,	77.8,	82.2,	83.7], [67.4,	77.4,	82,	83.5], [67.5,	77.8,	82.2,	83.7], [67.3,	77.5,	82,	83.4], [63.9,	75.4,	80.3,	81.9]]
SVM = [[68.5,	79.8,	83.7,	85.1], [68.5,	79.9,	83.7,	85], [68.5,	79.8,	83.7,	85.1], [68.2,	79.8,	83.7,	85], [65,	77.5,	81.9,	83.5]]
DNN = [[62.4,	76.2,	80.2,	85.4], [63.3,	78.0,	83.8,	85.4], [62.4,	76.2,	80.2,	85.4], [62.2,	75.4,	80.3,	85.4], [58.4,	73.9,	78.4,	83.8]]
Dropout = [[61.7,	71.8,	84.4,	87], [61.9,	73.2,	84.1,	87.2], [61.7,	71.8,	84.4,	87], [59.2,	71.8,	84.1,	87], [58,	68.8,	82.7,	85.6]]
GAP = [[68.3,	79.2,	85.4,	84.6], [68.1,	79.6,	85.7,	85.8], [68.3,	79.2,	85.4,	84.6], [67.9,	79,	85.3,	84.5], [64.9,	77.1,	83.8,	83]]
BNORM = [[61.3,	78.1,	81.6,	86.6], [63,	79.4,	82.8,	86.8], [61.3,	78.1,	81.6,	86.6], [61.1,	77,	81.5,	86.6], [57.3,	76.2,	79.7,	85.1]]
CosineLoss = [[63.1,	74.3,	80.3,	85.5], [64.3,	74.6,	81.1,	86.2], [63.1,	74.3,	80.3,	85.5], [62.9,	73.5,	80.3,	85.6], [59.1,	71.7,	78.2,	83.9]]
DilatedConv = [[65.3,	78,	83.7,	85.6], [63.9,	78,	84,	85.4], [65.3,	78,	83.7,	85.6], [63.8,	77.7,	83.6,	85.4], [61.7,	75.7,	81.9,	84]]
CLR = [[63,	73.8,	82.7,	85.1], [63.3,	74,	83.5,	85.1], [63,	73.8,	82.7,	85.1], [62.9,	73.4,	82.9,	85], [58.9,	71,	80.8,	83.4]]
Combined = [[67.6,	78.1,	81.3,	84.3], [68.3,	78.9,	82.2,	85.1], [67.6,	78.1,	81.3,	84.3], [67.8,	78.3,	81.2,	84.5], [64.1,	75.7,	79.3,	82.7]]
Combinedda = [[65.5,	78.4,	83.4,	84.3], [68.3,	80.3,	84.3,	85.4], [65.5,	78.4,	83.4,	84.3], [64.4,	78.6,	83.3,	84.5], [62.3,	76.1,	81.7,	82.7]]
Ensemble = [[70.2,	81.9,	87.4,	89], [69.9,	81.4,	87.4,	89], [70.2,	81.9,	87.4,	89], [69.5,	81.5,	87.4,	89], [67,	79.9,	86,	87.8]]

all_data = [dt, rf, SVM, DNN, Dropout, GAP, BNORM, CosineLoss, DilatedConv, CLR, Combined, Combinedda, Ensemble]
names = ['Árbol de decisión', 'Bosque aleatorio', 'Máquina de soporte de vectores', 'Red neuronal profunda',
         'Dropout', 'Agrupación de promedio global', 'Normalización por lotes', 'Similitud de coseno', 'Convolución dilatada',
         'Decadencia cíclica de tasa de aprendizaje', 'Combinación de técnicas (C1)', 'Combinación de técnicas (C2)', 'Ensamblaje de múltiples modelos']



colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4', '#469990', '#dcbeff', '#9A6324',  '#800000', '#000075']
samples = [0, 15, 30, 45]

pyplot.rcParams.update({'font.size': 15})

fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Exactitud en Fashion MNIST', xlabel='Ejemplos por clase', ylabel='Exactitud')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[exactitud]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.45, 0.90])
pyplot.yticks(np.arange(0.45, 0.95, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/fmnist-exactitud.pdf', bbox_inches = 'tight')



fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Precisión en Fashion MNIST', xlabel='Ejemplos por clase', ylabel='Precisión')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[precision]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.45, 0.90])
pyplot.yticks(np.arange(0.45, 0.95, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/fmnist-precision.pdf', bbox_inches = 'tight')



fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Recall en Fashion MNIST', xlabel='Ejemplos por clase', ylabel='Recall')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[recall]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.45, 0.90])
pyplot.yticks(np.arange(0.45, 0.95, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/fmnist-recall.pdf', bbox_inches = 'tight')



fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='F1 en Fashion MNIST', xlabel='Ejemplos por clase', ylabel='F1')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[f1]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.45, 0.90])
pyplot.yticks(np.arange(0.45, 0.95, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/fmnist-f1.pdf', bbox_inches = 'tight')



fig, ax = pyplot.subplots(figsize=(15,10))
ax.set(title='Coeficiente de Correlacion de Matthews en Fashion MNIST', xlabel='Ejemplos por clase', ylabel='MCC')
for index, data in enumerate(all_data):
    ax.bar([x+index for x in samples], [x/100 for x in data[mcc]], label=names[index], color=colors[index])
pyplot.grid(axis='y')
pyplot.ylim([0.40, 0.90])
pyplot.yticks(np.arange(0.40, 0.95, step=0.05))
pyplot.xticks([6.5, 21.5, 36.5, 51.5], ['10', '50', '250', '500'])
pyplot.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
pyplot.savefig('results/fmnist-mcc.pdf', bbox_inches = 'tight')