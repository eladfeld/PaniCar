from matplotlib import pyplot as plt
import pickle
import numpy as np


models = [ 'half and half 50', 'half and half 100', 'half and half 150', 'half and half 200','denoising', 'denoising + half and half 100']
mins = [ 0.197, 0.224, 0.178, 0.204, 0.592, 0.644]
maxs = [ 0.885, 0.922, 0.887, 0.891, 0.821, 0.946]
avgs = [ 0.535, 0.589, 0.531, 0.552, 0.714, 0.802]

barWidth = 0.5
fig = plt.subplots(figsize=(12, 8))



br1 = np.arange(len(maxs)) + barWidth

difs = np.array(maxs) - np.array(mins)
# Make the plot

def range_graph():
    plt.bar(br1, difs, width=barWidth,
            edgecolor='grey', bottom=mins)

    # Adding Xticks
    plt.xlabel('model', fontweight='bold', fontsize=15)
    plt.ylabel('Confidence Score', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(models))],
               models)
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f'Confidence range per model', fontdict={'fontsize': 28})
    plt.show()


def avarage_graph():
    plt.bar(br1, avgs, width=barWidth,
            edgecolor='grey')

    # Adding Xticks
    plt.xlabel('model', fontweight='bold', fontsize=15)
    plt.ylabel('Confidence Score', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(models))],
               models)
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f'Confidence average per model', fontdict={'fontsize': 28})
    plt.show()

avarage_graph()