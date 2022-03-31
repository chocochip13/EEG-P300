import matplotlib.pyplot as plt

import numpy as np
import matplotlib
matplotlib.use('Agg')


def plot_within_aucs(aucpath, filepath, title):
    """
    Plot AUCs
    """
    data_to_plot = []
    aucs = np.zeros((8, 50))
    for i in range(8):
        aucs[i, :] = np.loadtxt(aucpath + '/s' + str(i) + '_auc.npy')
        data_to_plot.append(np.loadtxt(aucpath + '/s' + str(i) + '_auc.npy'))

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot, showmeans=True, meanline=True)
    plt.title(title + "\n Mean AUC:" + str('%.2f' % np.mean(aucs)) + u"\u00B1" + str('%.2f' % np.std(aucs)))
    plt.ylabel("AUC")
    plt.xlabel('Subjects')
    plt.grid(False)
    plt.savefig(filepath)
    return np.mean(aucs), np.std(aucs)


def plot_cross_aucs(aucpath, filepath, title):
    aucs = np.loadtxt(aucpath)

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.bar([1, 2, 3, 4, 5, 6, 7, 8], aucs)
    plt.title(title + "\n Mean AUC:" + str('%.2f' % np.mean(aucs)) + u"\u00B1" + str('%.2f' % np.std(aucs)))
    plt.ylabel("AUC")
    plt.xlabel('Subjects')
    plt.grid(False)
    plt.savefig(filepath)
