import numpy as np


if __name__ == "__main__":
    galaxy_labels = np.load('../data/result/galaxy_labels.npy')
    galaxy_labels2 = np.load('../data/result/galaxy_labels2.npy')
    comp_table = np.zeros((galaxy_labels.max(), len(set(galaxy_labels2))))

    for l, l2 in zip(galaxy_labels, galaxy_labels2):
        comp_table[l-1, l2+1] += 1

    np.save('../data/result/comp_table.npy', comp_table)