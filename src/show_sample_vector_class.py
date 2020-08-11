import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


if __name__ == "__main__":
    
    sample_vectors = np.load('../data/result/patch_features.npy')
    patch_labels = np.load('../data/result/patch_labels.npy')
    galaxy_labels = np.load('../data/result/galaxy_labels.npy') 

    class_num = galaxy_labels.max()

    for i in range(1, class_num+1):
        idxs1 = np.where(galaxy_labels == i)[0]
        idxs1 += 1
        sv_sum = 0

        # プロット
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(sample_vectors[:, 0], sample_vectors[:, 1], sample_vectors[:, 2], s=5, alpha=0.03, zorder=1)

        for idx in idxs1:
            idxs2 = np.where(patch_labels == idx)[0]
            v = sample_vectors[idxs2]
            sv_sum += len(v)
            ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=5, zorder=2, c='r')

        ax.set_title('cluster{}\n galaxy(or star):{}/{}, sample vector:{}/{}'.format(i, len(idxs1), len(galaxy_labels), sv_sum, len(sample_vectors)))
        plt.tight_layout()
        fig.show()
        fig.savefig('../reports/sample-vector/' + str(i))
        plt.close(fig)