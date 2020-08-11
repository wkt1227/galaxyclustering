import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, centroid
from scipy.spatial.distance import pdist
from libs.funcs import download_sdss_img


if __name__ == "__main__":
    
    galaxy_vectors = np.load('../data/result/galaxy_vectors.npy')
    galaxy_skycrds = np.load('../data/result/galaxy_skycrds.npy')

    # 階層的クラスタリング
    Z = centroid(pdist(galaxy_vectors, 'correlation'))
    maxclust = 40
    ct = Z[-(maxclust-1), 2]
    cl = fcluster(Z, maxclust, criterion='maxclust')
    plt.axhline(ct, linestyle='--', c='purple')
    dendrogram(Z, color_threshold=ct, labels=cl)
    plt.title('cluster:{}'.format(cl.max()))
    plt.savefig('../reports/hc2_result')
    plt.show()

    np.save('../data/result/galaxy_labels', cl)

    for c in range(1, cl.max()+1):
        idxs = np.where(cl == c)[0]

        for idx in idxs:
            ra, dec = galaxy_skycrds[idx]
            print(ra, dec)
            download_sdss_img('../data/result/galaxycluster/'+str(c)+'/'+str(idx)+'.jpg', ra, dec, 256, 256)
