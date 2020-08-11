import numpy as np


if __name__ == "__main__":
    
    patch_labels = np.load('../data/result/patch_labels.npy').astype(np.int64)
    patch_labels2 = np.load('../data/result/patch_labels2.npy')

    galaxy_num = patch_labels.max()
    class_num = patch_labels2.max()

    galaxy_vectors = np.zeros((galaxy_num, class_num))
    
    for i in range(len(patch_labels)):
        if patch_labels[i] == 0:
            continue

        galaxy_vectors[patch_labels[i]-1, patch_labels2[i]-1] += 1

    # 正規化
    s = galaxy_vectors.sum(axis=1, keepdims=True)
    galaxy_vectors /= s

    np.save('../data/result/galaxy_vectors.npy', galaxy_vectors)