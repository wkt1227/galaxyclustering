import glob

import astropy.io.fits as iofits
import numpy as np
from astropy.modeling import models, fitting
    
from libs.funcs import get_psd1d, get_psd2d

import cv2


def gaussian_fitting(img):
    f_max, f_min, n_bin = 3000, 0, 3000
    x = np.linspace(f_min, f_max, n_bin + 1) + (f_max - f_min) / n_bin / 2
    x = x[:-1]
    pdf_fitted = np.histogram(img.flatten(), bins=n_bin, range=(f_min, f_max))
    pdf_fitted = pdf_fitted[0]
    fit = fitting.LevMarLSQFitter()
    gauss_init = models.Gaussian1D(mean=x[np.argmax(pdf_fitted)], stddev=100, amplitude=max(pdf_fitted))
    result = fit(gauss_init, x, pdf_fitted)
    
    return result.mean[0], result.stddev[0]


def ccl(img_p, sigma, picked_crds, label_cursor=1, min_pix=10):
    img_b = cv2.threshold(img_p, 4*sigma, 255, cv2.THRESH_BINARY)[1].astype('uint8')
    retval, labels = cv2.connectedComponents(img_b)
    num = labels.max()
    height, width = img.shape
    patch_labels = np.zeros(len(picked_crds))
    
    labels[0:patch_size, :] = 0
    labels[height-patch_size:height, :] = 0
    labels[:, 0:patch_size] = 0
    labels[:, width-patch_size] = 0
    
    for i in range(1, num+1):
        pts = np.where(labels == i)

        if len(pts[0]) < min_pix:
            labels[pts] = 0
            continue
        
        for x, y in np.array(pts).T:
            if (x - patch_size < 0) or (x + patch_size >= height) \
                    or (y - patch_size < 0) or (y + patch_size >= width):
                continue
            
            patch_id = picked_crds.index([x, y])
            patch_labels[patch_id] = label_cursor
            
        # labels[pts] = label_cursor
        label_cursor += 1
        
    return patch_labels, label_cursor

if __name__ == "__main__":

    fits_paths = glob.glob('../data/fits/*')
    all_patch_features = []
    all_patch_labels = []
    label_cursor = 1

    for path in fits_paths:

        fits = iofits.open(path)
        img = fits[0].data
        header = fits[0].header
        
        mean, stddev = gaussian_fitting(img)
        img_p = img - mean
        
        picked_crds = np.array(np.where(img_p >= 4*stddev)).T
        patch_size = 6
        height, width = img.shape
        patch_features = []
        picked_crds_t = []
        
        # patchを切り出し、特徴量ベクトルに変換する    
        for x, y in picked_crds:
            # patchを切り出せない場合
            if (x - patch_size < 0) or (x + patch_size >= height) \
                    or (y - patch_size < 0) or (y + patch_size >= width):
                continue

            patch = img[x - patch_size:x + patch_size, y - patch_size:y + patch_size]
            patch_feature = get_psd1d(get_psd2d(patch))
            patch_features.append(patch_feature)
            picked_crds_t.append([x, y])


        patch_labels, label_cursor = ccl(img_p, stddev, picked_crds_t, label_cursor=label_cursor)
        
        all_patch_features.extend(patch_features)
        all_patch_labels.extend(patch_labels)
        print('銀河の数：{}, sample vectorの数：{}'.format(label_cursor, len(all_patch_features)))
        
    print(len(set(all_patch_labels)))
    np.save('../data/result/patch_labels', all_patch_labels)
    np.save('../data/result/patch_features', all_patch_features)