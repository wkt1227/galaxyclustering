import glob

import astropy.io.fits as iofits
import numpy as np
from astropy.modeling import models, fitting
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from libs.funcs import get_psd1d, get_psd2d

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table

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
    height, width = img_p.shape
    patch_labels = np.zeros(len(picked_crds))

    labels[0:patch_size, :] = 0
    labels[height-patch_size:height, :] = 0
    labels[:, 0:patch_size] = 0
    labels[:, width-patch_size:width] = 0
    
    for i in range(1, num+1):
        pts = np.where(labels == i)
        print(label_cursor)
        if len(pts[0]) < min_pix:
            labels[pts] = 0
            continue
        
        for x, y in np.array(pts).T:
            # いらないかも
            if (x - patch_size < 0) or (x + patch_size >= height) \
                    or (y - patch_size < 0) or (y + patch_size >= width):
                continue
            
            patch_id = picked_crds.index([x, y])
            patch_labels[patch_id] = label_cursor
            
        #labels[pts] = label_cursor
        label_cursor += 1
        
    return patch_labels, label_cursor


def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in set(numbers):
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2


if __name__ == "__main__":

    fits_paths = glob.glob('../data/fits/*')
    all_patch_features = []
    all_patch_labels = []
    all_picked_crds = []
    all_picked_pix_vals = []
    all_galaxy_skycrds = []
    label_cursor = 1

    gz_table = Table.read('../data/GalaxyZoo1_DR_table2.fits')
    galaxy_fits_name_in_gz = np.load('../data/galaxy_fits_name_in_gz.npy')
    all_galaxy_labels_in_gz = []
    all_galaxy_idxs_in_gz = []

    for path in fits_paths:

        fits = iofits.open(path)
        img = fits[0].data
        header = fits[0].header
        wcs = WCS(header)
        
        mean, stddev = gaussian_fitting(img)
        img_p = img - mean

        # 取り出すピクセルを画像で保存
        fig = plt.figure()
        plt.imshow(img_p >= 4*stddev, cmap='gray')
        plt.title(path[13:-7])
        fig.show()
        fig.savefig('../reports/picked_pix/' + path[13:-7] + '.png')
        plt.close(fig)

        picked_crds = np.array(np.where(img_p >= 4*stddev)).T
        patch_size = 6
        height, width = img.shape
        patch_features = []
        picked_crds_t = []
        picked_pix_vals = []
        
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
            picked_pix_vals.append(img[x, y])


        # connected component labeling
        patch_labels, label_cursor = ccl(img_p, stddev, picked_crds_t, label_cursor=label_cursor)

        # GalaxyZoo との比較1
        fits_name = path[-25:]
        idxs = np.where(galaxy_fits_name_in_gz == fits_name)
        print('idxs', idxs)
        for r in gz_table[idxs]:
            ra_hms = r[1]
            dec_hms = r[2]

            c = SkyCoord(ra_hms + ' ' + dec_hms, unit=(u.hourangle, u.deg))
            ra_deg = c.ra.degree
            dec_deg = c.dec.degree

            y, x = wcs.wcs_world2pix(ra_deg, dec_deg, 0, ra_dec_order=True)
            x = x + 0.0
            y = y + 0.0
            x = round(x)
            y = round(y)
            idx = picked_crds_t.index([x, y])
            all_galaxy_idxs_in_gz.append(patch_labels[idx])
            all_galaxy_labels_in_gz.append(np.argmax([r[-3], r[-2], r[-1]]))
            print('GalaxyZoo: ',(picked_crds_t.index([x, y])), [r[-3], r[-2], r[-1]])


        # 銀河の座標を計算する
        galaxy_skycrds = []
        picked_crds_t = np.array(picked_crds_t)
        for i in range(int(second_smallest(patch_labels)), label_cursor):
            print(i)
            pts = np.where(patch_labels == i)
            s = len(pts[0])
            x, y = picked_crds_t[pts].sum(axis=0) / s

            ra, dec = wcs.wcs_pix2world(y, x, 0)
            ra = ra + 0.0
            dec = dec + 0.0
            
            if header['CTYPE1'] == 'DEC--TAN':
                ra, dec = dec, ra

            galaxy_skycrds.append([ra, dec])

        all_patch_features.extend(patch_features)
        all_patch_labels.extend(patch_labels)
        all_picked_crds.extend(picked_crds_t)
        all_picked_pix_vals.extend(picked_pix_vals)
        all_galaxy_skycrds.extend(galaxy_skycrds)
        print('銀河の数：{}, sample vectorの数：{}'.format(label_cursor-1, len(all_patch_features)))

    # 正規化
    all_patch_features = np.array(all_patch_features)
    all_patch_features = (all_patch_features - all_patch_features.mean(axis=0)) / all_patch_features.std(axis=0)
        
    np.save('../data/result/patch_labels', all_patch_labels)
    np.save('../data/result/patch_features', all_patch_features)
    np.save('../data/result/picked_crds', all_picked_crds)
    np.save('../data/result/galaxy_skycrds', all_galaxy_skycrds)


    # 銀河の明るさ、大きさを計算する
    galaxy_num = int(max(all_patch_labels))
    galaxy_sizes = np.zeros(galaxy_num)
    galaxy_brightnesses = np.zeros(galaxy_num)

    all_picked_pix_vals = np.array(all_picked_pix_vals)
    all_patch_labels = np.array(all_patch_labels).astype(np.int64)

    for i in range(galaxy_num):
        pts = np.where(all_patch_labels == i+1)

        galaxy_sizes[i] = len(pts[0])
        galaxy_brightnesses[i] = all_picked_pix_vals[pts].sum() / galaxy_sizes[i]

    np.save('../data/result/galaxy_sizes', galaxy_sizes)
    np.save('../data/result/galaxy_brightness', galaxy_brightnesses)


    # GalaxyZoo との比較2
    galaxy_labels2 = np.full(len(all_galaxy_skycrds), -1)
    for idx, label in zip(all_galaxy_idxs_in_gz, all_galaxy_labels_in_gz):
        galaxy_labels2[int(idx-1)] = label

    np.save('../data/result/galaxy_labels2.npy', galaxy_labels2)