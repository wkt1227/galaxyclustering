import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    galaxy_sizes = np.load('../data/result/galaxy_sizes.npy')
    galaxy_brightness = np.load('../data/result/galaxy_brightness.npy')
    galaxy_labels = np.load('../data/result/galaxy_labels.npy')

    galaxy_num = galaxy_labels.max()

    # logscaleにする
    galaxy_sizes = np.log(galaxy_sizes)
    galaxy_brightness = np.log(galaxy_brightness)

    #=======================================
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Histgram of Size')
    ax1.set_xlabel('Size (logscale)')
    ax1.set_ylabel('Frequency')
    ax1.hist(galaxy_sizes, color='gray')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Histgram of Brightness')
    ax2.set_xlabel('Brightness (logscale)')
    ax2.set_ylabel('Frequency')
    ax2.hist(galaxy_brightness, color='gray')

    plt.tight_layout()
    fig.suptitle('All ({})'.format(len(galaxy_brightness)))
    plt.subplots_adjust(top=0.85)
    fig.show()
    fig.savefig('../reports/distribution_logscale/all')
    plt.close()
    #=======================================

    for i in range(1, galaxy_num+1):
        idxs = np.where(galaxy_labels == i)[0]

        size_sample = galaxy_sizes[idxs]
        brightness_sample = galaxy_brightness[idxs]

        fig = plt.figure()

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('Histgram of Size')
        ax1.set_xlabel('Size (logscale)')
        ax1.set_ylabel('Freqency')
        ax1.hist(size_sample)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title('Histgram of Brightness')
        ax2.set_xlabel('Brightness (logscale)')
        ax2.set_ylabel('Frequency')
        ax2.hist(brightness_sample)

        plt.tight_layout()
        fig.suptitle('cluster{} ({})'.format(i, len(idxs)))
        plt.subplots_adjust(top=0.85)
        fig.show()
        fig.savefig('../reports/distribution_logscale/' + str(i))
        plt.close()