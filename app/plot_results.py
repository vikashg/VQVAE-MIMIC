import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def make_gif(filenames):
    """
    Make a gif from a numpy array
    :return:
    """
    import imageio
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('./latent_space.gif', images)

if __name__ == '__main__':
    data_dir = '/raid/Vikash/Data/chest-x-ray-dataset-with-lung-segmentation-1.0.0/files/multi_gpu_model_VAE_latent_size_256/results'

    # Load the data
    for i in range(5, 205, 5):
        filename = os.path.join(data_dir, "result_mu_var{}.npy".format(i))
        print(filename)
        a = np.load(filename)
        j = 0
        print(a.shape)
        mu = a[j, 0, 0]
        logvar = a[j, 0, 1]
        print(mu, logvar)
        std = np.exp(0.5 * logvar)
        x = np.linspace(-4, 4, 100)
        plt.plot(x, norm.pdf(x, mu, std))
        plt.savefig(os.path.join('./test',"result_mu_logvar_{}.png".format(i)))
        plt.close()

    filelist = os.listdir('./test')
    filelist = [os.path.join('./test', f) for f in filelist if f.endswith('.png')]
    make_gif(filelist)

