import torch
import torch.utils.data as data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time
# from easydict import EasyDict as edict
import yaml
from PIL import Image
import imageio


def plot_result(G, fixed_noise, image_size, num_iter, num_epoch, save_dir, fig_size=(5, 5), is_gray=False):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    G.eval()
    generate_images = G(fixed_noise)
    G.train()

    n_rows = n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

    for ax, img in zip(axes.flatten(), generate_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        if is_gray:
            img = img.cpu().data.view(image_size, image_size).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')
        else:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
                np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}/{0}'.format(num_epoch, num_iter)
    fig.text(0.5, 0.04, title, ha='center')

    plt.savefig(os.path.join(save_dir, 'DCGAN_epoch_{:4d}_{:4d}.png'.format(num_epoch, num_iter)))
    plt.close()