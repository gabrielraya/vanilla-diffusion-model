import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def save_image(batch_images, workdir, n=64, padding=2, pos="horizontal", w=5.5, file_format="png", name="data_samples",  scale=4, show=False):
    
    if pos=="horizontal":
        sample_grid = make_grid(batch_images[:n], nrow=n, padding=padding)
    else:
        sample_grid = make_grid(batch_images[:n], nrow=int(np.sqrt(n)), padding=padding)
        
    fig = plt.figure(figsize=(n*5.5/scale,w))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu())
    fig.savefig(os.path.join(workdir, "{}.{}".format(name, file_format)), bbox_inches='tight')
    if show: 
        plt.show()
    plt.close(fig)


def show_image(x, workdir, name="image", h=6, w=5.5, show=False):
    fig = plt.figure(figsize=(h,w))
    plt.axis('off')
    if x.shape[0] == 1:
        plt.imshow(x.permute(1, 2, 0).cpu(), cmap="gray")
    else:
        plt.imshow(x.permute(1, 2, 0).cpu())
    fig.savefig(os.path.join(workdir, "{}.png".format(name)), dpi=fig.dpi, bbox_inches='tight')
    if not show:
        plt.close(fig)


def prepare_grid(batch, denoise_batch, inverse_scaler, grid_size=36):
    """ Construct grid to show denoise image next to original one

    Args
        batch: A mini-batch of evaluation data.
        denoise_batch: the denoise batch of data
        inverse_scaler: scaling function
        grid_size: size of the grid
    """
    assert(np.sqrt(grid_size) % 1== 0, "This function only plots images with a grid_size of nxn")
    B, C, H, W = batch.shape
    nrow = int(np.sqrt(grid_size))
    n=nrow**2//2

    x = inverse_scaler(batch[:n])
    denoise = torch.clip(denoise_batch[:n] * 1, 0, 1)

    a = []

    i = 0
    j = 0
    for m in range(grid_size):
        if m %2 == 0:
            a.append(x[i])
            i+=1
        else:
            a.append(denoise[j])
            j+=1

    return torch.cat(a).reshape(grid_size, C, H, W)