"""
Library with the samplers and functions related to the sampling processes both forward and backwards
"""
import torch
import numpy as np


def progressive_encoding(x, diffusion, config, n_samples=15):
    """ Runs a forward process iteratively

    We assume t=0 is already a one-step perturbed image

    Args:
        x: batch tensor NxCXHxW
        diffusion: an object from the DiffusionProcess class
        config: the config settings as set in the ml-collections file
        n_samples: a scalar value representing the number of samples equally distributed over the trajectory
    returns:
        n_samples at times T//15
    """

    xs = []

    # Initial sample - sampling from given state
    x = x.to(config.device)

    # equally subsample noisy states
    indx = np.linspace(0, diffusion.T-1, n_samples,  dtype = int)

    with torch.no_grad():
        # time partition [0,T]
        timesteps = torch.arange(0, diffusion.T, device=config.device)

        for t in timesteps:
            t_vec = torch.ones(x.shape[0], dtype=torch.int64, device=t.device) * t
            x, noise = diffusion.forward_step(x, t_vec)
            if t.item() in indx:
                xs.append(x)

        xs = torch.stack(xs, dim=1)

    return xs
