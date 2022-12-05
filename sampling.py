"""
Library with the samplers and functions related to the sampling processes both forward and backwards
"""
import torch
import numpy as np
import abc
import functools
from tqdm import tqdm

from models import utils as mutils

_SAMPLERS = {}


def register_sampler(cls=None, *, name=None):
    """A decorator for registering sampler classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _SAMPLERS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _SAMPLERS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_sampler(name):
    return _SAMPLERS[name]


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
    indx = np.linspace(0, diffusion.T - 1, n_samples, dtype=int)

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


class Sampler(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, diffusion, model_fn):
        super().__init__()
        self.diffusion = diffusion
        self.model_fn = model_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the sampler.

        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_sampler(name='ancestral_sampling')
class AncestralSampling(Sampler):
    """The ancestral sampler used in the DDPM paper"""

    def __init__(self, diffusion, model_fn):
        super().__init__(diffusion, model_fn)

    def denoise_update_fn(self, x, t):
        diffusion = self.diffusion

        beta = diffusion.discrete_betas.to(t.device)[t.long()]
        std = diffusion.sqrt_1m_alphas_cumprod.to(t.device)[t.long()]

        predicted_noise = self.model_fn(x, t)  # set the model either for training or evaluation
        score = - predicted_noise / std[:, None, None, None]

        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        return self.denoise_update_fn(x, t)


def shared_denoise_update_fn(x, t, diffusion, model, sampler):
    """A wrapper that configures and returns the update function of samplers."""

    model_fn = mutils.get_model_fn(model, train=False)
    sampler_obj = sampler(diffusion, model_fn)
    return sampler_obj.update_fn(x, t)


def get_sampling_fn(config, diffusion, shape, inverse_scaler, denoise=True):
    """Create a sampling function.

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        diffusion: A `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers representing the expected shape of a single sample.
        inverse_scaler: The inverse data normalizer function.
        denoise: If `True`, add one-step denoising to the final samples.

    Returns:
        A function that takes random states and a replicated training state
        and outputs samples with the training dimensions matching `shape`.
    """

    sampler = get_sampler(config.sampling.sampler.lower())

    denoise_update_fn = functools.partial(shared_denoise_update_fn,
                                          diffusion=diffusion,
                                          sampler=sampler)

    def sampling_fn(model):
        """"The sampling function.

        Args:
            model: A noise prediction model.

        Returns:
            Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample - sampling from tractable prior
            x = diffusion.prior_sampling(shape).to(config.device)
            # reverse time partition [T, 0]
            timesteps = torch.flip(torch.arange(0, diffusion.T, device=config.device), dims=(0,))

            for i in tqdm(range(diffusion.N)):
                t = torch.ones(shape[0], device=config.device) * timesteps[i]
                x, x_mean = denoise_update_fn(x, t, model=model)

            return inverse_scaler(x_mean if denoise else x)

    return sampling_fn
