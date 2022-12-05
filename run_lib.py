"""
    TRAINING AND EVALUATION FOR DIFFUSION MODELS
"""
import os
import io
import time

import torch
import logging
from torch.utils import tensorboard


import losses
import datasets
import plots as plts
import models.utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint
from diffusion_lib import GaussianDiffusion
import sampling


def train(config, workdir):
    """
    Runs the training pipeline.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    noise_model, model_name = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, noise_model.parameters())
    ema = ExponentialMovingAverage(noise_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=noise_model, ema=ema, step=0)

    print("Training ", model_name)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Save training samples
    batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    batch = inverse_scaler(batch)
    plts.save_image(batch, workdir, pos="vertical", name="data_samples")

    # Set uo the Forward diffusion process
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)  # defines the diffusion process

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)

    train_step_fn = losses.get_step_fn(diffusion, train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_step_fn(diffusion, train=False, optimize_fn=optimize_fn)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step, ))


    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (64, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        # sampling_fn = sampling.get_sampling_fn(config, diffusion, sampling_shape, inverse_scaler)

    # Generate and save samples
    logging.info("Generating samples of grid_size = 20x20")
    samples = sampling.sampling_fn(config, diffusion, noise_model, sampling_shape, inverse_scaler, denoise=True)
    logging.info("Saving generated samples at {}".format(workdir))
    plts.save_image(samples.clamp(0,1), workdir, n=64, pos="vertical", padding=1, w=22, scale=64,
                    name="{}_{}_data_samples_20x20".format(config.model.name, config.data.dataset.lower()))

###############
def evaluate(config,
             workdir,
             eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results.
                     Default to "eval".
    """

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # Build data pipeline
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    noise_model, model_name = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, noise_model.parameters())
    ema = ExponentialMovingAverage(noise_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=noise_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    diffusion = GaussianDiffusion(config.model.beta_min, config.model.beta_max, T=1000)  # defines the diffusion process

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.training.batch_size,
                          config.data.num_channels,
                          config.data.image_size,
                          config.data.image_size)
        sampling_fn = sampling.sampling_fn(config, diffusion, noise_model, sampling_shape, inverse_scaler, denoise=True)

    begin_ckpt = config.eval.begin_ckpt
    logging.info("Evaluation checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True

    logging.info("Evaluation checkpoint file: " +  ckpt_filename)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
        time.sleep(60)
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(120)
            state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(noise_model.parameters())

    # Generate and save samples
    logging.info("Generating samples")
    samples = sampling_fn(noise_model)
    samples = torch.clip(samples * 255, 0, 255).int()
    logging.info("Saving generated samples at {}".format(eval_dir))
    plts.save_image(samples, eval_dir, n=config.sampling.grid_size, pos="vertical", padding=0,
                    name="{}_{}_data_samples".format(config.model.name, config.data.dataset.lower()))

