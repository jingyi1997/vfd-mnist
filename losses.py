import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

class VAELoss():
    """
    Compute the Variational Autoencoder (VAE) loss.

    Parameters:
        beta (float): Weight of the KL divergence term.
        steps_anneal (int, optional): Number of steps for annealing.

    """

    def __init__(self, beta, steps_anneal=10000):
        super().__init__()
        self.n_train_steps = 0
        self.beta = beta
        self.steps_anneal = steps_anneal

    def __call__(self, data, recon_data, latent_dist):
        self.n_train_steps += 1
        rec_loss = _reconstruction_loss(data, recon_data)
        kl_loss = _kl_normal_loss(*latent_dist)
        anneal_reg = linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
        kl_loss = anneal_reg * (self.beta * kl_loss)

        return rec_loss, kl_loss


def _reconstruction_loss(data, recon_data):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images).

    recon_data : torch.Tensor
        Reconstructed data. 

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy loss.
    """
    batch_size, n_chan, height, width = recon_data.size()
    loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    loss = loss / batch_size

    return loss


def _kl_normal_loss(mean, logvar):
    """
    Calculates the KL divergence between a normal distribution and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution.

    logvar : torch.Tensor
        Log variance of the normal distribution.

    Returns
    -------
    total_kl: torch.Tensor
        Total KL divergence loss.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()


    return total_kl


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

