import torch
from torch import nn, optim
from torch.nn import functional as F
import pdb
from .initialization import weights_init
from .encoders import get_encoder
from .decoders import get_decoder



def init_model(img_size, latent_dim, num_classes):
    """
    Initialize a Variational Autoencoder (VAE) model.

    Parameters
    ----------
    img_size : tuple of ints
        Size of images. E.g. (1, 32, 32) or (3, 64, 64).
    latent_dim : int
        Dimensionality of the latent space.
    num_classes : int
        Number of classes in the dataset.

    Returns
    -------
    VAE:
        Instance of the VAE model.
    """      
    encoder = get_encoder()
    decoder = get_decoder()
    model = VAE(img_size, encoder, decoder, latent_dim, num_classes)
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim, num_classes):
        """
        Initialize the VAE model.
        """
        super(VAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported.")

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim)
        self.classifier = nn.Linear(self.latent_dim, num_classes)
        self.reset_parameters()

    def reparameterize(self, mean, logvar, aug_var=1):
        """
        Reparameterization trick for sampling from a normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim).
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size, latent_dim).
        aug_var : float, optional
            Augmentation factor for the standard deviation, defaults to 1.

        Returns
        -------
        torch.Tensor
            Sampled latent vector.
        """
       
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mean + std * eps * aug_var

    def forward(self, x):
        """
        Forward pass of the VAE model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width).

        Returns
        -------
        tuple
            Tuple containing reconstructed data, latent distribution parameters, 
            sampled latent vector, classifier logits before and after augmentation.
        """
        mu, logvar, cls_feature = self.encoder(x)
        latent_dist = mu, logvar

        # Intra-class variance features sampled from latent distribution
        latent_sample = self.reparameterize(*latent_dist)

        # Classifier logits before feature augmentation
        cls_logits =  self.classifier(cls_feature)

        # Classifier logits after feature augmentation
        aggr_feature = cls_feature + latent_sample
        aug_logits =  self.classifier(aggr_feature)  

        # Reconstructed data from aggregated features
        reconstruct = self.decoder(aggr_feature)

        return reconstruct, latent_dist, latent_sample, cls_logits, aug_logits

    def reset_parameters(self):
        self.apply(weights_init)

