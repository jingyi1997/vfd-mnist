import imageio
import numpy as np
import os
from time import gmtime, strftime
from collections import defaultdict
from timeit import default_timer
import pdb
from tqdm import tqdm

import torch
from torchvision.utils import save_image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from losses import VAELoss
from logger import Logger

class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    config : dict
        Configuration parameters for training.

    model : models.VAE
        The model to be trained.

    train_dataset : torch.utils.data.Dataset
        Dataset for training.

    eval_dataset : torch.utils.data.Dataset
        Dataset for evaluation.

    log_dir : str
        Directory for saving logs.

    device : torch.device, optional
        Device on which to run the code. Default is CPU.
    """

    def __init__(self, config, model, train_dataset, eval_dataset,
            log_dir, device=torch.device("cpu")):
        train_params = config['train_params']

        # Device
        self.device = device

        # Model
        self.model = model.to(self.device)

        # Dataset
        self.eval_dataset = eval_dataset
        self.data_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)

        # Logger
        self.logger = Logger(log_dir=log_dir, checkpoint_freq=train_params['checkpoint_freq'])

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])

        # Loss functions
        self.vae_loss = VAELoss(beta=train_params['beta'], steps_anneal=train_params['reg_anneal'])
        self.sfm_loss = torch.nn.CrossEntropyLoss()

        # Training parameters
        self.train_params = train_params

    def train(self):
        """
        Trains the model.

        """
        start = default_timer()
        self.model.train()

        for epoch in range(self.train_params['epochs']):
            for batch_idx, (data, label) in enumerate(tqdm(self.data_loader)):
                iter_loss = self.train_iteration(data, label)

                # Compute the total loss
                loss_values = [val.mean() for val in iter_loss.values()]
                loss = sum(loss_values)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (batch_idx+1) % self.train_params['print_freq'] == 0 :
                    print(np.array(self.logger.loss_list).mean(axis=0))
                
                # Log iteration loss
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in iter_loss.items()}
                self.logger.log_iter(losses=losses)

            self.logger.log_epoch(epoch, self.model)

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        print('Finished training after {:.1f} min.'.format(delta_time))


    def train_iteration(self, data, label):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).
 
        label : torch.Tensor
        Tensor of labels corresponding to the data batch.

        Returns
        -------
        dict[str, float]
            A dictionary containing the computed loss values for the iteration.
        """       
        # Ensure data and labels are on the correct device
        data, label = data.to(self.device), label.to(self.device)

        # Forward pass through the model
        recon_batch, latent_dist, latent_sample, cls_logits, aug_logits = self.model(data)

        # Compute losses
        rec_loss, kl_loss = self.vae_loss(data, recon_batch, latent_dist)
        cls_loss = self.sfm_loss(cls_logits, label)
        aug_loss = self.sfm_loss(aug_logits, label)
        
        # Aggregate loss values in a dictionary for return
        loss_values = {
            'rec_loss': rec_loss,  
            'kl_loss': kl_loss,
            'cls_loss': cls_loss,
            'aug_loss': aug_loss
        }

        return loss_values

    def visualize(self):
        """
        Visualizes the results.

        """
        self.model.eval()

        # Choose random indices from the evaluation dataset
        idcs = np.random.choice(len(self.eval_dataset), 5)
        all_to_plot = []

        with torch.no_grad():
          for i in idcs:
            img = self.data_loader.dataset[i][0].to(self.device)
            to_plot = img.clone().cpu().unsqueeze(0)

            mu, logvar, cls_feature = self.model.encoder(img)

            # Generate multiple augmented images
            for _ in range(self.train_params['aug_num']):
                aug_feature = self.model.reparameterize(mu, logvar, self.train_params['aug_var'])
                aggr_feature = cls_feature + aug_feature 
                img_recon = self.model.decoder(aggr_feature)
                to_plot = torch.cat([to_plot, img_recon.cpu()])     

            all_to_plot.append(to_plot)

        all_to_plot = torch.stack(all_to_plot, dim=0)

        row, col, c, h, w = all_to_plot.shape

        save_image(all_to_plot.view((-1,c,h,w)), 'visualize.png', nrow=col)


