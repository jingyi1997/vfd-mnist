import argparse
import sys
import os
import yaml
import pdb
from argparse import ArgumentParser
from time import gmtime, strftime

from torch import optim
import torch
from models.vae import init_model
from training import Trainer
from datasets import MNIST


def load_config(config_path):
    with open(config_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    parser = ArgumentParser()
    parser = argparse.ArgumentParser(description="Train the VAE model for feature disentangling.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--log_dir", type=str,default="./output/", help="Directory to store logs.")
    parser.add_argument('--eval_only', action='store_true', help='Only to evaluate the model.')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint to load.")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Initialize model
    model = init_model(**config['model_params'])
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    # Setup logging directory 
    log_dir = os.path.join(args.log_dir, strftime("%d_%m_%y_%H.%M.%S", gmtime()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Initialize datasets
    train_dataset = MNIST(**config['dataset_params'], is_train=True)
    eval_dataset = MNIST(**config['dataset_params'], is_train=False)
    
    # Initialize trainer
    trainer = Trainer(config, model, train_dataset, eval_dataset, log_dir, device=device)

    if not args.eval_only:
        trainer.train()
    trainer.visualize()

if __name__ == '__main__':
    main()
