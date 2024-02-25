import os
import zipfile
import tarfile

from skimage.io import imread
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
DIR = os.path.abspath(os.path.dirname(__file__))


class MNIST(datasets.MNIST):
    """
    Mnist wrapper. Docs: `datasets.MNIST.`
    """

    def __init__(self, root=os.path.join(DIR,'./data/mnist'), is_train=True):
        super().__init__(root,
                         train=is_train,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))

