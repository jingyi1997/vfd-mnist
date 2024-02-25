import numpy as np
import torch
import torch.nn.functional as F
import imageio
import pdb
import os

import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    
    def save_cpk(self):
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        print('save to: ', cpk_path)
        torch.save(self.model.state_dict(), cpk_path)



    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
            print(self.names)
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, model):
        self.epoch = epoch
        self.model = model
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)


