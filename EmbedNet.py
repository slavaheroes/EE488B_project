#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys
import time, importlib
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import timm

class EmbedNet(nn.Module):

    def __init__(self, config):
        super(EmbedNet, self).__init__();

        ## __S__ is the embedding model
        EmbedNetModel = importlib.import_module(config['model']['module']).__getattribute__(config['model']['name'])
        self.__S__ = EmbedNetModel(**config['model']['params'])

        ## __L__ is the classifier plus the loss function
        LossFunction = importlib.import_module('loss.'+config['loss']['name']).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**config['loss']['params']);

        self.nPerClass = config['dataloader']['nPerClass']

    def forward(self, data, label=None):
        data    = data.reshape(-1,data.size()[-3],data.size()[-2],data.size()[-1])
        outp    = self.__S__.forward(data)
        if label == None:
            return outp
        else:
            outp    = outp.reshape(self.nPerClass,-1,outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss = self.__L__.forward(outp,label)
            return nloss
