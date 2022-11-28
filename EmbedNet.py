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
        EmbedNetModel = timm.create_model(config['model']['name'], num_classes = config['model']['embed_dim'])
        self.__S__ = EmbedNetModel

        ## __L__ is the classifier plus the loss function
        LossFunction = importlib.import_module('loss.'+config['loss']['name']).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**config['loss']['params']);

    def forward(self, data, label=None):
        data    = data.reshape(-1,data.size()[-3],data.size()[-2],data.size()[-1])
        outp    = self.__S__.forward(data)
        if label == None:
            return outp
        else:
            label   = label.view(-1)
            nloss = self.__L__.forward(outp,label)
            return nloss
