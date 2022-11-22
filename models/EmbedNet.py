#!/usr/bin/python
#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import timm

class EmbedNet(nn.Module):

    def __init__(self, config, criterion):
        super(EmbedNet, self).__init__()
        self.embed_net = timm.create_model(config['model']['name'], num_classes = config['model']['embed_dim'])
        self.criterion = criterion

    def forward(self, x, label=None):
        x = self.embed_net(x)
        
        if label == None:
            return x

        nloss = self.criterion(x, label)
        return x, nloss