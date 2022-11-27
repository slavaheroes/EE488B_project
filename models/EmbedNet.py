#!/usr/bin/python
#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import timm
import torchvision

class EmbedNet(nn.Module):

    def __init__(self, config):
        super(EmbedNet, self).__init__()
        self.embed_net = timm.create_model(config['model']['name'], num_classes = config['model']['embed_dim'])

    def forward(self, data, label=None):
        x = data.reshape(-1,data.size()[-3],data.size()[-2],data.size()[-1])
        x = self.embed_net(x)
        return x