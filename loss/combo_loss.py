#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from pytorch_metric_learning import distances, losses, miners, reducers

class LossFunction(nn.Module):

    def __init__(self, nOut, nClasses, **kwargs):
        super(LossFunction, self).__init__()
        # self.arcFaceLoss = losses.ArcFaceLoss(num_classes=nClasses, embedding_size=nOut)
        self.normSoftmax = losses.NormalizedSoftmaxLoss(num_classes=nClasses, embedding_size=nOut)
        self.contrastive = losses.ContrastiveLoss()

    def forward(self, embeddings, labels):
        # arcFace = self.arcFaceLoss(embeddings, labels)
        softmax = self.normSoftmax(embeddings, labels)
        contrast = self.contrastive(embeddings, labels)
        loss = contrast + softmax
        return loss