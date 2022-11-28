#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from pytorch_metric_learning import distances, losses, miners, reducers

class LossFunction(nn.Module):

    def __init__(self, nOut, nClasses, mining_margin, margin, **kwargs):
        super(LossFunction, self).__init__()
        self.arcFaceLoss = losses.ArcFaceLoss(num_classes=nClasses, embedding_size=nOut)
        distance = distances.CosineSimilarity()
        self.mining_func = miners.TripletMarginMiner(
            margin=mining_margin, distance=distance, type_of_triplets="semihard"
        )
        self.tripletMarginLoss = losses.TripletMarginLoss(margin=margin, distance=distance)

    def forward(self, embeddings, labels):
        indices_tuple = self.mining_func(embeddings, labels)
        arcFace = self.arcFaceLoss(embeddings, labels)
        triplet = self.tripletMarginLoss(embeddings, labels, indices_tuple)
        loss = triplet + softmax
        print(f'Number of mined triplets = {mining_func.num_triplets}')
        return loss