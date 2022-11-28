#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from pytorch_metric_learning import distances, losses, miners, reducers

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, margin, **kwargs):
	    super(LossFunction, self).__init__()
        ## pytorch metric learning stuff
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        self.mining_func = miners.TripletMarginMiner(
            margin=margin, distance=distance, type_of_triplets="semihard"
        )
        self.tripletMarginLoss = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer)


	def forward(self, embeddings, labels):
        indices_tuple = self.mining_func(embeddings, labels)
        triplet = self.tripletMarginLoss(embeddings, labels, indices_tuple)
        loss = triplet
        print("Number of mined triplets = {}".format(self.mining_func.num_triplets))
        return loss