#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, margin, scale, **kwargs):
	    super(LossFunction, self).__init__()

	    self.m = margin
        self.s = scale
	    self.fc = nn.Linear(nOut, nClasses, bias = False)
	    print('Initialised AM Softmax Loss')

	def forward(self, x, label=None):
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)