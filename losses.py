#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, nPerClass, **kwargs):
	    super(LossFunction, self).__init__()
	    self.test_normalize = True
	    self.nPerClass = nPerClass
	    self.criterion  = torch.nn.CrossEntropyLoss()
	    self.fc 		= nn.Linear(nOut,nClasses)
	    print('Initialised Combo Loss')

	def forward(self, x, label=None):
		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		return nloss