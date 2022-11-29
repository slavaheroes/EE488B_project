import torch.nn as nn
from pytorch_metric_learning import losses

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
	    super(LossFunction, self).__init__()
	    self.criterion  = losses.ArcFaceLoss(num_classes = nClasses, embedding_size = nOut)
	    print('Initialised ArcFace Loss')

	def forward(self, x, label=None):
		nloss   = self.criterion(x, label)
		return nloss
