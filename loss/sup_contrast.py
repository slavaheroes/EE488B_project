import torch.nn as nn
from pytorch_metric_learning import losses

class LossFunction(nn.Module):
	def __init__(self, temperature, **kwargs):
	    super(LossFunction, self).__init__()

	    self.criterion  = losses.SupConLoss(temperature=temperature)

	    print('Initialised SupContrast Loss')

	def forward(self, x, label=None):
		nloss   = self.criterion(x, label)
		return nloss