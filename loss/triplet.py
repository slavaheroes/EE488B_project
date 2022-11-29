import torch.nn as nn
import torch.nn.functional as F
import torch
import random

class LossFunction(nn.Module):
    def __init__(self, mining_margin, margin, **kwargs):
        super(LossFunction, self).__init__()
        self.margin     = margin   
        self.mining_margin = mining_margin     

    def forward(self, x, label=None):

        # Normalize anchor and positive
        out_anchor      = F.normalize(x[:,0,:], p=2, dim=1)
        out_positive    = F.normalize(x[:,1,:], p=2, dim=1)

        # Choose appropriate negative indices
        negidx      = self.choose_negative(out_anchor.detach(),out_positive.detach(),type='semihard')
        # Get negative pairs
        out_negative = out_positive[negidx, :]

        ## Calculate positive and negative pair distances

        pos_dist    = F.pairwise_distance(out_anchor, out_positive) 
        neg_dist    = F.pairwise_distance(out_anchor, out_negative) 

        ## Triplet loss function - write the loss equation yourself (squared L2 loss)
        loss   = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))
        return loss

    def choose_negative(self, embed_a, embed_p, type=None):
        # Get batch size
        batch_size = embed_a.size(0)

        # Positive and negative indices
        negidx = [] # empty list to fill
        posidx = range(0,batch_size)

        for idx in posidx:

            if type == 'any':
                # Random negative mining
                negidx.append(random.choice(posidx))

            elif type == 'semihard':
                # Compute pairwise distance between the anchor and all positives - use F.pairwise_distance
                distance = F.pairwise_distance(embed_a[idx, :], embed_p)

                # Semi-hard negative mining - criteria (distance greater than the positive pair, and less than positive + margin )
                # (your code)
                hardidx = torch.IntTensor(posidx)[(distance > distance[idx]) & (distance < distance[idx] + self.mining_margin)]

                if len(hardidx) == 0:
                    # append random index if no index matches the criteria
                    # (your code)
                    negidx.append(random.choice(posidx))
                else:
                    # append a random index that meets the criteria
                    # (your code)
                    negidx.append(random.choice(hardidx))

            elif type == 'hard':
                # Compute pairwise distance between the anchor and all positives - use F.pairwise_distance
                distance = F.pairwise_distance(embed_a[idx, :], embed_p)

                # Hard negative mining - criteria (distance less than the positive pair)
                # (your code)
                hardidx = torch.IntTensor(posidx)[(distance < distance[idx])]

                if len(hardidx) == 0:
                    # append random index if no index matches the criteria
                    # (your code)
                    negidx.append(random.choice(posidx))
                else:
                    # append a random index that meets the criteria
                    # (your code)
                    negidx.append(random.choice(hardidx))

            else:
                ValueError('Undefined type of mining.')
            
        return negidx
    