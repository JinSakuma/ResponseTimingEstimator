import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


class ConcatFusionBlock(nn.Module):

    def __init__(self,
                 device,
                 acoustic_dim,
                 semantic_dim,
                 ):
        super().__init__()
        
        self.device = device
        self.fc_y = nn.Linear(acoustic_dim+semantic_dim, 2)
        
        self.criterion = nn.CrossEntropyLoss().to(device)
        
        
    def forward(self, r_a, r_s):
        """ Fusion multi-modal inputs
        Args:
            r_a: acoustic representations (N, D)
            r_s: semantic representations (N, D)
            r_t: timing representations   (N, D)
            
        Returns:
            (N, 2)
        """
        
        r_sa = torch.cat([r_s, r_a], dim=-1)
        y = torch.sigmoid(self.fc_y(r_sa))
        
        return y
   
    def get_loss(self, probs, targets):
        return self.criterion(probs, targets)
