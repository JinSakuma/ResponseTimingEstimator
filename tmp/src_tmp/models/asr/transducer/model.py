import os
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.utils import (
    AverageMeter,
    save_checkpoint as save_snapshot,
    copy_checkpoint as copy_snapshot,
)
from src.models.asr.transducer.rnn_transducer import RNNT
from src.utils.utils import get_cer


torch.autograd.set_detect_anomaly(True)


class RNNTModel(nn.Module):

    def __init__(self, config, args, device):
        super().__init__()
        self.config = config
        self.args = args
        self.device = device
        self.create_models()

    def create_models(self):
        #self.config.loss_params.asr_weight>0.0:

        asr_model = RNNT(
            self.config.model_params.input_dim,
            self.config.model_params.output_dim,
            self.args,
        )
        self.asr_model = asr_model
        
#         if self.config.emb_params.sabsampling_type == "conb2d":
#             self.emb = Conv2dSubsampling(
#                 self.config.emb_params.input_dim,
#                 self.config.emb_params.output_dim,
#                 self.config.emb_params.dropout_rate,
#             )
#         else:
#             self.emb = None

    def configure_optimizer_parameters(self):
        
        parameters = chain(
            self.asr_model.parameters(),
        )        

        return parameters

    def get_asr_loss(self, log_probs, input_lengths, targets, target_lengths):
        loss = self.asr_model.get_loss(
            log_probs,
            input_lengths,
            labels,
            label_lengths,
            blank=0,
        )
        return loss

    def forward(self, batch, split='train', epoch=0):
        inputs = batch[0].to(self.device)
        input_lengths= batch[1]
        targets = batch[2].to(self.device)
        target_lengths= batch[3]
        indices = batch[4]
        # text = batch[5]
        # phon = batch[6]

        batch_size = len(indices)

        if split=='train':
            is_training = True
        else:
            is_training = False

        loss, _, cer = self.asr_model(inputs, input_lengths, targets, target_lengths, is_training)
                    
        outputs = {
            f'{split}_loss': loss,
            f'{split}_cer': cer,
        }

        return outputs
