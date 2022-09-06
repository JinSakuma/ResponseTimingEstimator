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
from src.models.asr.rnn.ctc import CTC
from src.models.asr.rnn.subsampling import Conv2dSubsampling
from src.utils.utils import get_cer


torch.autograd.set_detect_anomaly(True)


class LSTMModel(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.create_models()

    def create_models(self):
        #self.config.loss_params.asr_weight>0.0:
        asr_model = CTC(
            self.config.model_params.input_dim,
            self.config.model_params.output_dim,
            num_layers=self.config.model_params.num_layers,
            hidden_dim=self.config.model_params.hidden_dim,
            bidirectional=self.config.model_params.bidirectional,
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

    def get_asr_loss(self, log_probs, input_lengths, labels, label_lengths):
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
        targets = batch[2]
        target_lengths= batch[3]
        indices = batch[4]
        # text = batch[5]
        # phon = batch[6]

        batch_size = len(indices)

        log_probs, _, _ = self.asr_model(inputs, input_lengths)
        loss = self.get_asr_loss(log_probs, input_lengths, targets, target_lengths)

        if split=="val":
            with torch.no_grad():
                hypotheses, hypothesis_lengths, references, reference_lengths = \
                    self.asr_model.decode(
                        log_probs, input_lengths, 
                        targets, target_lengths,    
                )
                asr_cer = get_cer(hypotheses, hypothesis_lengths, references, reference_lengths)
        else:
            asr_cer = 0

            
        outputs = {
            f'{split}_loss': loss,
            f'{split}_cer': asr_cer,
        }

        return outputs
