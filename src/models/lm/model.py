import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from src.models.lm.lstm import LSTMModel


torch.autograd.set_detect_anomaly(True)


class LSTMLM(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.create_models()

    def create_models(self):
        lm = LSTMModel(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.hidden_dim,
            self.config.model_params.output_dim,
            self.config.model_params.nlayers,
        )
        self.lm = lm

    def configure_optimizer_parameters(self):

        parameters = chain(
            self.lm.parameters(),
        )
        return parameters

    def forward(self, batch, split='train'):
        texts = batch[0]
        phonemes = batch[1]
        idxs = batch[2]      
        input_lengths = batch[3]
        indices = batch[4]
        batch_size = int(len(indices))

        inputs = idxs[:, :-1].to(self.device)
        targets = idxs[:, 1:].to(self.device)
        
        loss, acc = 0, 0
        outputs = self.lm(inputs, input_lengths)
        
        b, n, c = outputs.shape
        loss = loss+self.lm.get_loss(outputs.view(b*n, -1), targets.view(-1))
        
        # TODO: accuracy calculation
        acc = 0

        outputs = {
            f'{split}_loss': loss,
            f'{split}_acc': acc,
        }

        return outputs
