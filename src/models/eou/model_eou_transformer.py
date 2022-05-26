import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.vad_transformer import VAD
# from src.models.vad_ce import VAD
#from src.models.vad_softlabel import VAD

torch.autograd.set_detect_anomaly(True)


class EoUDetactor(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.R = 60
        
        self.create_models()

    def create_models(self):
        vad = VAD(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.vad_hidden_dim,
        )
        self.vad = vad

    def configure_optimizer_parameters(self):

        parameters = chain(
            self.vad.parameters(),
        )
        return parameters

    def forward(self, batch, split='train'):
        chs = batch[0]
        texts = batch[1]
        vad_labels = batch[2].to(self.device)
        #eou_labels = batch[3].to(self.device)
        #last_ipu = batch[4].to(self.device)
        eou_labels = batch[5].to(self.device)
        # targets = batch[5].to(self.device)
        feats = batch[6].to(self.device)
        input_lengths = batch[7] #.to(self.device)
        offsets = batch[8] #.to(self.device)
        batch_size = int(len(chs))

        
        vad_loss, vad_acc = 0, 0
        outputs = self.vad(feats, input_lengths)
        for i in range(batch_size):
            output = outputs[i]
            vad_loss = vad_loss+self.vad.get_loss(output[:input_lengths[i]], eou_labels[i][:input_lengths[i]])
            vad_acc = vad_acc+self.vad.get_acc(output[:input_lengths[i]], eou_labels[i][:input_lengths[i]])
#             vad_loss = vad_loss+self.vad.get_loss(output[:input_lengths[i]][vad_labels[i][:input_lengths[i]]==1], eou_labels[i][:input_lengths[i]][vad_labels[i][:input_lengths[i]]==1])
#             vad_acc = vad_acc+self.vad.get_acc(output[:input_lengths[i]][vad_labels[i][:input_lengths[i]]==1], eou_labels[i][:input_lengths[i]][vad_labels[i][:input_lengths[i]]==1])
        vad_acc = vad_acc / float(batch_size)

        outputs = {
            f'{split}_loss': vad_loss,
            f'{split}_vad_acc': vad_acc,
        }

        return outputs