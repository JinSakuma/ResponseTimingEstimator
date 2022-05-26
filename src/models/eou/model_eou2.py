import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.vad2 import VAD
from src.models.transformer_encoder import TransformerEncoder
from src.models.gmf.gmf import AcousticEncoder #, SemanticEncoder

torch.autograd.set_detect_anomaly(True)


class EoUDetactor(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.R = 60
        
        self.create_models()

    def create_models(self):
        
        ae = AcousticEncoder(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.acoustic_hidden_dim,
            self.config.model_params.acoustic_encoding_dim,
        )
        self.acoustic_encoder = ae
        
        bert_encoder = TransformerEncoder(
            self.config,
            self.device,
        )
        self.bert_encoder = bert_encoder
        
        vad = VAD(
            self.device,
            self.config.model_params.acoustic_encoding_dim+self.config.model_params.semantic_encoding_dim,
            self.config.model_params.vad_hidden_dim,
        )
        self.vad = vad

    def configure_optimizer_parameters(self):

        parameters = chain(
            self.acoustic_encoder.parameters(),
            self.bert_encoder.parameters(),
            self.vad.parameters(),
        )
        return parameters

    def forward(self, batch, split='train'):
        chs = batch[0]
        texts = batch[1]
        #vad_labels = batch[2].to(self.device)
        #eou_labels = batch[3].to(self.device)
        #last_ipu = batch[4].to(self.device)
        eou_labels = batch[5].to(self.device)
        #targets = batch[5].to(self.device)
        feats = batch[6].to(self.device)
        input_lengths = batch[7] #.to(self.device)
        offsets = batch[8] #.to(self.device)
        starts = batch[10] #.to(self.device)
        ends = batch[11] #.to(self.device)
        batch_size = int(len(chs))

        r_a = self.acoustic_encoder(feats, input_lengths)
        r_s = self.bert_encoder(texts)
        embs = torch.cat([r_s, r_a], dim=-1)
        
        vad_loss, vad_acc = 0, 0
        outputs = self.vad(embs, input_lengths)
               
        for i in range(batch_size):
            output = outputs[i]
            #idx = starts[i][-3:][0] # 最適化の範囲
            vad_loss = vad_loss+self.vad.get_loss(output[:input_lengths[i]], eou_labels[i][:input_lengths[i]])
            vad_acc = vad_acc+self.vad.get_acc(output[:input_lengths[i]], eou_labels[i][:input_lengths[i]])
        vad_acc = vad_acc / float(batch_size)

        outputs = {
            f'{split}_loss': vad_loss,
            f'{split}_vad_acc': vad_acc,
        }

        return outputs