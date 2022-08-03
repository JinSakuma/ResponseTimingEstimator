import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.rtg import RTG
from src.models.vad import VAD
from src.models.bert_encoder import BERTEncoder

torch.autograd.set_detect_anomaly(True)


class TimingEstimator(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.R = 60
        
        self.create_models()

    def create_models(self):
        rtg = RTG(
            self.device,
            self.config.model_params.input_dim+self.config.model_params.semantic_encoding_dim,
            self.config.model_params.hidden_dim,
            silence_encoding_type=None,
        )
        self.timing_model = rtg
        
        bert_encoder = BERTEncoder(
            self.config,
            self.device,
        )
        self.bert_encoder = bert_encoder

    def configure_optimizer_parameters(self):

        parameters = chain(
            self.timing_model.parameters(),
            # self.vad.parameters(),
            self.bert_encoder.linear.parameters(),
        )
        return parameters

    def forward(self, batch, split='train'):
        chs = batch[0]
        texts = batch[1]
        vad = batch[2]
        #turn = batch[3].to(self.device)
        #last_ipu = batch[4].to(self.device)
        targets = batch[5].to(self.device)
        feats = batch[6].to(self.device)
        input_lengths = batch[7] #.to(self.device)
        indices = batch[9] #.to(self.device)
        batch_size = int(len(chs))

        loss, acc = 0, 0        
#         for i in range(batch_size):
#             output = self.timing_model(feats[i], vad[i])
#             loss = loss+self.timing_model.get_loss(output[:input_lengths[i]], targets[i][:input_lengths[i]])
            # loss = loss+self.vad.get_loss(output[:input_lengths[i]][-self.R:], targets[i][:input_lengths[i]][-self.R:])
            #P, R, F1, A = self.f1_score(output[:input_lengths[i]][-self.R:], targets[i][:input_lengths[i]][-self.R:])
            #acc += A
            
        #acc = acc / float(batch_size)
        
        bert_embs = self.bert_encoder(texts)
        embs = torch.cat([feats, bert_embs], dim=-1)
        
        outputs = self.timing_model(embs, vad, input_lengths, indices, split)
        for i in range(batch_size):
            loss = loss+self.timing_model.get_loss(outputs[i][:input_lengths[i]], targets[i][:input_lengths[i]])

        outputs = {
            f'{split}_loss': loss,
        }

        return outputs
    
#     def f1_score(self, outputs, labels):

#         P, R, F1, acc = 0, 0, 0, 0
#         outputs = torch.sigmoid(outputs)

#         for i in range(outputs.shape[0]):
#             TP, FP, FN = 0, 0, 0
#             for j in range(outputs.shape[1]):
#                 if outputs[i][j] > 0.5 and labels[i][j] == 1:
#                     TP += 1
#                 elif outputs[i][j] <= 0.5 and labels[i][j] == 1:
#                     FN += 1
#                 elif outputs[i][j] > 0.5 and labels[i][j] == 0:
#                     FP += 1
#             precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
#             recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
#             F1 += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
#             P += precision
#             R += recall

#             p = (torch.where(outputs[i]>0.5)[0])
#             r = (torch.where(labels[i]==1)[0])
#             if len(p) == len(r) and (p == r).all():
#                 acc += 1

#         P /= outputs.shape[0]
#         R /= outputs.shape[0]
#         F1 /= outputs.shape[0]
#         return P, R, F1, acc
