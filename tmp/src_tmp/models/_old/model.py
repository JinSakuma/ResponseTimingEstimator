import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.rtg import RTG

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
            self.config.model_params.input_dim,
            self.config.model_params.hidden_dim,
            silence_encoding_type=None,
        )
        self.timing_model = rtg

    def configure_optimizer_parameters(self):

        parameters = chain(
            self.timing_model.parameters(),
        )
        return parameters

    def forward(self, batch, split='train'):
        chs = batch[0]
        vad = batch[1]
        #turn = batch[2].to(self.device)
        #last_ipu = batch[3].to(self.device)
        targets = batch[4].to(self.device)
        feats = batch[5].to(self.device)
        input_lengths = batch[6] #.to(self.device)
        batch_size = int(len(chs))

        loss, acc = 0, 0        
#         for i in range(batch_size):
#             output = self.timing_model(feats[i], vad[i])
#             loss = loss+self.timing_model.get_loss(output[:input_lengths[i]], targets[i][:input_lengths[i]])
            # loss = loss+self.vad.get_loss(output[:input_lengths[i]][-self.R:], targets[i][:input_lengths[i]][-self.R:])
            #P, R, F1, A = self.f1_score(output[:input_lengths[i]][-self.R:], targets[i][:input_lengths[i]][-self.R:])
            #acc += A
            
        #acc = acc / float(batch_size)
        
        outputs = self.timing_model(feats, vad, input_lengths)
        for i in range(batch_size):
            loss = loss+self.timing_model.get_loss(outputs[i][:input_lengths[i]], targets[i][:input_lengths[i]])

        outputs = {
            f'{split}_loss': loss,
        }

        return outputs


class VoiceActivityDetactor(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.R = 60
        
        self.create_models()

    def create_models(self):
        rtg = RTG(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.hidden_dim,
            silence_encoding_type=None,
        )
        self.timing_model = rtg

    def configure_optimizer_parameters(self):

        parameters = chain(
            self.timing_model.parameters(),
        )
        return parameters

    def forward(self, batch, split='train'):
        chs = batch[0]
        texts = batch[1]
        vad_labels = batch[2]
        #turn = batch[3].to(self.device)
        #last_ipu = batch[4].to(self.device)
        targets = batch[5].to(self.device)
        feats = batch[6].to(self.device)
        input_lengths = batch[7] #.to(self.device)
        offsets = batch[8] #.to(self.device)
        batch_size = int(len(chs))

        
        vad_loss, vad_acc = 0, 0
        outputs = self.vad(feats, input_lengths)
        for i in range(batch_size):
            output = outputs[i]
            vad_loss = vad_loss+self.vad.get_loss(output[:input_lengths[i]], vad_labels[i][:input_lengths[i]])
            vad_acc = vad_acc+self.vad.get_acc(output[:input_lengths[i]], vad_labels[i][:input_lengths[i]])
        vad_acc = vad_acc / float(batch_size)

        outputs = {
            f'{split}_loss': vad_loss,
            f'{split}_vad_acc': vad_acc,
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
