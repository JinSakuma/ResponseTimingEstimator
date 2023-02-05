import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from src.models.lm.lstm import LSTMModel
from src.models.lm.transformer import TransformerModel


torch.autograd.set_detect_anomaly(True)


class LSTMLM(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.create_models()

    def create_models(self):
        lm = LSTMModel(
            self.config,
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
        
        tpr, tnr, bAcc = 0, 0, 0
        if split!='train':
            _, preds = torch.max(outputs.data, -1)        
            EOU = 2306
            for i in range(batch_size):
                prd = preds[i][:input_lengths[i]].detach().cpu()
                trt = targets[i][:input_lengths[i]].detach().cpu()

                TP, FP, FN, TN = 0, 0, 0, 0
                for i in range(len(prd)):
                    if trt[i]==EOU and prd[i]==EOU:
                        TP+=1
                    elif trt[i]==EOU and prd[i]!=EOU:
                        FN+=1
                    elif trt[i]!=EOU and prd[i]==EOU:
                        FP+=1
                    else:
                        TN+=1

                if TP+FN>0:
                    tpr += TP / (TP+FN)
                else:
                    tpr += 0

                if TN+FP > 0:
                    tnr += TN / (TN+FP)
                else:
                    tnr += 0
                    
                bAcc += (tpr+tnr)/2
        
        # TODO: accuracy calculation
#         acc = 0
        outputs = {
            f'{split}_loss': loss,
            f'{split}_tpr': tpr,
            f'{split}_tnr': tnr,
            f'{split}_cnt': batch_size,
#             f'{split}_acc': acc,
        }

        return outputs

    
class TransformerLM(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.create_models()

    def create_models(self):
        lm = TransformerModel(
            self.config,
            self.device,
            self.config.model_params.output_dim,
            self.config.model_params.input_dim,            
            self.config.model_params.nheads,
            self.config.model_params.hidden_dim,
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
