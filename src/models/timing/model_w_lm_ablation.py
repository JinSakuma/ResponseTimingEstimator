import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.timing.rtg import RTG
from src.models.encoder.acoustic_encoder import AcousticEncoder
from src.models.encoder.timing_encoder6 import TimingEncoder
from src.models.encoder.transformer_encoder import TransformerEncoder

torch.autograd.set_detect_anomaly(True)


class TimingEstimator(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device 
        
        self.is_use_acoustic = False
        self.is_use_semantic = False
        self.is_use_timing = False
        
        self.R = 60
        
        self.create_models()

    def create_models(self):
        
        encoding_dim = self.config.model_params.acoustic_encoding_dim \
                        + self.config.model_params.semantic_encoding_dim \
                        + self.config.model_params.timing_encoding_dim
        
        rtg = RTG(
            self.device,
            encoding_dim,
            self.config.model_params.hidden_dim,
        )
        self.timing_model = rtg
        
        if self.config.model_params.acoustic_encoding_dim > 0:        
            ae = AcousticEncoder(
                self.device,
                self.config.model_params.input_dim,
                self.config.model_params.acoustic_hidden_dim,
                self.config.model_params.acoustic_encoding_dim,
            )
            self.acoustic_encoder = ae
            self.is_use_acoustic = True
            
        if self.config.model_params.semantic_encoding_dim > 0:        
            se = TransformerEncoder(
                self.config,
                self.device,
            )
            self.semantic_encoder = se
            self.is_use_semantic = True
            
        if self.config.model_params.timing_encoding_dim > 0:
            is_use_silence=True
            is_use_n_word=True
            
            #if not self.is_use_acoustic:     
            #    is_use_silence=False
                
            if self.config.model_params.n_best<=1:
                is_use_n_word=False
                
            te = TimingEncoder(
                self.config,
                self.device,
                self.config.model_params.timing_input_dim,
                self.config.model_params.timing_encoding_dim,
                is_use_silence,
                is_use_n_word,
            )
            self.timing_encoder = te   
            self.is_use_timing = True
                

    def configure_optimizer_parameters(self):

        if self.is_use_acoustic and self.is_use_semantic and self.is_use_timing:
            parameters = chain(
                self.timing_model.parameters(),
                self.acoustic_encoder.parameters(),
                self.timing_encoder.linear.parameters(),
                self.semantic_encoder.parameters(),
            )
            return parameters
        elif self.is_use_acoustic and self.is_use_semantic and not self.is_use_timing:
            print('use acoustic and semantic')
            parameters = chain(
                self.timing_model.parameters(),
                self.acoustic_encoder.parameters(),
                self.semantic_encoder.parameters(),
            )
            return parameters
        elif self.is_use_acoustic and not self.is_use_semantic and self.is_use_timing:
            print('use acoustic and timing')
            parameters = chain(
                self.timing_model.parameters(),
                self.acoustic_encoder.parameters(),
                self.timing_encoder.parameters(),
            )
            return parameters
        elif not self.is_use_acoustic and self.is_use_semantic and self.is_use_timing:
            print('use semantic and timing')
            parameters = chain(
                self.timing_model.parameters(),
                self.semantic_encoder.parameters(),
                self.timing_encoder.parameters(),
            )
            return parameters
        elif self.is_use_acoustic and not self.is_use_semantic and not self.is_use_timing:
            print('use only acoustic feature')
            parameters = chain(
                self.timing_model.parameters(),
                self.acoustic_encoder.parameters(),
            )
            return parameters
        elif not self.is_use_acoustic and self.is_use_semantic and not self.is_use_timing:
            print('use only semantic feature')
            parameters = chain(
                self.timing_model.parameters(),
                self.semantic_encoder.parameters(),
            )
            return parameters

    def forward(self, batch, split='train'):
        chs = batch[0]
        texts = batch[1]
        kanas = batch[2]
        idxs = batch[3]
        vad = batch[4]
        #turn = batch[5].to(self.device)
        #last_ipu = batch[6].to(self.device)
        targets = batch[7].to(self.device)
        feats = batch[8].to(self.device)
        input_lengths = batch[9] #.to(self.device)
        offsets = batch[10] #.to(self.device)
        indices = batch[11] #.to(self.device)
        batch_size = int(len(chs))

        loss, acc = 0, 0
        
        if self.is_use_acoustic:
            r_a = self.acoustic_encoder(feats, input_lengths)
        if self.is_use_semantic:
            r_s = self.semantic_encoder(texts)
        if self.is_use_timing:
            r_t = self.timing_encoder(feats, idxs, input_lengths, indices, split)        
        
        if self.is_use_acoustic and self.is_use_semantic and self.is_use_timing:
            embs = torch.cat([r_s, r_a, r_t], dim=-1)    
        elif self.is_use_acoustic and not self.is_use_semantic and self.is_use_timing:
            embs = torch.cat([r_a, r_t], dim=-1)    
        elif not self.is_use_acoustic and self.is_use_semantic and self.is_use_timing:
            embs = torch.cat([r_s, r_t], dim=-1)    
        elif self.is_use_acoustic and self.is_use_semantic and not self.is_use_timing:
            embs = torch.cat([r_s, r_a], dim=-1)
        elif self.is_use_acoustic and not self.is_use_semantic and not self.is_use_timing:
            embs = r_a
        elif not self.is_use_acoustic and self.is_use_semantic and not self.is_use_timing:
            embs = r_s
                    
        outputs = self.timing_model(embs, input_lengths)
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
