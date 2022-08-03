import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from itertools import chain

from src.models.gmf.gmf import GatedFusionBlock, AcousticEncoder, SemanticEncoder, TimingEncoder

torch.autograd.set_detect_anomaly(True)


class GMFModel(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        alpha = config.loss_params.loss_weight
        self.weights = torch.tensor([1.0, alpha])#.to(device)
        
        self.R = 60
        
        self.create_models()

    def create_models(self):
           
        ae = AcousticEncoder(
            self.device,
            self.config.model_params.acoustic_input_dim,
            self.config.model_params.acoustic_hidden_dim,
            self.config.model_params.acoustic_encoding_dim,
        )
        self.acoustic_encoder = ae               
        
        se = SemanticEncoder(
            self.device,
            self.config.model_params.bert_hidden_dim,
            self.config.model_params.semantic_encoding_dim,
        )
        self.semantic_encoder = se
        
        te = TimingEncoder(
            self.device,
            self.config.model_params.timing_input_dim,
            self.config.model_params.timing_encoding_dim,
        )
        self.timing_encoder = te
        
        gfblock = GatedFusionBlock(
            self.device,
            self.config.model_params.acoustic_encoding_dim,
            self.config.model_params.semantic_encoding_dim,
            self.config.model_params.timing_encoding_dim,
            self.config.model_params.encoding_dim,
            self.weights,
        )
        self.gfblock = gfblock

    def configure_optimizer_parameters(self):

        parameters = chain(
            self.acoustic_encoder.parameters(),
            self.semantic_encoder.parameters(),
            self.timing_encoder.parameters(),
            self.gfblock.parameters(),
        )
        return parameters

    def forward(self, batch, split='train'):
        chs = batch[0]
        texts = batch[1]
        # kanas = batch[2]
        # idxs = batch[3]
        
        vads = batch[4]
        #turn = batch[5].to(self.device)
        #last_ipu = batch[6].to(self.device)
        targets = batch[7].to(self.device)
        feats = batch[8].to(self.device)
        input_lengths = batch[9] #.to(self.device)
        offsets = batch[10]
        indices = batch[11]
        batch_size = int(len(chs))

        
        r_a_tmp = self.acoustic_encoder(feats, input_lengths)
        
        r_a_list = []
        targets_list = []
        text_list = []
        ends_list = []
        intervals_list = []
        
        correct = 0
        total = 0
        for i in range(len(chs)):
            res_timing=(targets[i][:input_lengths[i]][:-1]-targets[i][:input_lengths[i]][1:]).detach().cpu().numpy()
            timing = np.where(res_timing==-1)[0][0]+1

            res=(vads[i][:input_lengths[i]][:-1]-vads[i][:input_lengths[i]][1:]).detach().cpu().numpy()
            starts = np.where(res==-1)[0]+1
            ends = np.where(res==1)[0]+1

            if len(ends)>0:
                ends = ends[ends<=5+timing-offsets[i]//50+1]                                
            elif offsets[i]<=0:
                ends = np.array([input_lengths[i]-1])
#             else:                
#                 raise NotImplemented
                
            if len(ends)==0:
                ends = np.array([input_lengths[i]-1])
                ends_ = np.array([input_lengths[i]-1])
            else:
                ends_ = ends.copy()
                for j in range(len(ends)):                   
                    if ends[j]+4 < input_lengths[i]:
                        ends_[j] = ends[j] + 4
                        
                
            # ends += 4
            # print(timing)
            # print(starts)
            # print(ends)
            tmp = targets[i][ends]
            try:
                tmp[:-1] = 0
                tmp[-1] = 1
            except:
                print(np.where(res==1)[0]+1)
                print(tmp)
                print(len(targets[i]))
                print(ends, len(ends))
                print(timing)
                print(offsets[i]//50+1)
                print(aaa)
                

            r_a_list.append(r_a_tmp[i][ends])
            targets_list.append(tmp)
            text_list += np.array(texts[i])[ends_].tolist()
            ends_list += ends.tolist()
            interval = [0]+list(ends[1:]-ends[:-1])
            intervals_list+=interval

        r_a = torch.cat(r_a_list)
        targets_ = torch.cat(targets_list)
        tlen = [len(t) for t in text_list]
        
        tl_pre = 0
        end_pre = 0
        speaking_rates = []
        for i in range(len(tlen)):
            speaking_rates.append((tlen[i]-tl_pre)/ends_list[i]-end_pre)
            tl_pre = tlen[i]
            end_pre = ends_list[i]

        feat_t = torch.tensor([ends_list, tlen, intervals_list, speaking_rates]).permute(1,0).float().to(self.device)
        
        r_s = self.semantic_encoder(text_list)
        r_t = self.timing_encoder(feat_t)
        
        probs = self.gfblock(r_a, r_s, r_t)
        loss = self.gfblock.get_loss(probs, targets_)
        
        correct +=(torch.argmax(probs, dim=1)==targets_).sum().detach().cpu().numpy()
        total += len(targets_)

        outputs = {
            f'{split}_loss': loss,
            f'{split}_correct': correct,
            f'{split}_total': total,
        }

        return outputs