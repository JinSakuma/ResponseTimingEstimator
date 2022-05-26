import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from itertools import chain

from src.models.gmf.gmf2 import GatedFusionBlock, AcousticEncoder, SemanticEncoder, TimingEncoder

torch.autograd.set_detect_anomaly(True)


class GMFModel(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.R = 60
        self.early_frame_num = 10
        
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
        vads = batch[2]
        #turn = batch[3].to(self.device)
        #last_ipu = batch[4].to(self.device)
        targets = batch[5].to(self.device)
        feats = batch[6].to(self.device)
        input_lengths = batch[7] #.to(self.device)
        offsets = batch[8]
        indices = batch[9]
        batch_size = int(len(chs))

        
        r_a_tmp = self.acoustic_encoder(feats, input_lengths)
        
        r_a_list = []
        targets_list = []
        text_list = []
        ends_list = []
        intervals_list = []
        speaking_rates = []
        
        correct = 0
        total = 0
        for i in range(len(chs)):
            res_timing=(targets[i][:input_lengths[i]][:-1]-targets[i][:input_lengths[i]][1:]).detach().cpu().numpy()
            timing = np.where(res_timing==-1)[0][0]+1

            res=(vads[i][:input_lengths[i]][:-1]-vads[i][:input_lengths[i]][1:]).detach().cpu().numpy()
            starts = np.where(res==-1)[0]+1
            ends = np.where(res==1)[0]+1

            if len(ends)>0:
                ends = ends[ends<=timing-offsets[i]//50+1]
            elif offsets[i]<=0:
                ends = np.array([input_lengths[i]-1])
            else:
                raise NotImplemented
                
            starts = starts.tolist()
            ends = ends.tolist()
            if vads[i][0]>0:
                starts = [0]+starts
            # ends += 4
            # print(timing)
            # print(starts)
            # print(ends)
            tmp = np.zeros(len(ends))
            tmp[:-1] = 0
            tmp[-1] = 1
            
            tgt_list = []
            src_id_list = []
            interval_tmp_list = []
            tl_pre = 0
            end_pre = 0            
            for j in range(len(ends)):
                start = starts[j]
                end = ends[j]
                tgt_tmp = tmp[j]
                n = min(self.early_frame_num, end-start)
                src_id_list+=[i for i in range(end-n+1, end+1)]
                tgt_list += [tgt_tmp]*n
                if i == 0:
                    interval_tmp_list+=[0]*n
                else:
                    interval_tmp_list+=[ends[j]-ends[j-1]]*n
                    
                for k in range(n):
                    sr = (len(texts[i][end-n+k+1])-tl_pre)/((end-n+k+1)-end_pre)
                    speaking_rates.append(sr)
                
                tl_pre = len(texts[i][end])
                end_pre = ends[j]

            r_a_list.append(r_a_tmp[i][src_id_list])
            targets_list.append(torch.tensor(tgt_list))
            text_list += np.array(texts[i])[src_id_list].tolist()
            ends_list += src_id_list#.tolist()
            intervals_list+=interval_tmp_list

        r_a = torch.cat(r_a_list)
        targets_ = torch.cat(targets_list).long().to(self.device)
        tlen = [len(t) for t in text_list]
        
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