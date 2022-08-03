import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

import numpy as np
from queue import PriorityQueue
import operator

from src.utils.utils import load_config
from src.models.vad.vad import VAD
from src.models.lm.model import LSTMLM

torch.autograd.set_detect_anomaly(True)


path = '/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr/asr1/data/jp_token_list/char/tokens.txt'
with open(path) as f:
    lines =f.readlines()
    
tokens = [line.split()[0] for line in lines]

def token2idx(token): 
    if token != token or token == '':
        return [0]
    
    token = token.replace('<eou>', '')
    idxs = [tokens.index(t) for t in token]+[len(tokens)-1]
    
    return idxs

def idx2token(idxs): 
    token = [tokens[idx] for idx in idxs]
    
    return token


silence_dict_train={}
silence_dict_val={}
silence_dict_test={}

silence_dict = {'train': silence_dict_train, 'val': silence_dict_val, 'test': silence_dict_test}

n_word_dict_train={}
n_word_dict_val={}
n_word_dict_test={}

n_word_dict = {'train': n_word_dict_train, 'val': n_word_dict_val, 'test': n_word_dict_test}

n_best_score_dict_train={}
n_best_score_dict_val={}
n_best_score_dict_test={}

n_best_score_dict = {'train': n_best_score_dict_train, 'val': n_best_score_dict_val, 'test': n_best_score_dict_test}

class TimingEncoder(nn.Module):

    def __init__(self,
                 config,
                 device,
                 input_dim,
                 encoding_dim,
                 is_use_silence=True,
                 is_use_n_word=False,
                ):
        
        super().__init__()
        self.config = config
        self.device = device
        self.is_use_silence=is_use_silence
        self.is_use_n_word=is_use_n_word
        
        self.reset_frame = 5
        self.beam_width = config.model_params.n_best
        self.EOU_id = config.model_params.eou_id
        self.max_n_word = config.model_params.max_n_word
        
        if is_use_silence:
            vad_model_path = config.model_params.vad_model_path
            vad = VAD(
                self.device,
                self.config.model_params.input_dim,
                self.config.model_params.vad_hidden_dim,
            )
            self.vad = vad        
            self.vad.load_state_dict(torch.load(vad_model_path), strict=False)
        
        if is_use_n_word:
            lm_config_path = config.model_params.lm_config_path
            lm_model_path = config.model_params.lm_model_path
            lm_config = load_config(lm_config_path)    
            self.lm = LSTMLM(lm_config, device)
            self.lm.load_state_dict(torch.load(lm_model_path), strict=False)
        
        self.linear = nn.Linear(input_dim,
                                encoding_dim,
                               )
        
    def get_silence(self, uttr_pred, indice, split):
        """ Fusion multi-modal inputs
        Args:
            vad_pred: acoustic feature (N, 1)
            indice: data idx 
            split: training phase
            
        Returns:
            outputs: silence count (N, 1)
        """
        
        if indice not in silence_dict[split]:
            uttr_pred = torch.sigmoid(uttr_pred)
            uttr_pred = uttr_pred.detach().cpu()
            silence = torch.zeros(len(uttr_pred))#.to(self.device)
            silence[0] = uttr_pred[0]
            et = 0
            cnt = 0
            pre = 0
            for i, u in enumerate(uttr_pred):
                uu = 1 - u
                et += uu  # 尤度を足す
                if uu > 0.5:  # 尤度が0.5以上なら非発話区間
                    cnt = 0
                if pre >= 0.5 and uu < 0.5:  # 尤度が0.5以下になったらカウント開始
                    cnt += 1
                elif uu < 0.5 and cnt > 0:  # カウントが始まっていればカウント
                    cnt += 1
                if cnt > self.reset_frame:  # 一定時間カウントされたらリセット
                    et = 0
                    cnt = 0

                silence[i] = et
                pre = uu
                
            silence_dict[split][indice] = silence
            
        else:
            silence = silence_dict[split][indice]
        
        return silence.unsqueeze(1).to(self.device)
    
    def lm_generate(self, inp):
        n = len(inp)
        outputs = inp.detach().cpu().tolist()
        prev = inp.unsqueeze(0)
        flg = False
        with torch.no_grad():
            hidden = None
            for i in range(self.max_n_word):
                out, hidden = self.lm.lm.forward_step(prev, hidden)

                _, pred = torch.topk(out[0][-1], 1)

                idx = int(pred[-1].detach().cpu())
                outputs += [idx]

                if idx == self.EOU_id:
                    flg=True
                    break

                prev = pred[-1].unsqueeze(0).unsqueeze(0)

        m = len(outputs)
        N = m-n
        if flg:
            N-=1

        return outputs, N
    
    def get_n_word(self, idxs, indice, split):
        """ Fusion multi-modal inputs
        Args:
            vad_pred: acoustic feature (N, 1)
            indice: data idx 
            split: training phase
            
        Returns:
            outputs: silence count (N, 1)
        """
        
        if indice not in n_word_dict[split]:
            rest = []
            pre = [0]
            n = self.max_n_word
            for j in range(len(idxs)):
                if idxs[j] != pre:
                    inp = torch.tensor(idxs[j])
                    pre = idxs[j]

                    sample, n = self.lm_generate(inp.to(self.device))                           
                else:
                    pass

                rest.append(n)            
            
            n_word = torch.tensor(rest).unsqueeze(1)
            n_word_dict[split][indice] = n_word
            
            
        else:
            n_word = n_word_dict[split][indice]
        
        return n_word.to(self.device)    

    def get_n_best_length(self, inp):        
        endnodes = []
        with torch.no_grad():        
            decoder_output, hidden = self.lm.lm.forward_step(inp.unsqueeze(0))
            prob = torch.softmax(decoder_output, dim=-1)

            prob_, indexes = torch.topk(prob, self.beam_width)
            prob_ = prob_[0][-1] 
            indexes = indexes[0][-1]

            n_end = 0
            cur_nodes = PriorityQueue()
            for i in range(len(indexes)):        
                n = BeamSearchNode(hidden, None, indexes[i], prob_[i], 1)
                cur_nodes.put((-prob_[i], i+100*(0+1), n))

            cnt = 1
            while True:
                if cur_nodes.qsize()== 0:            
                    break
                    
                if cnt >= self.max_n_word:
                    for bw in range(self.beam_width):
                        p, _, cur_node = cur_nodes.get()
                        endnodes.append(cur_node)
                    break

#                 elif n_end >= self.beam_width:
#                     break

                nxt_nodes = PriorityQueue()

                eou_prob_tmp = 0
                for bw in range(self.beam_width):
                    if cur_nodes.qsize()== 0:            
                        break
                
                    p, _, cur_node = cur_nodes.get()            
                    idx = cur_node.wordid
                    hidden = cur_node.hidden_state

                    if cur_node.wordid.item() == self.EOU_id:
                        n_end+=1
                        endnodes.append(cur_node)
                        continue

                    decoder_output, hidden = self.lm.lm.forward_step(idx.unsqueeze(0).unsqueeze(0), hidden)
                    prob = torch.softmax(decoder_output, dim=-1)


                    prob_, indexes = torch.topk(prob, self.beam_width)
                    prob_ = prob_[0][-1] 
                    indexes = indexes[0][-1]

                    for i in range(len(indexes)):                                    
                        n = BeamSearchNode(hidden, cur_node, indexes[i], cur_node.prob*prob_[i], cnt+1)
                        # n = BeamSearchNode(hidden, cur_node, indexes[i], cur_node.prob+prob_[i]/float(cnt+1), cnt+1)
                        try:
                            nxt_nodes.put((-prob_[i], i+100*(bw+1), n))
                        except:
                            print(prob_[i])
                            print(aaaaaaaaaa)                            
                            

                cur_nodes = nxt_nodes        
                cnt += 1

        return endnodes
    
    def get_n_word_n_best_scores(self, idxs, indice, split, max_len):
        """ Fusion multi-modal inputs
        Args:
            vad_pred: acoustic feature (N, 1)
            indice: data idx 
            split: training phase
            
        Returns:
            outputs: silence count (N, beam_width)
        """
        
        if indice not in n_best_score_dict[split]:

            pre = [0]
            n_best_scores = np.zeros([len(idxs), self.max_n_word])
            texts = []
            text = ''
#             texts = []
            n_score = np.zeros(self.max_n_word)
            n_score[-1] = 1
            for j in range(len(idxs)):                
                if idxs[j] != pre:
                    inp = torch.tensor(idxs[j])
                    pre = idxs[j]
                    
                    endnodes = self.get_n_best_length(inp.to(self.device))
                    context = inp.detach().cpu().tolist()

                    utterances = []
                    scores = []
                    n_score = np.zeros(self.max_n_word)
#                     total = 0
                    for node in endnodes:
                        utterance = []
                        n = node
                        score = n.prob
                        while n is not None:
                            utterance.append(n.wordid.item())        
                            n = n.prevNode        

                        utterances.append(context+utterance[::-1])
                        length = len(utterance)
                        n_score[length-1:]+=score.item()
                        scores.append(score.item())

                    text = ''.join(idx2token(utterances[np.argmax(scores)]))
                        
                else:
                    pass

                n_best_scores[j] = n_score
                texts.append(text)
                
            
            n_best_scores = torch.tensor(n_best_scores)
            n_best_score_dict[split][indice] = (n_best_scores, texts)
            
        else:
            n_best_scores, texts = n_best_score_dict[split][indice]
        
        return n_best_scores.to(self.device), texts
    
    
    def forward(self, feats, idxs, input_lengths, indices, split):
        """ Fusion multi-modal inputs
        Args:
            feats: acoustic feature (batch_size, N, input_dim)
            idxs: idx of lm tokens (batch_size, M)
            input_lengths: list of input lengths (batch_size)
            
        Returns:
            outputs: timing representation (N, encoding_dim)
        """
        b, n, h = feats.shape
        silences = None
        n_words = None               
        
        # silence encoding                            
        max_len = max(input_lengths)  
        if self.is_use_silence:
            with torch.no_grad():
                vad_preds = self.vad(feats, input_lengths)

            silences = torch.zeros([b, max_len, 1]).to(self.device)
            for i in range(b):                
                silence = self.get_silence(vad_preds[i][:input_lengths[i]], indices[i], split)                
                silences[i][:len(silence)] = silence                
        
        # Estimate Characters to the EoU
        n_words = torch.zeros([b, max_len, self.max_n_word]).to(self.device)
        texts_list = []
        if self.is_use_n_word:            
            for i in range(b): 
                if self.beam_width == 0: 
                    n_words = torch.zeros([b, max_len, 1]).to(self.device)
                    n_word = self.get_n_word(idxs[i], indices[i], split)
                else:                    
                    n_word, texts = self.get_n_word_n_best_scores(idxs[i], indices[i], split, max_len)
                                        
                n_words[i][:len(n_word)] = n_word
                texts_list.append(texts+['[PAD]']*(max_len-len(texts)))
        
        if silences is not None and n_words is not None:
            x_t = torch.cat([silences, n_words], dim=-1)            
        elif silences is not None:
            x_t = silences
        else:
            x_t = n_words
            
        r_t = self.linear(x_t)
        
        return r_t, texts_list
    
    
    def get_features(self, feats, idxs, input_lengths, indices, split):
        
        b, n, h = feats.shape
        silences = None
        n_words = None                    
        
        # silence encoding                            
        max_len = max(input_lengths)  
        if self.is_use_silence:
            with torch.no_grad():
                vad_preds = self.vad(feats, input_lengths)

            silences = torch.zeros([b, max_len, 1]).to(self.device)
            for i in range(b):                
                silence = self.get_silence(vad_preds[i][:input_lengths[i]], indices[i], split)                
                silences[i][:len(silence)] = silence                
        
        # Estimate Characters to the EoU        
        if self.is_use_n_word:
            n_words = torch.zeros([b, max_len, 1]).to(self.device)
            for i in range(b):                
                n_word = self.get_n_word(idxs[i], indices[i], split)
                n_words[i][:len(n_word)] = n_word
    
        return vad_preds, silences, n_words
    
    
class BeamSearchNode(object):
    def __init__(self, hidden_state, previousNode, wordId, prob, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.hidden_state = hidden_state
        self.prevNode = previousNode
        self.wordid = wordId
        self.prob = prob
        self.leng = length

#     def eval(self, alpha=1.0):
#         reward = 0
#         # Add here a function for shaping a reward

#         return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
    
    
# class SilenceEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=300):
#         super(SilenceEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.max_len = max_len
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)#.transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, src, silence):
#         tmp = torch.zeros(1, src.size(1), src.size(2))
#         for i, s in enumerate(silence): 
#             if s>0:
#                 if s >= self.max_len:
#                     tmp[:, i, :] = self.pe[:, int(self.max_len-1), :]
#                     #tmp[:, i, :] = self.pe[:, int(s):int(s)+1, :]
#                 else:
#                     tmp[:, i, :] = self.pe[:, int(s), :]
#                     #tmp[:, i, :] = self.pe[:, int(s):int(s)+1, :]

#         device = src.device
#         src = src + tmp.to(device)
#         return self.dropout(src)
    
    
# silence_dict_train={}
# silence_dict_val={}
# silence_dict_test={}

# silence_dict = {'train': silence_dict_train, 'val': silence_dict_val, 'test': silence_dict_test}


# class RTG(nn.Module):

#     def __init__(self, device, input_dim, hidden_dim, silence_encoding_type="concat"):
#         super().__init__()
        
#         self.device = device
#         self.silence_encoding_type = silence_encoding_type
#         if silence_encoding_type=="concat":
#             input_dim += 1

#         self.lstm = torch.nn.LSTM(
#                 input_size=input_dim,
#                 hidden_size=hidden_dim,
#                 batch_first=True,
#             )

#         self.fc = nn.Linear(hidden_dim, 1)
#         self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

# #     def get_silence_count(self, uttr_label):
# #         silence = torch.zeros(len(uttr_label)).to(self.device)
# #         silence[0] = uttr_label[0]
# #         for i, u in enumerate(uttr_label):
# #             if u == 1:
# #                 silence[i] = 0
# #             else:
# #                 silence[i] = silence[i-1]+1
# #         return silence.unsqueeze(1)
    
#     def get_silence(self, uttr_pred, indice, split):
        
#         if indice not in silence_dict[split]:
#             uttr_pred = torch.sigmoid(uttr_pred)
#             uttr_pred = uttr_pred.detach().cpu()
#             silence = torch.zeros(len(uttr_pred))#.to(self.device)
#             silence[0] = uttr_pred[0]
#             et = 0
#             cnt = 0
#             pre = 0
#             for i, u in enumerate(uttr_pred):
#                 uu = 1 - u
#                 et += uu  # 尤度を足す
#                 if uu > 0.5:  # 尤度が0.5以上なら非発話区間
#                     cnt = 0
#                 if pre >= 0.5 and uu < 0.5:  # 尤度が0.5以下になったらカウント開始
#                     cnt += 1
#                 elif uu < 0.5 and cnt > 0:  # カウントが始まっていればカウント
#                     cnt += 1
#                 if cnt > 25:  # 一定時間カウントされたらリセット
#                     et = 0
#                     cnt = 0

#                 silence[i] = et
#                 pre = uu
                
#             silence_dict[split][indice] = silence
            
#         else:
#             silence = silence_dict[split][indice]
        
#         return silence.unsqueeze(1).to(self.device)

#     def forward(self, inputs, uttr_preds, input_lengths, indices, split):
#         b, n, h = inputs.shape        
#         if self.silence_encoding_type=="concat":
#             max_len = max(input_lengths)
#             inputs_ = torch.zeros(b, max_len, h+1).to(self.device)
#             for i in range(b):
#                 silence = self.get_silence(uttr_preds[i][:input_lengths[i]], indices[i], split)
#                 inp = torch.cat([inputs[i][:input_lengths[i]], silence], dim=-1)
#                 inputs_[i, :input_lengths[i]] = inp
#         else:
#             inputs_ = inputs
#             pass
#             # raise Exception('Not implemented')

#         #inputs = inputs.unsqueeze(0)
        
#         t = max(input_lengths)
       
#         inputs = rnn_utils.pack_padded_sequence(
#             inputs_, 
#             input_lengths, 
#             batch_first=True,
#             enforce_sorted=False,
#         )

#         # outputs : batch_size x maxlen x hidden_dim
#         # rnn_h   : num_layers * num_directions, batch_size, hidden_dim
#         # rnn_c   : num_layers * num_directions, batch_size, hidden_dim
#         outputs, _ = self.lstm(inputs, None)
#         h, _ = rnn_utils.pad_packed_sequence(
#             outputs, 
#             batch_first=True,
#             padding_value=0.,
#             total_length=t,
#         )        
        
#         logits = self.fc(h)
#         b, n, c = logits.shape
#         logits = logits.view(b, -1)
#         return logits
   
#     def get_loss(self, probs, targets):
#         return self.criterion(probs, targets.float())




