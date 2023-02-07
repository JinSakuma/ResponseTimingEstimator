# IPU Lengthを入れる

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
            if device == torch.device('cpu'):
                self.load_state_dict(torch.load(vad_model_path, map_location=torch.device('cpu')), strict=False)
            else:            
#                 self.load_state_dict(torch.load(vad_model_path), strict=False)
                self.load_state_dict(torch.load(vad_model_path), strict=False)
        
        if is_use_n_word:
            lm_config_path = config.model_params.lm_config_path
            lm_model_path = config.model_params.lm_model_path 
            print('LM pretrained weight path')
            print(lm_model_path)
            lm_config = load_config(lm_config_path)    
            self.lm = LSTMLM(lm_config, device)
            if device == torch.device('cpu'):           
                self.lm.load_state_dict(torch.load(lm_model_path, map_location=torch.device('cpu')))#, strict=False)
            else:
                self.lm.load_state_dict(torch.load(lm_model_path)) #, strict=False)
        
        self.linear = nn.Linear(input_dim,
                                encoding_dim,
                               )
        
        self.set_cash()
        self.reset_state()

    def set_cash(self):       
        self.silence_dict = {'train': {}, 'val': {}, 'test': {}}
        self.n_word_dict = {'train': {}, 'val': {}, 'test': {}}
        self.n_best_score_dict = {'train': {}, 'val': {}, 'test': {}} 
        
    def reset_state(self):
        self.vad.reset_state()
        self.cnt = 0        
        self.pre = 0
        self.et = 0
        self.ul = 0
        self.total = 0
        self.pre_token = [0]
        
    def get_silence(self, uttr_pred, indice, split, streaming_inference=False):
        """ Fusion multi-modal inputs
        Args:
            vad_pred: acoustic feature (N, 1)
            indice: data idx 
            split: training phase
            
        Returns:
            outputs: silence count (N, 1)
        """
        
        if (indice not in self.silence_dict[split]) or streaming_inference:
            uttr_pred = torch.sigmoid(uttr_pred)
            uttr_pred = uttr_pred.detach().cpu()
            silence = torch.zeros(len(uttr_pred))#.to(self.device)
            utter_length = torch.zeros(len(uttr_pred))#.to(self.device)
            silence[0] = uttr_pred[0]                       
            for i, u in enumerate(uttr_pred):
                uu = 1 - u
                self.et += uu  # 尤度を足す
                if uu > 0.5:  # 尤度が0.5以上なら非発話区間
                    self.cnt = 0
                    self.ul = 0
                else:
                    self.ul += 1
                    
                if self.pre >= 0.5 and uu < 0.5:  # 尤度が0.5以下になったらカウント
                    self.cnt += 1                    
                elif uu < 0.5 and self.cnt > 0:  # カウントが始まっていればカウント
                    self.cnt += 1
                    
                if self.cnt > self.reset_frame:  # 一定時間カウントされたらリセット
                    self.et = 0
                    self.cnt = 0                    

                silence[i] = self.et
                utter_length[i] = self.ul
                self.pre = uu
                self.total += 1
                
            self.silence_dict[split][indice] = (silence, utter_length)
            
        else:
            silence, utter_length = self.silence_dict[split][indice]
        
        dialog_length = torch.tensor(np.arange(len(silence)))
        return silence.unsqueeze(1).to(self.device), utter_length.unsqueeze(1).to(self.device), dialog_length.unsqueeze(1).to(self.device)
    
    def get_silence_streaming(self, uttr_pred, indice, split, streaming_inference=False):
        """ Fusion multi-modal inputs
        Args:
            vad_pred: acoustic feature (N, 1)
            indice: data idx 
            split: training phase
            
        Returns:
            outputs: silence count (N, 1)
        """
        
        if (indice not in self.silence_dict[split]) or streaming_inference:
            uttr_pred = torch.sigmoid(uttr_pred)
            uttr_pred = uttr_pred.detach().cpu()
            silence = torch.zeros(len(uttr_pred))#.to(self.device)
            utter_length = torch.zeros(len(uttr_pred))#.to(self.device)
            silence[0] = uttr_pred[0]                       
            for i, u in enumerate(uttr_pred):
                uu = 1 - u
                self.et += uu  # 尤度を足す
                if uu > 0.5:  # 尤度が0.5以上なら非発話区間
                    self.cnt = 0
                    self.ul = 0
                else:
                    self.ul += 1
                    
                if self.pre >= 0.5 and uu < 0.5:  # 尤度が0.5以下になったらカウント
                    self.cnt += 1                    
                elif uu < 0.5 and self.cnt > 0:  # カウントが始まっていればカウント
                    self.cnt += 1
                    
                if self.cnt > self.reset_frame:  # 一定時間カウントされたらリセット
                    self.et = 0
                    self.cnt = 0                    

                silence[i] = self.et
                utter_length[i] = self.ul
                self.pre = uu
                self.total += 1
                
            self.silence_dict[split][indice] = (silence, utter_length)
            
        else:
            silence, utter_length = self.silence_dict[split][indice]
               
        dialog_length = torch.tensor([self.total])
        return silence.unsqueeze(1).to(self.device), utter_length.unsqueeze(1).to(self.device), dialog_length.unsqueeze(1).to(self.device)    
    
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
        
        if indice not in self.n_word_dict[split]:
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
            self.n_word_dict[split][indice] = n_word
            
            
        else:
            n_word = self.n_word_dict[split][indice]
        
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
    
    def get_n_word_n_best_scores(self, idxs, indice, split):
        """ Fusion multi-modal inputs
        Args:
            vad_pred: acoustic feature (N, 1)
            indice: data idx 
            split: training phase
            
        Returns:
            outputs: silence count (N, beam_width)
        """
        
        if indice not in self.n_best_score_dict[split]:

            pre = [0]
            n_best_scores = np.zeros([len(idxs), self.max_n_word])
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
                        
                else:
                    pass

                n_best_scores[j] = n_score
#                 texts.append(text)
                
            
            n_best_scores = torch.tensor(n_best_scores)
            self.n_best_score_dict[split][indice] = n_best_scores
            
        else:
            n_best_scores = self.n_best_score_dict[split][indice]
        
        return n_best_scores.to(self.device)
    
    def get_n_word_n_best_scores_streaming(self, idxs, indice, split):
        """ Fusion multi-modal inputs
        Args:
            vad_pred: acoustic feature (N, 1)
            indice: data idx 
            split: training phase
            
        Returns:
            outputs: silence count (N, beam_width)
        """
                
        self.pre_token = [0]
        n_best_scores = np.zeros([len(idxs), self.max_n_word])
        n_score = np.zeros(self.max_n_word)
        n_score[-1] = 1
        for j in range(len(idxs)):          
            if idxs[j] != self.pre_token:
                inp = torch.tensor(idxs[j])
                self.pre_token = idxs[j]

                endnodes = self.get_n_best_length(inp.to(self.device))
                context = inp.detach().cpu().tolist()

                utterances = []
                scores = []
                n_score = np.zeros(self.max_n_word)
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

            else:
                pass

            n_best_scores[j] = n_score


        n_best_scores = torch.tensor(n_best_scores)
        
        return n_best_scores.to(self.device)
    
    def forward(self, feats, idxs, input_lengths, indices, split, debug=False):
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
            utter_lengths = torch.zeros([b, max_len, 1]).to(self.device)
            dialog_lengths = torch.zeros([b, max_len, 1]).to(self.device)
            for i in range(b):                
                silence, utter_length, dialog_length = self.get_silence(vad_preds[i][:input_lengths[i]], indices[i], split)                
                silences[i][:len(silence)] = silence
                utter_lengths[i][:len(utter_length)] = utter_length
                dialog_lengths[i][:len(dialog_length)] = dialog_length
                
            silences = torch.cat([silences, utter_lengths, dialog_lengths], dim=-1)                
                
            self.vad.reset_state()
        
        # Estimate Characters to the EoU        
        if self.is_use_n_word:
            n_words = torch.zeros([b, max_len, self.max_n_word]).to(self.device)
            for i in range(b): 
                if self.beam_width == 0: 
                    n_words = torch.zeros([b, max_len, 1]).to(self.device)
                    n_word = self.get_n_word(idxs[i], indices[i], split)
                else:                    
                    n_word = self.get_n_word_n_best_scores(idxs[i], indices[i], split)

                n_words[i][:len(n_word)] = n_word
        
        if silences is not None and n_words is not None:
            x_t = torch.cat([silences, n_words], dim=-1)
        elif silences is not None:
            x_t = silences
        else:
            x_t = n_words
            
        r_t = self.linear(x_t) 
        
        if debug:
            return r_t, silences#torch.sigmoid(vad_preds)
        
        return r_t
    
    def streaming_inference(self, feats, idxs, input_lengths, indices, split, debug=False):
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
            utter_lengths = torch.zeros([b, max_len, 1]).to(self.device)
            dialog_lengths = torch.zeros([b, max_len, 1]).to(self.device)
            for i in range(b):                
                silence, utter_length, dialog_length = self.get_silence_streaming(vad_preds[i][:input_lengths[i]], indices[i], split, streaming_inference=True)                
                silences[i][:len(silence)] = silence
                utter_lengths[i][:len(utter_length)] = utter_length
                dialog_lengths[i][:len(dialog_length)] = dialog_length
                
            silences = torch.cat([silences, utter_lengths, dialog_lengths], dim=-1)                
                
            self.vad.reset_state()
        
        # Estimate Characters to the EoU        
        if self.is_use_n_word:
            n_words = torch.zeros([b, max_len, self.max_n_word]).to(self.device)
            for i in range(b): 
                if self.beam_width == 0: 
                    n_words = torch.zeros([b, max_len, 1]).to(self.device)
                    n_word = self.get_n_word(idxs[i], indices[i], split)
                else:                    
                    n_word = self.get_n_word_n_best_scores_streaming(idxs[i], indices[i], split)

                n_words[i][:len(n_word)] = n_word
        
        if silences is not None and n_words is not None:
            x_t = torch.cat([silences, n_words], dim=-1)
        elif silences is not None:
            x_t = silences
        else:
            x_t = n_words
            
        r_t = self.linear(x_t) 
        
        if debug:
            return r_t, silences, torch.sigmoid(vad_preds)
        
        return r_t
    
#     def streaming_inference(self, feats, idxs, input_lengths, indices, split='val'):
#         """ Fusion multi-modal inputs
#         Args:
#             feats: acoustic feature (batch_size, N, input_dim)
#             idxs: idx of lm tokens (batch_size, M)
#             input_lengths: list of input lengths (batch_size)
            
#         Returns:
#             outputs: timing representation (N, encoding_dim)
#         """
#         b, n, h = feats.shape
#         silences = None
#         n_words = None               
        
#         # silence encoding                            
#         max_len = max(input_lengths)
#         if self.is_use_silence:
#             with torch.no_grad():
#                 vad_preds = self.vad(feats, input_lengths)

#             silences = torch.zeros([b, max_len, 1]).to(self.device)
#             for i in range(b):                
#                 silence = self.get_silence(vad_preds[i][:input_lengths[i]], indices[i], split)                
#                 silences[i][:len(silence)] = silence                
        
#         # Estimate Characters to the EoU        
#         if self.is_use_n_word:
#             n_words = torch.zeros([b, max_len, self.max_n_word]).to(self.device)
#             for i in range(b): 
#                 if self.beam_width == 0: 
#                     n_words = torch.zeros([b, max_len, 1]).to(self.device)
#                     n_word = self.get_n_word(idxs[i], indices[i], split)
#                 else:                    
#                     n_word = self.get_n_word_n_best_scores_inference(idxs[i], indices[i], split)

#                 n_words[i][:len(n_word)] = n_word
        
#         if silences is not None and n_words is not None:
#             x_t = torch.cat([silences, n_words], dim=-1)            
#         elif silences is not None:
#             x_t = silences
#         else:
#             x_t = n_words
            
#         r_t = self.linear(x_t)
        
#         return r_t, silence    
    
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
