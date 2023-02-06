import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nout, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, nout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = self.decoder(output)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.max_length = 70
                        
        path = self.config.data_params.token_list_path
        with open(path) as f:
            lines =f.readlines()
        self.tokens = [line.split()[0] for line in lines]
        ntokens = len(self.tokens) # the size of vocabulary
                
        emsize = 300 # embedding dimension
        nout = self.config.model_params.semantic_encoding_dim # embedding dimension
        nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2 # the number of heads in the multiheadattention models
        dropout = 0.2 # the dropout value
        self.transformer = TransformerModel(ntokens, emsize, nout, nhead, nhid, nlayers, dropout)
        
#         self.linear = nn.Linear(self.config.model_params.bert_hidden_dim,
#                                 self.config.model_params.semantic_encoding_dim,
#                                )

    def tokenize(self, inputs, maxlength=70):
        labels = []
        masks = []
        for inp in inputs:
            if len(inp) < maxlength:
                labels.append(inp+[0]*(maxlength-len(inp)))
                masks.append([1]*len(inp)+[0]*(maxlength-len(inp)))
            else:
                labels.append(inp[-maxlength:])
                masks.append([1]*maxlength)

        return torch.tensor(labels), torch.tensor(masks)
        
        
    def forward(self, inputs, input_lengths):
        
        max_input_length = max(input_lengths)
        batch_size = len(inputs)
        inputs_list = []
        input_lengths_list = []
        batch_idx_list = []
        for i in range(batch_size):
            text = inputs[i]
            transcripts = []
            idx_list = []
            pre = None
            for idx, t in enumerate(text):
#                 if t == '[PAD]':
#                     pass
                if pre != t:
                    transcripts.append(t)
                    idx_list.append(idx)
                    pre = t
            idx_list.append(max_input_length) 

            inputs_list += transcripts
            input_lengths_list.append(len(transcripts))
            batch_idx_list.append(idx_list)
        
        labels, masks = self.tokenize(inputs_list, self.max_length)        

        output = self.transformer(labels.to(self.device).transpose(0, 1), (1-masks).bool().to(self.device))
        pooled_output = output.transpose(0, 1)[:, 0, :]
        #pooled_output = self.linear(pooled_output)
        
        bacth_feat = []
        for i in range(batch_size):
            idx_list = batch_idx_list[i]
            start_idx = sum(input_lengths_list[:i])

            feat = []
            for j, idx in enumerate(idx_list[:-1]):
                feat[idx_list[j]:idx_list[j+1]] = [pooled_output[start_idx+j].unsqueeze(0)]*(idx_list[j+1]-idx_list[j])

            feat = torch.cat(feat)
            bacth_feat.append(feat.unsqueeze(0))

        bacth_feat = torch.cat(bacth_feat)
        
        return bacth_feat
