import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

torch.autograd.set_detect_anomaly(True)


class SATG(nn.Module):

    def __init__(self, device, in_size, out_size=1, n_unit=256, n_head=2, n_hid=256, n_layer=1, dropout=0.1):
        super(SATG, self).__init__()
        
        self.model_type = 'Transformer'
        self.in_size = in_size
        self.out_size = out_size
        self.n_unit = n_unit
        self.n_head = n_head
        self.n_layers = n_layer
        
        self.src_mask = None
        self.N = 100
        self.M = 0
        
        self.encoder = nn.Linear(in_size, n_unit)
        self.pos_encoder = PositionalEncoding(n_unit, dropout)
        encoder_layers = TransformerEncoderLayer(n_unit, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer)
        self.fc1 = nn.Linear(n_unit, n_unit//4)
        self.fc2 = nn.Linear(n_unit//4, out_size)
        
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

#         self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        N = self.N - 1
        M = self.M
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        for i in range(sz):
            if i>N:
                mask[i][i-N:i+M] = True
                mask[i][:i-N] = False
            else:
                mask[i][:i+M] = True
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        #mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.bias.data.zero_()
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, input_lengths):
        
        src_ = self.encoder(src)
        # src: (B, T, E)
        src_len = src_.size(1)            
        if (self.src_mask is None or self.src_mask.size(0) != src_.size(1)):
            device = src.device
            mask = self._generate_square_subsequent_mask(src_.size(1)).to(device)
            self.src_mask = mask
          
        # src: (T, B, E)
        src_ = src_.transpose(0, 1)
        # src: (T, B, E)
        src_ = self.pos_encoder(src_)
        # output: (T, B, E)
        output = self.transformer_encoder(src_, self.src_mask)
        output = output[-src_len:]
        # output: (B, T, E)
        output = output.transpose(0, 1)       
        
        output = self.fc1(output)
        logits = self.fc2(output)
        b, n, c = logits.shape
        logits = logits.view(b, -1)
        return logits
   
    def get_loss(self, probs, targets):
        return self.criterion(probs, targets.float())
    
    def get_attention_weight(self, src, N, M):
        
        N = self.N - 1
        M = self.M
        
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []
        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1].detach().cpu().numpy())
            
        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward_(src, N, mem_flg=mem_flg)

        for handle in handles:
            handle.remove()
#         self.train()

        return attn_weight


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2001):
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
        if x.size(0)>2000:
            print(x.shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# class RTGA(nn.Module):

#     def __init__(self, device, input_dim, hidden_dim):
#         super().__init__()
        
#         self.device = device
#         self.lstm = torch.nn.LSTM(
#                 input_size=input_dim,
#                 hidden_size=hidden_dim,
#                 batch_first=True,
#             )

#         self.attn = Attention(hidden_dim, hidden_dim, hidden_dim*2)
#         self.fc = nn.Linear(hidden_dim, 1)
#         self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

#     def forward(self, inputs, input_lengths):
#         b, n, h = inputs.shape                
#         t = max(input_lengths)
       
#         inputs = rnn_utils.pack_padded_sequence(
#             inputs, 
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
        
#         h = self.attn(h)
#         logits = self.fc(h)
#         b, n, c = logits.shape
#         logits = logits.view(b, -1)
#         return logits
   
#     def get_loss(self, probs, targets):
#         return self.criterion(probs, targets.float())
