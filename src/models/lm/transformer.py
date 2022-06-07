import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=600):
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


class TransformerModel(nn.Module):

    def __init__(self, device, vocab_size, d_model, n_head=2, n_hid=256, n_layer=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_hid = n_hid
        self.n_head = n_head
        self.n_layers = n_layer
        
        self.src_mask = None        
        self.encoder = torch.nn.Embedding(vocab_size, d_model, padding_idx=vocab_size-1)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer)
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab_size-1).to(device)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz, N, M=0):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
#         N = N-1
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         for i in range(sz):
#             if i>N:
#                 mask[i][i-N:i+M] = True
#                 mask[i][:i-N] = False
#             else:
#                 mask[i][:i+M] = True
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, N=30):
        src = self.encoder(src) * math.sqrt(self.d_model)
        # src: (B, T, E)
        src_len = src.size(1)
        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(1), N).to(device)
            self.src_mask = mask
          
        # src: (T, B, E)
        src = src.transpose(0, 1)
        # src: (T, B, E)
        src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, self.src_mask)
        # output: (B, T, E)
        output = output.transpose(0, 1)
        # output: (B, T, C)
        output = self.decoder(output)
        
        return output
    
    def get_attention_weight(self, src, N=30, M=0):
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
            self.forward_(src, N)

        for handle in handles:
            handle.remove()

        return attn_weight

    def get_loss(self, logits, targets):
        return self.criterion(logits, targets.long())