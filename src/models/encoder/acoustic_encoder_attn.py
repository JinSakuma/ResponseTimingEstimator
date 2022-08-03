import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch import einsum
from itertools import chain

torch.autograd.set_detect_anomaly(True)


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)
        self.mask = None
        
        self.N = 100
        self.M = 0
        
    def generate_square_subsequent_mask(self, inp, N, M=0):
        b, sz, d = inp.shape
        N -= 1
        N = min(sz, N)
        mask = (torch.triu(torch.ones(sz, sz)) == 0).transpose(0, 1)
        for i in range(sz):
            if i>N:
                mask[i][i-N:i+M] = False
                mask[i][:i-N] = True
            else:
                mask[i][:i+M] = False
        
        return mask

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        #if self.causal:
        mask = self.generate_square_subsequent_mask(sim, self.N, self.M).to(device)
        #mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
        sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out), attn


class AcousticEncoder(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, encoding_dim):
        super().__init__()
        
        self.device = device
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
       
        self.attn = Attention(hidden_dim, hidden_dim, hidden_dim*2)
        self.fc = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, inputs, input_lengths):
        """ Fusion multi-modal inputs
        Args:
            inputs: acoustic feature (B, L, input_dim)
            
        Returns:
            logits: acoustic representation (B, L, encoding_dim)
        """
        b, n, h = inputs.shape        
        t = max(input_lengths)
       
        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )

        # outputs : batch_size x maxlen x hidden_dim
        # rnn_h   : num_layers * num_directions, batch_size, hidden_dim
        # rnn_c   : num_layers * num_directions, batch_size, hidden_dim
        outputs, _ = self.lstm(inputs, None)
        h, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )        
        
        h_attn, attn = self.attn(h)
        h = h + h_attn
        logits = self.fc(h)
        return logits, attn
