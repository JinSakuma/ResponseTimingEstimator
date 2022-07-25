import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from src.utils.utils import edit_distance
from src.models.asr.transformer.encoder import Encoder
from src.models.asr.transformer.nets_utils import get_subsample
from src.models.asr.transformer.nets_utils import make_non_pad_mask

#pretrained_model = distilhubert()
#hubert = hubert()
#pretrained_model.eval()

idim=256
adim=256
odim=256
transformer_encpder_selfattn_layer_type="selfattn"
attention_dim=256
aheads=4
wshare=4
conv_kernel_length="21_23_25_27_29_31_33_35_37_39_41_43"
conv_usebias=False
eunits=1024
elayers=3
transformer_input_layer=None #"conv2d"
dropout_rate=0.1
positional_dropout_rate=0.1
attention_dropout_rate=0.0
stochastic_depth_rate=0.0
intermediate_ctc_layers=""
ctc_softmax=None
conditioning_layer_dim=odim


class CTC(nn.Module):
    """
    Connectionist Temporal Classification Model. Use a bidirectional
    LSTM as the encoder and a linear layer to decode to class probabilities.

    Args:
        input_dim: integer
                    number of input features
        num_class: integer
                    size of transcription vocabulary
        num_layers: integer (default: 2)
                    number of layers in encoder LSTM
        hidden_dim: integer (default: 128)
                    number of hidden dimensions for encoder LSTM
        bidirectional: boolean (default: True)
                        is the encoder LSTM bidirectional?
    """

    def __init__(
            self,
            input_dim,
            num_class,
            num_layers=2,
            hidden_dim=128,
            bidirectional=False,
        ):
        super().__init__()

        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=transformer_encpder_selfattn_layer_type,
            attention_dim=adim,
            attention_heads=aheads,
            conv_wshare=wshare,
            conv_kernel_length=conv_kernel_length,
            conv_usebias=conv_usebias,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=transformer_input_layer,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            stochastic_depth_rate=stochastic_depth_rate,
            intermediate_layers=None,
            ctc_softmax=None,
            conditioning_layer_dim=odim,
        )        

        self.fc1 = nn.Linear(input_dim, idim)
        self.fc2 = nn.Linear(odim, num_class)
        self.dropout = nn.Dropout()
       # self.input_dim = input_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = hidden_dim * num_layers * 2 * \
                             (2 if bidirectional else 1)
        
    def forward(self, inputs, input_lengths):
        out = self.fc1(inputs)
        mask = make_non_pad_mask(input_lengths.tolist()).to(out.device).unsqueeze(-2)
        out, _ = self.encoder(out, mask)
        logits = self.fc2(self.dropout(out))
        #logits = logits.view(batch_size, uttr_num, maxlen, self.num_class)
        
        log_probs = F.log_softmax(logits, dim=-1)
        # this embedding will be used for other transfer tasks

        return log_probs, input_lengths, out

    def get_loss(
            self,
            log_probs,
            input_lengths,
            labels,
            label_lengths,
            blank=0,
        ):
        log_probs = log_probs.permute(1, 0, 2)
        ctc_loss = F.ctc_loss(
            log_probs.contiguous(), 
            labels.long(), 
            input_lengths.long(), 
            label_lengths.long(), 
            blank=blank,
            zero_infinity=True,
        )
        return ctc_loss

    def decode(
            self, 
            log_probs, 
            input_lengths, 
            labels, 
            label_lengths,
            #sos_index=1,
            #eos_index=2, 
            #pad_index=3, 
            #eps_index=0,
        ):
        # Use greedy decoding.
        decoded = torch.argmax(log_probs, dim=2)
        batch_size = decoded.size(0)
        # Collapse each decoded sequence using CTC rules.
        hypotheses = []
        for i in range(batch_size):
            hypotheses_i = self.ctc_collapse(
                decoded[i], 
                input_lengths[i].item(),
            )
            hypotheses.append(hypotheses_i)

        hypothesis_lengths = input_lengths.cpu().numpy().tolist()
        if labels is None: # Run at inference time.
            references, reference_lengths = None, None
        else:
            references = labels.cpu().numpy().tolist()
            reference_lengths = label_lengths.cpu().numpy().tolist()

        return hypotheses, hypothesis_lengths, references, reference_lengths

    def ctc_collapse(self, seq, seq_len, blank_index=0):
        result = []
        for i, tok in enumerate(seq[:seq_len]):
            if tok.item() != blank_index:  # remove blanks
                if i != 0 and tok.item() == seq[i-1].item():  # remove dups
                    pass
                else:
                    result.append(tok.item())
        return result


