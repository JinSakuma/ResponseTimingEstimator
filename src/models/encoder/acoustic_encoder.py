import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


class AcousticEncoder(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, encoding_dim):
        super().__init__()
        
        self.device = device
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

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
        
        logits = self.fc(h)
        return logits 
