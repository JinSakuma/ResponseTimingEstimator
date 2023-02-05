import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


class TimingEstimator(nn.Module):

    def __init__(self, device, input_dim, hidden_dim):
        super().__init__()
        
        self.device = device
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        self.fc = nn.Linear(hidden_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
        self.reset_state()

    def forward(self, inputs, input_lengths):
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
        outputs, self.hidden_state = self.lstm(inputs, self.hidden_state)
        h, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )        
        
        logits = self.fc(h)
        b, n, c = logits.shape
        logits = logits.view(b, -1)
        return logits
    
    def reset_state(self):
        self.hidden_state = None
   
    def get_loss(self, probs, targets):
        return self.criterion(probs, targets.float())
