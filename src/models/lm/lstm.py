import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


# phoneme_list = ['a', 'i', 'u', 'e', 'o', 'a:', 'i:', 'u:', 'e:', 'o:', 
#                 'b', 'by', 'k', 'ky', 'g', 'gy', 's', 'sh', 'z', 'zy', 'j',
#                 't', 'ts', 'ty', 'ch', 'd', 'dy', 'n', 'ny', 'hy', 'p', 'py', 'm', 'my',
#                 'y', 'r', 'ry', 'w', 'f', 'q', 'N', ':', '<eou>', '<pad>'
#                ]


class LSTMModel(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, vocab_size, nlayers=1):
        super().__init__()
        
        self.device = device
        
        self.embed = torch.nn.Embedding(vocab_size, input_dim, padding_idx=vocab_size-1)
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=nlayers,
            )

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab_size-1).to(device)

    def forward(self, inputs, input_lengths):
        t = max(input_lengths)
        
        embs = self.embed(inputs)

        inputs = rnn_utils.pack_padded_sequence(
            embs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )
        
        # if len(inputs.shape)==2:
        #     inputs = inputs.unsqueeze(0)
        
        outputs, _ = self.lstm(inputs, None)
        outputs, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )
        
        logits = self.fc(outputs)
        return logits
    
    def forward_step(self, inputs):
        embs = self.embed(inputs)
        outputs, _ = self.lstm(embs, None)
        logits = self.fc(outputs)
                
        return logits

    def get_loss(self, logits, targets):
        return self.criterion(logits, targets.long())
        # return self.criterion(logits, targets.float())

    def get_acc(self, outputs, targets):
        # pred = torch.round(torch.sigmoid(outputs))
        _, pred = torch.max(outputs.data, 1)
        correct = (pred == targets).sum().float()
        acc = correct / targets.size(0)
        return acc.detach().cpu()
   