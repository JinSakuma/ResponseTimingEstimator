import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


class VAD(nn.Module):

    def __init__(self, device, input_dim, hidden_dim): #, silence_encoding_type="concat"):
        super().__init__()
        
        self.device = device
        self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )

        self.fc = nn.Linear(hidden_dim, 3)
        self.criterion = nn.CrossEntropyLoss().to(device)
        # self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)

    def forward(self, inputs, input_lengths):
        t = max(input_lengths)
        batch = inputs.size(0)
        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
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
        #print(logits.shape)
        #logits = logits.view(batch, -1)
        return logits

    def recog(self, inputs, input_lengths):
        outs = []
        with torch.no_grad():
            for i in range(len(input_lengths)):
                output = self.forward(inputs[i][:input_lengths[i]])
                outs.append(torch.sigmoid(output))            
        return outs

    def get_loss(self, logits, targets):
        return self.criterion(logits, targets.long())
        # return self.criterion(logits, targets.float())

    def get_acc(self, outputs, targets):
        # pred = torch.round(torch.sigmoid(outputs))
        _, pred = torch.max(outputs.data, 1)
        correct = (pred == targets).sum().float()
        acc = correct / targets.size(0)
        return acc.detach().cpu()
