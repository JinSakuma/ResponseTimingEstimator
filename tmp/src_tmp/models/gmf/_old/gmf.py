import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
import transformers

# BERT = 'bert-base-uncased'
BERT = 'cl-tohoku/bert-base-japanese-v2'
tokenizer = BertTokenizer.from_pretrained(BERT, do_lower_case=True)
# config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
config = BertConfig()
# config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
bert_model = BertModel.from_pretrained(BERT, output_hidden_states=True, output_attentions=True)



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
    
    
class SemanticEncoder(nn.Module):

    def __init__(self, device, bert_hidden_dim, encoding_dim):
        super().__init__()
        self.device = device
        self.max_length = 70
        
        self.linear = nn.Linear(bert_hidden_dim,
                                encoding_dim,
                               )
        
    def forward(self, inputs):
        """ Fusion multi-modal inputs
        Args:
            inputs: list of text (N,)
            
        Returns:
            pooled_output: semantic representation (N, encoding_dim)
        """
            
        result = tokenizer(inputs,
                           max_length=self.max_length,
                           padding="max_length",
                           truncation=True,
                           return_tensors='pt')
        
        labels = result['input_ids']
        masks = result['attention_mask']

        bert_model.to(self.device)
        output = bert_model(labels.to(self.device), attention_mask=masks.to(self.device))
        pooled_output = output[1]
        pooled_output = self.linear(pooled_output)
        
        return pooled_output
    
    
class TimingEncoder(nn.Module):

    def __init__(self, device, input_dim, encoding_dim):
        super().__init__()
        self.device = device
        self.max_length = 70
        
        self.linear = nn.Linear(input_dim,
                                encoding_dim,
                               )
        
    def forward(self, inputs):
        """ Fusion multi-modal inputs
        Args:
            inputs: timing feature (N, input_dim)
            
        Returns:
            outputs: timing representation (N, encoding_dim)
        """
            
        outputs = self.linear(inputs)
        
        return outputs


class GatedFusionBlock(nn.Module):

    def __init__(self,
                 device,
                 acoustic_dim,
                 semantic_dim,
                 timing_dim,
                 encoding_dim):
        super().__init__()
        
        self.device = device
        
        self.fc_sa = nn.Linear(acoustic_dim+semantic_dim, encoding_dim)
        self.fc_st = nn.Linear(timing_dim+semantic_dim, encoding_dim)
        
        self.fc_g = nn.Linear(encoding_dim*2, encoding_dim, bias=False)
        self.fc_y = nn.Linear(encoding_dim, 2)
        
        self.criterion = nn.CrossEntropyLoss().to(device)
        
        
    def forward(self, r_a, r_s, r_t):
        """ Fusion multi-modal inputs
        Args:
            r_a: acoustic representations (N, D)
            r_s: semantic representations (N, D)
            r_t: timing representations   (N, D)
            
        Returns:
            (N, 2)
        """
        
        r_sa = torch.cat([r_s, r_a], dim=-1)
        r_sa = self.fc_sa(r_sa)
        
        r_st = torch.cat([r_s, r_t], dim=-1)
        r_st = self.fc_sa(r_st)
        
        r_sat = torch.cat([r_sa, r_st], dim=-1)
        g = torch.sigmoid(self.fc_g(r_sat))
        r = g * r_sa + (1-g) * r_st
        y = torch.sigmoid(self.fc_y(r))
        
        return y
   
    def get_loss(self, probs, targets):
        return self.criterion(probs, targets)
