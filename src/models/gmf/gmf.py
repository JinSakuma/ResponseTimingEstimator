import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
import transformers

# BERT = 'bert-base-uncased'
#BERT = 'cl-tohoku/bert-base-japanese-v2'
BERT = '/mnt/aoni04/jsakuma/transformers/'#'cl-tohoku/bert-base-japanese-v2'
tokenizer = BertTokenizer.from_pretrained(BERT, do_lower_case=True)
# config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#config = BertConfig()
# config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#bert_model = BertModel.from_pretrained(BERT, output_hidden_states=True, output_attentions=True)



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
        
        ntokens = len(tokenizer) # the size of vocabulary
        emsize = 300 # embedding dimension
        nout = encoding_dim # embedding dimension
        nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2 # the number of heads in the multiheadattention models
        dropout = 0.2 # the dropout value
        self.transformer = TransformerModel(ntokens, emsize, nout, nhead, nhid, nlayers, dropout)
        
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

        output = self.transformer(labels.to(self.device).transpose(0, 1), (1-masks).bool().to(self.device))
        pooled_output = output.transpose(0, 1)[:, 0, :]
        
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
                 encoding_dim,
                 weights):
        super().__init__()
        
        self.device = device
        
        self.fc_sa = nn.Linear(acoustic_dim+semantic_dim, encoding_dim)
        self.fc_st = nn.Linear(timing_dim+semantic_dim, encoding_dim)
        
        self.fc_g = nn.Linear(encoding_dim*2, encoding_dim, bias=False)
        self.fc_y = nn.Linear(encoding_dim, 2)
                
        self.criterion = nn.CrossEntropyLoss(weight=weights).to(device)
        
        
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

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask):
#         if self.src_mask is None or self.src_mask.size(0) != src.size(0):
#             device = src.device
#             mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
#             self.src_mask = mask

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
