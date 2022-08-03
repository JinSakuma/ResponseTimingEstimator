import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.rtg import RTG

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


class BERTEncoder(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.max_length = 70
        
        self.linear = nn.Linear(self.config.model_params.bert_hidden_dim,
                                self.config.model_params.semantic_encoding_dim,
                               )
        
    def forward(self, inputs):
        
        batch_size = len(inputs)
        bert_inputs_list = []
        bert_input_lengths_list = []
        batch_idx_list = []
        for i in range(batch_size):
            text = inputs[i]
            transcripts = []
            idx_list = []
            pre = None
            for idx, t in enumerate(text):
                if t == '[PAD]':
                    pass
                elif pre != t:
                    transcripts.append(t)
                    idx_list.append(idx)
                    pre = t
            idx_list.append(len(text)) 

            bert_inputs_list += transcripts
            bert_input_lengths_list.append(len(transcripts))
            batch_idx_list.append(idx_list)
            
        result = tokenizer(bert_inputs_list,
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
        
        bacth_bert_feat = []
        for i in range(batch_size):
            idx_list = batch_idx_list[i]
            start_idx = sum(bert_input_lengths_list[:i])

            bert_feat = []
            for j, idx in enumerate(idx_list[:-1]):
                bert_feat[idx_list[j]:idx_list[j+1]] = [pooled_output[start_idx+j].unsqueeze(0)]*(idx_list[j+1]-idx_list[j])

            bert_feat = torch.cat(bert_feat)
            bacth_bert_feat.append(bert_feat.unsqueeze(0))

        bacth_bert_feat = torch.cat(bacth_bert_feat)
        
        return bacth_bert_feat
