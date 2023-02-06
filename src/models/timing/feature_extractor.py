import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

#from src.models.encoder.transformer_encoder import TransformerEncoder
from src.models.encoder.transformer_encoder_mytokenizer import TransformerEncoder
from src.models.encoder.acoustic_encoder import AcousticEncoder
from src.models.encoder.timing_encoder import TimingEncoder

torch.autograd.set_detect_anomaly(True)


class FeatureExtractor(nn.Module):

    def __init__(self, config, device, is_use_silence=True, is_use_n_word=False):
        super().__init__()
        
        self.device = device
        self.config = config
        self.is_use_silence = is_use_silence
        self.is_use_n_word = is_use_n_word
        
        self.create_models()

    def create_models(self):
        
        ae = AcousticEncoder(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.acoustic_hidden_dim,
            self.config.model_params.acoustic_encoding_dim,
        )
        self.acoustic_encoder = ae
        
        te = TimingEncoder(
            self.config,
            self.device,
            self.config.model_params.timing_input_dim,
            self.config.model_params.timing_encoding_dim,
            is_use_silence=self.is_use_silence,
            is_use_n_word=self.is_use_n_word,
        )
        self.timing_encoder = te       
        
        se = TransformerEncoder(
            self.config,
            self.device,
        )
        self.semantic_encoder = se
        
    def get_params(self):

        parameters = chain(
            self.acoustic_encoder.parameters(),
            self.timing_encoder.linear.parameters(),
            self.semantic_encoder.parameters(),
        )
        return parameters

    def forward(self, feats, idxs, input_lengths, texts, indices, split):
        
        r_a = self.acoustic_encoder(feats, input_lengths)
#         r_s = self.semantic_encoder(texts)
        r_s = self.semantic_encoder(idxs, input_lengths)  # from src.models.encoder.transformer_encoder_mytokenizer import TransformerEncoderのbaai
        r_t = self.timing_encoder(feats, idxs, input_lengths, indices, split)   
        
        embs = torch.cat([r_s, r_a, r_t], dim=-1)              
        return embs
    
    def reset_state(self):
        self.acoustic_encoder.reset_state()
        self.timing_encoder.reset_state()
