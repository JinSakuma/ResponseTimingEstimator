# IPU Lengthを入れる

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.timing.timing_estimator import TimingEstimator
from src.models.timing.feature_extractor import FeatureExtractor

torch.autograd.set_detect_anomaly(True)


class System(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device               
        
        self.R = 60
        
        self.create_models()

    def create_models(self):
        
        encoding_dim = self.config.model_params.acoustic_encoding_dim \
                        + self.config.model_params.semantic_encoding_dim \
                        + self.config.model_params.timing_encoding_dim
        
        timing_estimator = TimingEstimator(
            self.device,
            encoding_dim,
            self.config.model_params.hidden_dim,
        )
        self.timing_model = timing_estimator
        
        feature_extractor = FeatureExtractor(
            self.config,
            self.device,
            is_use_silence=True,
            is_use_n_word=True,
        )
        self.feature_extractor = feature_extractor

    def configure_optimizer_parameters(self):

        parameters = chain(
            self.timing_model.parameters(),
            self.feature_extractor.get_params(),            
        )
        return parameters

    def forward(self, batch, split='train'):
        chs = batch[0]
        texts = batch[1]
        kanas = batch[2]
        idxs = batch[3]
        vad = batch[4]
        #turn = batch[5].to(self.device)
        #last_ipu = batch[6].to(self.device)
        targets = batch[7].to(self.device)
        feats = batch[8].to(self.device)
        input_lengths = batch[9] #.to(self.device)
        offsets = batch[10] #.to(self.device)
        indices = batch[11] #.to(self.device)
        batch_size = int(len(chs))
        
        embs = self.feature_extractor(feats, idxs, input_lengths, texts, indices, split)                    
        outputs = self.timing_model(embs, input_lengths)
        
        loss, acc = 0, 0
        for i in range(batch_size):
            loss = loss+self.timing_model.get_loss(outputs[i][:input_lengths[i]], targets[i][:input_lengths[i]])

        outputs = {
            f'{split}_loss': loss,
        }

        self.reset_state()
        return outputs
    
    def reset_state(self):
        self.feature_extractor.reset_state()
        self.timing_model.reset_state()
