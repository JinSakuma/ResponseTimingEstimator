"""Transducer model arguments."""

import argparse
from argparse import _ArgumentGroup
import ast
from distutils.util import strtobool
from src.models.asr.transducer.rnn_transducer import RNNT
import torch
import torch.nn as nn

#idim=256
#odim=256

#parser = argparse.ArgumentParser(description='This script is ...')
#parser.add_argument('--test', type=str, default='a')
#parser.add_argument('--report_cer', type=bool, default=False)
#parser.add_argument('--report_wer', type=bool, default=False)


class RNNTModel(nn.Module):
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
            idim,
            odim,
            args,
        ):
        super().__init__()

        self.model = RNNT(idim, odim, args)

    def forward(self, inputs, input_lengths, labels, label_lengths, split):
        if split=='train':
            is_training = True
        else:
            is_training = False

        loss, cer, wer = self.model(inputs, input_lengths, labels, label_lengths, is_training)
        return loss, wer


