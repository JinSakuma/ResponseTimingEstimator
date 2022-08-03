"""Transducer speech recognition model (pytorch)."""

from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import asdict
import logging
import math
import numpy
from typing import List

import chainer
import torch

from src.models.asr.transducer.beam_search_transducer import BeamSearchTransducer
from src.models.asr.nets_utils import get_subsample, make_non_pad_mask, fill_missing_args
from src.models.asr.transducer.arguments import (
    add_auxiliary_task_arguments,  # noqa: H301
    add_custom_decoder_arguments,  # noqa: H301
    add_custom_encoder_arguments,  # noqa: H301
    add_custom_training_arguments,  # noqa: H301
    add_decoder_general_arguments,  # noqa: H301
    add_encoder_general_arguments,  # noqa: H301
    add_rnn_decoder_arguments,  # noqa: H301
    add_rnn_encoder_arguments,  # noqa: H301
    add_transducer_arguments,  # noqa: H301
)
from src.models.asr.transducer.error_calculator import ErrorCalculator
from src.models.asr.transducer.initializer import initializer
from src.models.asr.transducer.rnn_decoder import RNNDecoder
from src.models.asr.transducer.rnn_encoder import encoder_for
from src.models.asr.transducer.transducer_tasks import TransducerTasks
from src.models.asr.transducer.utils import get_decoder_input
from src.models.asr.transducer.utils import valid_aux_encoder_output_layers


class RNNT(torch.nn.Module):
    """E2E module for Transducer models.
    Args:
        idim: Dimension of inputs.
        odim: Dimension of outputs.
        args: Namespace containing model options.
        ignore_id: Padding symbol ID.
        blank_id: Blank symbol ID.
        training: Whether the model is initialized in training or inference mode.
    """

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for Transducer model."""
        RNNT.encoder_add_general_arguments(parser)
        RNNT.encoder_add_rnn_arguments(parser)
        #RNNT.encoder_add_custom_arguments(parser)

        RNNT.decoder_add_general_arguments(parser)
        RNNT.decoder_add_rnn_arguments(parser)
        #RNNT.decoder_add_custom_arguments(parser)

        #RNNT.training_add_custom_arguments(parser)
        RNNT.transducer_add_arguments(parser)
        RNNT.auxiliary_task_add_arguments(parser)

        return parser

    @staticmethod
    def encoder_add_general_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add general arguments for encoder."""
        group = parser.add_argument_group("Encoder general arguments")
        group = add_encoder_general_arguments(group)

        return parser

    @staticmethod
    def encoder_add_rnn_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for RNN encoder."""
        group = parser.add_argument_group("RNN encoder arguments")
        group = add_rnn_encoder_arguments(group)

        return parser

   # @staticmethod
   # def encoder_add_custom_arguments(parser: ArgumentParser) -> ArgumentParser:
   #     """Add arguments for Custom encoder."""
   #     group = parser.add_argument_group("Custom encoder arguments")
   #     group = add_custom_encoder_arguments(group)

   #     return parser

    @staticmethod
    def decoder_add_general_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add general arguments for decoder."""
        group = parser.add_argument_group("Decoder general arguments")
        group = add_decoder_general_arguments(group)

        return parser

    @staticmethod
    def decoder_add_rnn_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for RNN decoder."""
        group = parser.add_argument_group("RNN decoder arguments")
        group = add_rnn_decoder_arguments(group)

        return parser

    #@staticmethod
    #def decoder_add_custom_arguments(parser: ArgumentParser) -> ArgumentParser:
    #    """Add arguments for Custom decoder."""
    #    group = parser.add_argument_group("Custom decoder arguments")
    #    group = add_custom_decoder_arguments(group)

    #    return parser

    #@staticmethod
    #def training_add_custom_arguments(parser: ArgumentParser) -> ArgumentParser:
    #    """Add arguments for Custom architecture training."""
    #    group = parser.add_argument_group("Training arguments for custom archictecture")
    #    group = add_custom_training_arguments(group)

    #    return parser

    @staticmethod
    def transducer_add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for Transducer model."""
        group = parser.add_argument_group("Transducer model arguments")
        group = add_transducer_arguments(group)

        return parser

    @staticmethod
    def auxiliary_task_add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for auxiliary task."""
        group = parser.add_argument_group("Auxiliary task arguments")
        group = add_auxiliary_task_arguments(group)

        return parser

    #@property
    #def attention_plot_class(self):
    #    """Get attention plot class."""
    #    return PlotAttentionReport

    def get_total_subsampling_factor(self) -> float:
        """Get total subsampling factor."""
        if self.etype == "custom":
            return self.encoder.conv_subsampling_factor * int(
                numpy.prod(self.subsample)
            )
        else:
            return self.enc.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(
        self,
        idim: int,
        odim: int,
        args: Namespace,
        ignore_id: int = -1,
        blank_id: int = 0,
        training: bool = True,
    ):
        """Construct an E2E object for Transducer model."""
        torch.nn.Module.__init__(self)

        args = fill_missing_args(args, self.add_arguments)

        self.is_transducer = True

        with open('src/datasets/tokens.txt') as f:
            self.char_list = f.read().split("\n")

        self.use_auxiliary_enc_outputs = (
            True if (training and args.use_aux_transducer_loss) else False
        )

        self.subsample = get_subsample(
            args, mode="asr", arch="transformer" if args.etype == "custom" else "rnn-t"
        )

        if self.use_auxiliary_enc_outputs:
            n_layers = (
                ((len(args.enc_block_arch) * args.enc_block_repeat) - 1)
                if args.enc_block_arch is not None
                else (args.elayers - 1)
            )

            aux_enc_output_layers = valid_aux_encoder_output_layers(
                args.aux_transducer_loss_enc_output_layers,
                n_layers,
                args.use_symm_kl_div_loss,
                self.subsample,
            )
        else:
            aux_enc_output_layers = []

        self.enc = encoder_for(
            args,
            idim,
            self.subsample,
            aux_enc_output_layers=aux_enc_output_layers,
        )
        encoder_out = args.eprojs

        self.dec = RNNDecoder(
            odim,
            args.dtype,
            args.dlayers,
            args.dunits,
            args.dec_embed_dim,
            dropout_rate=args.dropout_rate_decoder,
            dropout_rate_embed=args.dropout_rate_embed_decoder,
            blank_id=blank_id,
        )
        decoder_out = args.dunits

        self.transducer_tasks = TransducerTasks(
            encoder_out,
            decoder_out,
            args.joint_dim,
            odim,
            joint_activation_type=args.joint_activation_type,
            transducer_loss_weight=args.transducer_weight,
            ctc_loss=args.use_ctc_loss,
            ctc_loss_weight=args.ctc_loss_weight,
            ctc_loss_dropout_rate=args.ctc_loss_dropout_rate,
            lm_loss=args.use_lm_loss,
            lm_loss_weight=args.lm_loss_weight,
            lm_loss_smoothing_rate=args.lm_loss_smoothing_rate,
            aux_transducer_loss=args.use_aux_transducer_loss,
            aux_transducer_loss_weight=args.aux_transducer_loss_weight,
            aux_transducer_loss_mlp_dim=args.aux_transducer_loss_mlp_dim,
            aux_trans_loss_mlp_dropout_rate=args.aux_transducer_loss_mlp_dropout_rate,
            symm_kl_div_loss=args.use_symm_kl_div_loss,
            symm_kl_div_loss_weight=args.symm_kl_div_loss_weight,
            fastemit_lambda=args.fastemit_lambda,
            blank_id=blank_id,
            ignore_id=ignore_id,
            training=training,
        )

        if training and (args.report_cer or args.report_wer):
            self.error_calculator = ErrorCalculator(
                self.decoder if args.dtype == "custom" else self.dec,
                self.transducer_tasks.joint_network,
                self.transducer_tasks,
                self.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None

        self.etype = args.etype
        self.dtype = args.dtype

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim

        self.default_parameters(args)

        self.loss = None
        self.rnnlm = None

        if args.use_ctc_loss and args.ctc_loss_weight==1.0:
            self.ctc_pretrain = True
        else:
            self.ctc_pretrain = False

    def default_parameters(self, args: Namespace):
        """Initialize/reset parameters for Transducer.
        Args:
            args: Namespace containing model options.
        """
        initializer(self, args)

    def forward(
        self, feats: torch.Tensor, feats_len: torch.Tensor, labels: torch.Tensor, label_len: torch.Tensor, is_training: bool
    ) -> torch.Tensor:
        """E2E forward.
        Args:
            feats: Feature sequences. (B, F, D_feats)
            feats_len: Feature sequences lengths. (B,)
            labels: Label ID sequences. (B, L)
        Returns:
            loss: Transducer loss value
        """
        # 1. encoder
        feats = feats[:, : max(feats_len)]
        _enc_out, _enc_out_len, _ = self.enc(feats, feats_len)

        if self.use_auxiliary_enc_outputs:
            enc_out, aux_enc_out = _enc_out[0], _enc_out[1]
            enc_out_len, aux_enc_out_len = _enc_out_len[0], _enc_out_len[1]
        else:
            enc_out, aux_enc_out = _enc_out, None
            enc_out_len, aux_enc_out_len = _enc_out_len, None

        # 2. decoder
        if not self.ctc_pretrain:
            dec_in = get_decoder_input(labels, self.blank_id, self.ignore_id)
            self.dec.set_device(enc_out.device)
            dec_out = self.dec(dec_in)
        else:
            dec_out = None

        # 3. Transducer task and auxiliary tasks computation
        losses = self.transducer_tasks(
            enc_out,
            aux_enc_out,
            dec_out,
            labels,
            enc_out_len,
            aux_enc_out_len,
            self.ctc_pretrain,
        )

        if is_training or self.error_calculator is None:
        #if self.error_calculator is None:
            cer, wer = 0, 0
        else:
            cer, wer = self.error_calculator(
                enc_out, self.transducer_tasks.get_target(), self.ctc_pretrain
            )

        #if is_training or self.error_calculator is None:
        #if self.error_calculator is None:
        #    cer, wer = 0, 0
        #else:
        #    cer, wer = self.error_calculator(
        #        enc_out, self.transducer_tasks.get_target()
        #    )

        self.loss = sum(losses)
        loss_data = float(self.loss)

        if not math.isnan(loss_data):
            #self.reporter.report(
            #    loss_data,
            #    *[float(loss) for loss in losses],
            #    cer,
            #    wer,
            #)
            pass
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss, cer, wer

    def encode_rnn(self, feats: numpy.ndarray) -> torch.Tensor:
        """Encode acoustic features.
        Args:
            feats: Feature sequence. (F, D_feats)
        Returns:
            enc_out: Encoded feature sequence. (T, D_enc)
        """
        p = next(self.parameters())

        feats_len = [feats.shape[0]]

        feats = feats[:: self.subsample[0], :]
        feats = torch.as_tensor(feats, device=p.device, dtype=p.dtype)
        feats = feats.contiguous().unsqueeze(0)

        enc_out, _, _ = self.enc(feats, feats_len)

        return enc_out.squeeze(0)

    def recognize(
        self, feats: numpy.ndarray, beam_search: BeamSearchTransducer
    ) -> List:
        """Recognize input features.
        Args:
            feats: Feature sequence. (F, D_feats)
            beam_search: Beam search class.
        Returns:
            nbest_hyps: N-best decoding results.
        """
        self.eval()
        enc_out = self.encode_rnn(feats)
        nbest_hyps = beam_search(enc_out)

        return [asdict(n) for n in nbest_hyps]

