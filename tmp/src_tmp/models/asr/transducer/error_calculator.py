"""CER/WER computation for Transducer model."""

from typing import List
from typing import Tuple
from typing import Union

import torch

from src.models.asr.transducer.beam_search_transducer import BeamSearchTransducer
from src.models.asr.transducer.joint_network import JointNetwork
from src.models.asr.transducer.rnn_decoder import RNNDecoder
from src.models.asr.transducer.transducer_tasks import TransducerTasks
import time


class ErrorCalculator(object):
    """CER and WER computation for Transducer model.
    Args:
        decoder: Decoder module.
        joint_network: Joint network module.
        token_list: Set of unique labels.
        sym_space: Space symbol.
        sym_blank: Blank symbol.
        report_cer: Whether to compute CER.
        report_wer: Whether to compute WER.
    """

    def __init__(
        self,
        decoder: RNNDecoder,
        joint_network: JointNetwork,
        transducer_tasks: TransducerTasks,
        token_list: List[int],
        sym_space: str,
        sym_blank: str,
        report_cer: bool = False,
        report_wer: bool = False,
    ):
        """Construct an ErrorCalculator object for Transducer model."""
        super().__init__()

        self.beam_search = BeamSearchTransducer(
            decoder=decoder,
            joint_network=joint_network,
            beam_size=1, # greedy search
            search_type="default",
        )

        self.decoder = decoder
        self.ctc_decoder = transducer_tasks

        self.token_list = token_list
        self.space = sym_space
        self.blank = sym_blank

        self.report_cer = report_cer
        self.report_wer = report_wer

    def __call__(
        self, enc_out: torch.Tensor, target: torch.Tensor, ctc_pretrain: bool,
    ) -> Tuple[float, float]:
        """Calculate sentence-level CER/WER score for hypotheses sequences.
        Args:
            enc_out: Encoder output sequences. (B, T, D_enc)
            target: Target label ID sequences. (B, L)
        Returns:
            cer: Sentence-level CER score.
            wer: Sentence-level WER score.
        """
        cer, wer = None, None

        batchsize = int(enc_out.size(0))
        batch_nbest = []

        enc_out = enc_out.to(next(self.decoder.parameters()).device)

        if not ctc_pretrain:
            #start = time.time()
            for b in range(batchsize):
                nbest_hyps = self.beam_search(enc_out[b])
                batch_nbest.append(nbest_hyps[-1])
            #print("beam_search speed: {}".format(time.time()-start))

            batch_nbest = [nbest_hyp.yseq[1:] for nbest_hyp in batch_nbest]
        else:
            ctc_lin = self.ctc_decoder.ctc_lin(
                torch.nn.functional.dropout(
                    enc_out.to(dtype=torch.float32), p=self.ctc_decoder.ctc_dropout_rate
                )
            )
            ctc_logp = torch.log_softmax(ctc_lin, dim=-1)
            decoded = torch.argmax(ctc_logp, dim=-1)
            batch_size = decoded.size(0)
            batch_nbest = []
            for i in range(batch_size):
                hypotheses_i = self.ctc_collapse(
                    decoded[i], 
                    len(decoded[i]),
                )
                batch_nbest.append(hypotheses_i)

        hyps, refs = self.convert_to_char(batch_nbest, target.cpu())

        if self.report_cer:
            cer = self.calculate_cer(hyps, enc)

        if self.report_wer:
            #start=time.time()
            wer = self.calculate_wer(hyps, refs)
            #print("wer speed: {}".format(time.time()-start))
        return cer, wer

    def ctc_collapse(self, seq, seq_len, blank_index=0):
        result = []
        for i, tok in enumerate(seq[:seq_len]):
            if tok.item() != blank_index:  # remove blanks
                if i != 0 and tok.item() == seq[i-1].item():  # remove dups
                    pass
                else:
                    result.append(tok.item())
        return result

    def convert_to_char(
        self, hyps: torch.Tensor, refs: torch.Tensor
    ) -> Tuple[List, List]:
        """Convert label ID sequences to character.
        Args:
            hyps: Hypotheses sequences. (B, L)
            refs: References sequences. (B, L)
        Returns:
            char_hyps: Character list of hypotheses.
            char_hyps: Character list of references.
        """
        char_hyps, char_refs = [], []

        for i, hyp in enumerate(hyps):
            hyp_i = [self.token_list[int(h)] for h in hyp]
            ref_i = [self.token_list[int(r)] for r in refs[i]]

            char_hyp = " ".join(hyp_i).replace(self.space, "")
            char_hyp = char_hyp.replace("[CLS]", "")
            char_hyp = char_hyp.replace("[SEP]", "")
            char_ref = " ".join(ref_i).replace(self.space, "")
            char_ref = char_ref.replace("[CLS]", "")
            char_ref = char_ref.replace("[SEP]", "")

            char_hyps.append(char_hyp)
            char_refs.append(char_ref)

            #print(char_hyps)
            #print(char_refs)
        return char_hyps, char_refs

    def calculate_cer(self, hyps: torch.Tensor, refs: torch.Tensor) -> float:
        """Calculate sentence-level CER score.
        Args:
            hyps: Hypotheses sequences. (B, L)
            refs: References sequences. (B, L)
        Returns:
            : Average sentence-level CER score.
        """
        import editdistance

        distances, lens = [], []

        for i, hyp in enumerate(hyps):
            char_hyp = hyp.replace(" ", "")
            char_ref = refs[i].replace(" ", "")

            distances.append(editdistance.eval(char_hyp, char_ref))
            lens.append(len(char_ref))

        return float(sum(distances)) / sum(lens)

    def calculate_wer(self, hyps: torch.Tensor, refs: torch.Tensor) -> float:
        """Calculate sentence-level WER score.
        Args:
            hyps: Hypotheses sequences. (B, L)
            refs: References sequences. (B, L)
        Returns:
            : Average sentence-level WER score.
        """
        import editdistance

        distances, lens = [], []

        for i, hyp in enumerate(hyps):
            word_hyp = hyp.split()
            word_ref = refs[i].split()

            distances.append(editdistance.eval(word_hyp, word_ref))
            lens.append(len(word_ref))

        return float(sum(distances)) / sum(lens)


    def edit_distance(src_seq, tgt_seq):
        src_len, tgt_len = len(src_seq), len(tgt_seq)
        if src_len == 0: return tgt_len
        if tgt_len == 0: return src_len

        dist = np.zeros((src_len+1, tgt_len+1))
        for i in range(1, tgt_len+1):
            dist[0, i] = dist[0, i-1] + 1
        for i in range(1, src_len+1):
            dist[i, 0] = dist[i-1, 0] + 1
        for i in range(1, src_len+1):
            for j in range(1, tgt_len+1):
                cost = 0 if src_seq[i-1] == tgt_seq[j-1] else 1
                dist[i, j] = min(
                    dist[i,j-1]+1,
                    dist[i-1,j]+1,
                    dist[i-1,j-1]+cost,
                )
        return dist


    def get_cer(hypotheses, hypothesis_lengths, references, reference_lengths):
        assert len(hypotheses) == len(references)
        cer = []
        for i in range(len(hypotheses)):
            if len(hypotheses[i]) > 0:
                dist_i = edit_distance(
                    hypotheses[i][:hypothesis_lengths[i]],
                    references[i][:reference_lengths[i]],
                )
                # CER divides the edit distance by the length of the true sequence
                cer.append((dist_i[-1, -1] / float(reference_lengths[i])))
            else:
                cer.append(1)  # since we predicted empty 
        return np.mean(cer)
