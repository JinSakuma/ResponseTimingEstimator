import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import wave
import struct
import sys
from dotmap import DotMap
from tqdm import tqdm

sys.path.append('../')
import espnet
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from src.datasets.dataset_asr_inference import get_dataloader, get_dataset
#from src.datasets.dataset_reverse import get_dataloader, get_dataset
from src.utils.utils import load_config


import logging

logging.basicConfig(
    filename='beam.log',
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


speech2text = Speech2TextStreaming(
    asr_base_path="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr_kana/asr1",
    asr_train_config="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr_kana/asr1/exp/asr_train_asr_streaming_conformer_blk10_hop4_la2_raw_jp_word_sp/config.yaml",
    asr_model_file="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr_kana/asr1/exp/asr_train_asr_streaming_conformer_blk10_hop4_la2_raw_jp_word_sp/valid.acc.ave_10best.pth",
    lm_train_config="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr_kana/asr1/exp/lm_train_lm2_jp_word/config.yaml",
    lm_file="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr_kana/asr1/exp/lm_train_lm2_jp_word/valid.loss.best.pth",
    token_type=None,
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=2,
    ctc_weight=0.5,
    lm_weight=0.0,
    penalty=0.0,
    nbest=2,
    device = "cuda:1", # "cpu",
    disable_repetition_detection=True,
    decoder_text_length_limit=0,
    encoded_feat_length_limit=0
)


def read_wav(wavpath):
    wf = wave.open(wavpath, 'r')

    # waveファイルが持つ性質を取得
    ch = wf.getnchannels()
    width = wf.getsampwidth()
    fr = wf.getframerate()
    fn = wf.getnframes()

    x = wf.readframes(wf.getnframes()) #frameの読み込み
    x = np.frombuffer(x, dtype= "int16") #numpy.arrayに変換

    return x


def get_recognize_frame(data):
    output_list = []
    speech = data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    sim_chunk_length = 800 # 640

    if sim_chunk_length > 0:
        for i in range(len(speech)//sim_chunk_length):
            results = speech2text(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=False)
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                output_list.append(text)
            else:
                output_list.append("")

        results = speech2text(speech[(i+1)*sim_chunk_length:len(speech)], is_final=True)
    else:
        results = speech2text(speech, is_final=True)
    
    nbests = [text for text, token, token_int, hyp in results]
    if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                output_list.append(text)
    else:
        output_list.append("")
        
    return output_list
    

def run(args):
#     config = load_config(args.config)
#     if args.gpuid >= 0:
#         config.gpu_device = args.gpuid
        
#     if config.cuda:
#         device = torch.device('cuda:{}'.format(config.gpu_device))
#     else:
#         device = torch.device('cpu')
    
    # wavfile = '/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/wav/20131106-5_04/20131106-5_04_037_ch0.wav'
#     wavfile = "/mnt/aoni04/jsakuma/data/ATR2022/turn/wav/20131106-5_04/20131106-5_04_003_ch0.wav"
    wavfile = '/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/wav/20131106-5_04/20131106-5_04_037_ch0.wav'
    # wavfile = "/mnt/aoni04/jsakuma/data/ATR2022/turn/wav/20131101-1_01/20131101-1_01_001_ch0.wav"
    wav = read_wav(wavfile)
    text = get_recognize_frame(wav)
    print(text[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#     parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
#     parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    run(args)

