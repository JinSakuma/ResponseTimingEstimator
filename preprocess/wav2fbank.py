import os
import glob
import json
import torch
from torch.utils.data import DataLoader
import pickle
import string
import random
import librosa
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm


### fbank config ###
sr = 16000
n_mels = 80
n_fft = 512
win_length = int(sr*0.025)
hop_length = int(sr*0.01)
#####################


DATAROOT = '/mnt/aoni04/jsakuma/data/ATR2022/asr/wav'
file_ids = sorted(os.listdir(DATAROOT))

for file_id in tqdm(file_ids):
    
    wav_path_list = glob.glob(os.path.join(DATAROOT, file_id+'/*.wav'))
    for wavpath in wav_path_list:
        
        wav_dir = '/'.join(wavpath.split('/')[:-1])
        name = '_'.join(wavpath.split('/')[-1].split('.')[:-1])
        
        output_dir = wav_dir.replace('wav', 'fbank')
        os.makedirs(output_dir, exist_ok=True)
        
        fbank_path =  os.path.join(output_dir, name+'_fbank.npy')
        
        if os.path.isfile(fbank_path):
            continue

        wav, sr = librosa.core.load(wavpath, sr=sr, mono=True)
        # wav = wav.numpy()[0]

        wav_mel = librosa.feature.melspectrogram(
            y=wav, 
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft, 
            win_length=win_length, 
            hop_length=hop_length,
        )
        wav_logmel = np.log(wav_mel + 1e-7)
        wav_logmel = librosa.util.normalize(wav_logmel).T                
        
        #特徴量保存
        np.save(fbank_path, wav_logmel)
