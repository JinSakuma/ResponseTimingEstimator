import torch
import torchaudio

import wave
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

out_dir='/mnt/aoni04/jsakuma/data/ATR2022/wav_mono/'

def split_wav(wavepath, name, rate=16000, bits_per_sample=16):
    
    wf = wave.open(wavepath, 'r')

    # waveファイルが持つ性質を取得
    ch = wf.getnchannels()
    width = wf.getsampwidth()
    fr = wf.getframerate()
    fn = wf.getnframes()
    
    x = wf.readframes(wf.getnframes()) #frameの読み込み
    x = np.frombuffer(x, dtype= "int16") #numpy.arrayに変換
    
    user = x[::2]
    agent= x[1::2]

    wf.close()
    
    user_wav_path = os.path.join(out_dir, name+'_user.wav')
    agent_wav_path = os.path.join(out_dir, name+'_agent.wav')
    
    torchaudio.save(filepath=user_wav_path, src=torch.tensor([user]), sample_rate=rate, encoding="PCM_S", bits_per_sample=bits_per_sample)
    torchaudio.save(filepath=agent_wav_path, src=torch.tensor([agent]), sample_rate=rate, encoding="PCM_S", bits_per_sample=bits_per_sample)
    

turn_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/turn_info/*_sorted.turn.txt'))
root='/mnt/aoni04/jsakuma/data/ATR/wav_safia/'

for i in tqdm(range(len(turn_list))):
    name = '_'.join(turn_list[i].split('/')[-1].split('_')[:-1])
    wavpath = os.path.join(root, name+'.wav')
    split_wav(wavpath, name)