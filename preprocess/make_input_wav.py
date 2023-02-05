import torch
import torchaudio
import codecs
import wave
import numpy as np
import pandas as pd
import os
import glob
import json
from tqdm import tqdm



DATAROOT="/mnt/aoni04/jsakuma/data/ATR-Trek"
VADDIR  = os.path.join(DATAROOT, "vad")
OFFSET = 300  # vad hangover - 100ms
DUR = 1000  # max length of target 1(=agent speaking)


def save_turn_wav(wavepath, outpath, start, end, rate=16000, bits_per_sample=16):
    
    wf = wave.open(wavepath, 'r')

    # waveファイルが持つ性質を取得
    ch = wf.getnchannels()
    width = wf.getsampwidth()
    fr = wf.getframerate()
    fn = wf.getnframes()
    
    x = wf.readframes(wf.getnframes()) #frameの読み込み
    x = np.frombuffer(x, dtype= "int16") #numpy.arrayに変換
    
    turn = x[start*(rate//1000):end*(rate//1000)]

    wf.close()
    
    #torchaudio.save(filepath=outpath, src=torch.tensor([turn]), sample_rate=rate, encoding="PCM_S", bits_per_sample=bits_per_sample)
    
    waveFile = wave.open(outpath, 'wb')
    waveFile.setnchannels(ch)
    waveFile.setsampwidth(width)
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(turn))
    waveFile.close()

if __name__ == '__main__':
    file_paths = sorted(glob.glob(os.path.join(DATAROOT, 'turn_csv', '*.csv')))
    for file_path in tqdm(file_paths[:1]):
        df = pd.read_csv(file_path)
        file_name = file_path.split('/')[-1].replace('.csv', '')
        
        for i in range(len(df)):
            start, end, nstart, nend = df[['start', 'end', 'next_start', 'next_end']].iloc[i]
            fend = max(end, min(nstart+DUR, nend))
            
            wavpath = '/mnt/aoni04/jsakuma/data/ATR2022/wav_mono/{}_user.wav'.format(file_name)
            wav_out_dir = os.path.join(DATAROOT, 'input_wav2', file_name)
            os.makedirs(wav_out_dir, exist_ok=True)
            
            name = file_name+'_{:03}.wav'.format(i+1)
            wav_out_path = os.path.join(wav_out_dir, name)                
            save_turn_wav(wavpath, wav_out_path, max(0, start-OFFSET), fend)