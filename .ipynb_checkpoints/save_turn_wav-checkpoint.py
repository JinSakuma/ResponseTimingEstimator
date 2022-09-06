import torch
import torchaudio

import wave
import numpy as np
import pandas as pd
import os
import glob
import json
from tqdm import tqdm


def split_wav(wavepath, outpath, start, end, rate=16000, bits_per_sample=16):
    
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
    
    torchaudio.save(filepath=outpath, src=torch.tensor([turn]), sample_rate=rate, encoding="PCM_S", bits_per_sample=bits_per_sample)
    

file_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR2022/dataframe/*.csv'))
file_names = [file_path.split('/')[-1].replace('.csv', '') for file_path in file_list]
root='/mnt/aoni04/jsakuma/data/ATR2022/turn/'
offset = 200

if __name__ == "__main__":
    for file_name in tqdm(file_names):
        df = pd.read_csv('/mnt/aoni04/jsakuma/data/ATR2022/dataframe/{}.csv'.format(file_name))    
        json_path='/mnt/aoni04/jsakuma/data/ATR2022/json/{}.json'.format(file_name)
        
        json_dict = {'name': file_name,
                     'speaker': [],
                     'wav_path': [],
                     'start': [],
                     'end': [],
         }

        with open(json_path, 'r') as jf:
            meta_json = json.load(jf)
            
        for i in range(len(df)):
            ch = df['ch'].iloc[i]
            if ch == 0:
                wavpath = meta_json['wav_path_user']
                spk = 'user'
            else:
                wavpath = meta_json['wav_path_agent']
                spk = 'agent'
            
            start, end = df[['start', 'end']].iloc[i]
            start = start - offset
            end = end + offset
            
            name = file_name+'_{:03}_ch{}.wav'.format(i, ch)
            outdir = os.path.join(root, 'wav', file_name)
            
            os.makedirs(outdir, exist_ok=True)
            
            outpath = os.path.join(outdir, name)
            split_wav(wavpath, outpath, start, end)
            
            json_dict['speaker'].append(spk)
            json_dict['wav_path'].append(outpath)
            json_dict['start'].append(start)
            json_dict['end'].append(end)
            
        out_json_path = os.path.join(root, 'json', file_name+'_turn.json')
        with open(out_json_path, 'w') as f:
            json.dump(json_dict, f, indent=4)