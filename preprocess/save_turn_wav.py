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
    
    return len(x)
    

file_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR-Fujie/dataframe/*.csv'))
file_names = [file_path.split('/')[-1].replace('.csv', '') for file_path in file_list]
root='/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/'
offset = 200
sys_duration = 3000

if __name__ == "__main__":
    for file_name in tqdm(file_names):
        df = pd.read_csv('/mnt/aoni04/jsakuma/data/ATR-Fujie/dataframe/{}.csv'.format(file_name))    
        
        json_dict = {'name': file_name,
                     'length': 0,
                     'speaker': [],
                     'wav_path': [],
                     'start': [],
                     'end': [],
                     'cur_usr_uttr_end': [],
                     'next_sys_uttr_start': [],
                     'next_sys_uttr_end': [],
                     'next_usr_uttr_start': [],
                     
         }
    
        for i in range(len(df)-1):
            ch = df['spk'].iloc[i]
            if ch == 0:
                wavpath = '/mnt/aoni04/jsakuma/data/ATR2022/wav_mono/{}_user.wav'.format(file_name)
                spk = 'user'
            else:
                wavpath = '/mnt/aoni04/jsakuma/data/ATR2022/wav_mono/{}_agent.wav'.format(file_name)
                spk = 'agent'
            
            start=df['start'].iloc[i] - offset
            cur_usr_uttr_end = df['end'].iloc[i]
            next_sys_uttr_start = df['next_sys_uttr_start'].iloc[i]
            next_sys_uttr_end = df['next_sys_uttr_end'].iloc[i]
            next_usr_uttr_start = df['next_usr_uttr_start'].iloc[i]
            end=min(next_sys_uttr_end, next_sys_uttr_start+sys_duration)
            
            if end == 100000:
                continue
            
            name = file_name+'_{:03}_ch{}.wav'.format(i, ch)
            outdir = os.path.join(root, 'wav', file_name)
            
            os.makedirs(outdir, exist_ok=True)
            
            outpath = os.path.join(outdir, name)
            length = split_wav(wavpath, outpath, start, end)
            
            json_dict['length'] = length
            json_dict['speaker'].append(spk)
            json_dict['wav_path'].append(outpath)
            json_dict['start'].append(int(start))
            json_dict['end'].append(int(end))
            json_dict['cur_usr_uttr_end'].append(int(cur_usr_uttr_end))
            json_dict['next_sys_uttr_start'].append(int(next_sys_uttr_start))
            json_dict['next_sys_uttr_end'].append(int(next_sys_uttr_end))
            json_dict['next_usr_uttr_start'].append(int(next_usr_uttr_start))
            
        out_json_path = os.path.join(root, 'json', file_name+'_samples.json')
        with open(out_json_path, 'w') as f:
            json.dump(json_dict, f, indent=4)