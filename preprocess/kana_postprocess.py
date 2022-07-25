import os
import json
import numpy as np
import pandas as pd

import MeCab
import jaconv
import re
import glob

from tqdm import tqdm


path = '/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr_kana/asr1/data/jp_token_list/word/tokens.txt'
with open(path) as f:
    lines =f.readlines()
    
tokens = [line.split()[0] for line in lines]

def kana2idx(kana):    
    idxs = [tokens.index(k) for k in kana]
    
    return idxs


file_names = os.listdir("/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/kana2")
for file_name in tqdm(file_names):
    names = os.listdir("/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/kana2/{}".format(file_name))
    for name in names:
        df_path = "/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/kana2/{}/{}".format(file_name, name)
        df = pd.read_csv(df_path)        

        info = {'asr_recog': [], 'idx': []}
        for i in range(len(df)):
            recog = df['asr_recog'].iloc[i]
            if recog != recog or recog == '':
                info['asr_recog'].append('')
                info['idx'].append(0)
            else:
                info['asr_recog'].append(recog.replace(' ', ''))               
                try:                    
                    info['idx'].append(kana2idx(recog.split()))
                except:
                    print(df_path)
                    print(recog)
                    print(pre)
                    print(aaaa)


                pre = recog

        df_new = pd.DataFrame(info)       
        out_path = "/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/kana_postprocess2/{}/{}".format(file_name, name)
        df_new.to_csv(out_path, encoding='utf-8', index=False)