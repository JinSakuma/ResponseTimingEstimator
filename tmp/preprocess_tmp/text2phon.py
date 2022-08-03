import os
import json
import numpy as np
import pandas as pd

import MeCab
import jaconv
import re
import glob

from tqdm import tqdm

tagger = MeCab.Tagger("-Oyomi")
def str2yomi(text):
    yomi_kata = tagger.parse(text).split()[0]
    yomi_hira = jaconv.kata2hira(yomi_kata)
    
    return yomi_hira

phoneme_list = ['a', 'i', 'u', 'e', 'o', 'a:', 'i:', 'u:', 'e:', 'o:', 
                'b', 'by', 'k', 'ky', 'g', 'gy', 's', 'sh', 'z', 'zy', 'j',
                't', 'ts', 'ty', 'ch', 'd', 'dy', 'n', 'ny', 'hy', 'p', 'py', 'm', 'my',
                'y', 'r', 'ry', 'w', 'f', 'q', 'N', ':', '<eou>', '<pad>'
               ]

def phon2idx(phonemes):
    ids = []
    for phon in (phonemes+' <eou>').replace('::', ':').split():
        idx = phoneme_list.index(phon)
        ids.append(idx)
        
    return ids

DATAROOT = '/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/text'
file_ids = sorted(os.listdir(DATAROOT))

for file_id in tqdm(file_ids):
    path_list = glob.glob(os.path.join(DATAROOT, file_id+'/*.csv'))
    
    for text_path in path_list:
        
        text_dir = '/'.join(text_path.split('/')[:-1])
        name = text_path.split('/')[-1].replace('.csv', '')

        output_dir = text_dir.replace('text', 'phoneme')
        os.makedirs(output_dir, exist_ok=True)

        out_path =  os.path.join(output_dir, name+'.json')
        
#         if os.path.isfile(out_path):
#             continue
        
        try:
            df_text = pd.read_csv(text_path)
            df_text[pd.isna(df_text['asr_recog'])] = ''
            text_org = df_text['asr_recog'].tolist()[-1]

            text = re.sub(r"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", "", text_org)
            text_removed = re.sub(r"[a-zA-Z]", "", text)    
            code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
            text_clean = code_regex.sub('', text_removed)
            yomi = str2yomi(text_clean)
            phonemes = jaconv.hiragana2julius(jaconv.kata2hira(yomi))
            
            is_removed = False
            if len(text_org)!=len(text_removed):
                is_removed = True
                
            idx = phon2idx(phonemes)
        except:
            is_removed = True
            phonemes = ''
            idx = []
            
        json_dict = {'is_removed': is_removed, 'text': text_org, 'phoneme': phonemes, 'idx': idx}
        with open(out_path, 'w') as f:
            json.dump(json_dict, f, indent=4)
