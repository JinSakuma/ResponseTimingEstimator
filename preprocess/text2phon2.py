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

phoneme_list = [' ', 'a', 'i', 'u', 'e', 'o', 'a:', 'i:', 'u:', 'e:', 'o:', 
                'b', 'by', 'k', 'ky', 'g', 'gy', 's', 'sh', 'z', 'zy', 'j',
                't', 'ts', 'ty', 'ch', 'd', 'dy', 'n', 'ny', 'h', 'hy', 'p', 'py', 'm', 'my',
                'y', 'r', 'ry', 'w', 'f', 'q', 'N', ':', '<eou>', '<pad>'
               ]

def phon2idx(phonemes):
    ids = []
    for phon in (phonemes+' <eou>').replace('::', ':').split():
        idx = phoneme_list.index(phon)
        ids.append(idx)
        
    return ids

DATAROOT = '/mnt/aoni04/jsakuma/data/ATR2022/asr'
file_ids = sorted(os.listdir(DATAROOT))

for split in ['train', 'valid', 'test']:
    path = os.path.join(DATAROOT, 'data', split, 'text')
    with open(path) as f:
        lines = f.readlines()
    
    for i, line in enumerate(tqdm(lines)):
        try:
            name, text_org = line.split() 
        except:
            if len(line) > 30:
                print(i, 'aaa')
            else:
                print(i, 'bbb')
                print(line)
            continue
        
        spk, date, id1, id2, id3 = name.split('_')
        dir_name = date+'-'+id1+'_'+id2

        output_dir = os.path.join(DATAROOT, 'phoneme', dir_name)
        os.makedirs(output_dir, exist_ok=True)

        out_path =  os.path.join(output_dir, name+'.json')

#         if os.path.isfile(out_path):
#             continue

        try:            
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
