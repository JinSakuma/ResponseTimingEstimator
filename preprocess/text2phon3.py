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

phoneme_list = [' ', 'a', 'i', 'u', 'e', 'o', 'a:', 'i:', 'u:', 'e:', 'o:', 'N:',
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

spk_file_path = '/mnt/aoni04/jsakuma/data/ATR2022/speaker_ids.csv'
df_spk=pd.read_csv(spk_file_path, encoding="shift-jis")

name_mapping = {'F1(伊藤)': 'F1',
                'F2(不明)': 'F2',
                'F3(中川)': 'F3',
                'F4(川村)': 'F4',
                'M1(黒河内)': 'M1',
                'M2(平林)': 'M2',
                'M3(浜田)': 'M3',
                'M4(不明)': 'M4'
               }

df_spk['operator'] =  df_spk['オペレータ'].map(lambda x: name_mapping[x])

filenames = df_spk['ファイル名'].to_list()
spk_ids = df_spk['operator'].to_list()
spk_dict  =dict(zip(filenames, spk_ids))


for split in ['train', 'valid', 'test']:
    
    info = {'file_id': [], 'file_name': [], 'is_removed' :[], 'text': [], 'phoneme': [], 'idx': [], 'feat_path' : [], 'spk': []}
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

        output_dir = os.path.join(DATAROOT, 'fbank', dir_name)
        feat_path =  os.path.join(output_dir, name+'_fbank.npy')

        if not os.path.isfile(feat_path):
            print(feat_path)
            print(aaaaaa)
            continue

        try:        
            text_org = text_org.replace('―', 'ー').replace('－', 'ー').replace('—', 'ー')
            text = re.sub(r"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", "", text_org)
            text_removed = re.sub(r"[a-zA-Z]", "", text)    
            code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
            text_clean = code_regex.sub('', text_removed)
            yomi = str2yomi(text_clean)
            phonemes = jaconv.hiragana2julius(jaconv.kata2hira(yomi))
            phonemes = phonemes.replace('::', ':')

            is_removed = False
            if len(text_org)!=len(text_removed):
                is_removed = True

            idx = phon2idx(phonemes)
        except:
            is_removed = True
            phonemes = ''
            idx = []
                        
        spk = spk_dict[dir_name+'.wav']

        info['file_id'].append(dir_name)
        info['file_name'].append(name)
        info['is_removed'].append(is_removed)
        info['text'].append(text_org)
        info['phoneme'].append(phonemes)
        info['idx'].append(idx)
        info['feat_path'].append(feat_path)
        info['spk'].append(spk)

    df = pd.DataFrame(info)
    df_path = '/mnt/aoni04/jsakuma/data/ATR2022/asr/dataframe/{}.csv'.format(split)
    df.to_csv(df_path, encoding='utf-8', index=False)