import MeCab
import jaconv
import numpy as np
import os
import glob
from tqdm import tqdm

tagger = MeCab.Tagger("-Oyomi")


def remove_numbers(text):
    for i in range(10):
        text = text.replace('{}'.format(i), '')
        
    return text

def str2yomi(text):
    yomi_kata = tagger.parse(text).split()[0].replace('。', ' sp ').replace('、', ' sp ')
    yomi_hira = jaconv.kata2hira(yomi_kata)
    yomi_hira = remove_numbers(yomi_hira)
    
    return yomi_hira


ROOT = '/mnt/aoni04/jsakuma/data/ATR2022/asr_agent/wav_julius'
for split in ['train', 'valid', 'test']:
    path = '/mnt/aoni04/jsakuma/data/ATR2022/asr_agent/data/{}/text'.format(split)
    with open(path) as f:
        lines = f.readlines()

    for line in tqdm(lines):
        split_list = line.split()
            
        idx = split_list[0]
        text = ' sp '.join(split_list[1:])

        first, second, third = idx.split('_')[1:-1]
        name = first+'-'+second+'_'+third

        yomi = str2yomi(text)

        outdir = os.path.join(ROOT, name)
        out_path = os.path.join(outdir, idx+'.txt')
        with open(out_path, mode='w') as f:
            f.write(yomi)