import os
import glob
import json
import wave
import struct
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torchlibrosa.augmentation import SpecAugmentation
import numpy as np
import pandas as pd

import MeCab
import jaconv
import re

from tqdm import tqdm

from espnet.transform.spec_augment import spec_augment


DATAROOT="/mnt/aoni04/jsakuma/data/ATR2022"
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

# 直前の発話のみ
# 出力: CNN-AE feature, VAD出力ラベル, 最後のIPU=1ラベル
class ATRDataset(Dataset):
    def __init__(self, config,
                 split='train',
                 speaker_list=None,                 
                 is_use_eou=False,
                ):
        
        self.config = config
        self.split = split
        
#         name_path = "/mnt/aoni04/jsakuma/data/ATR2022/asr/names/{}.txt".format(split)
#         with open(name_path) as f:
#             lines = f.readlines()
    
#         self.file_names = [line.replace('\n', '') for line in lines]
#         if speaker_list is not None:
#             self.file_names = [name for name in self.file_names if spk_dict[name+'.wav'] in speaker_list]

        df_path = "/mnt/aoni04/jsakuma/data/ATR2022/asr/dataframe/{}.csv".format(split)
        self.df = pd.read_csv(df_path)
        self.spk_list = speaker_list
        
#         if is_use_specaug:
#             self.spec_augmenter = SpecAugmentation(
#                 time_drop_width=32,
#                 time_stripes_num=2,
#                 freq_drop_width=32,
#                 freq_stripes_num=2,
#             )
            
        self.is_use_specaug = config.is_use_specaug
        self.is_use_eou = is_use_eou              
        
        self.offset = 200  # VADのhang over
        self.frame_length = 50  # 1frame=50ms
        self.sample_rate = 16000
        self.max_positive_length = 20 # システム発話のターゲットの最大長(0/1の1の最大長) [frame]
        self.N = 10  # 現時刻含めたNフレーム先の発話状態を予測
        
        self.asr_delay1 = 200 # 実際のASRの遅延 [ms]
        self.asr_delay2 = 200 # 実験用の仮のASRの遅延 [ms]
        
        self.data = self.get_data()
        
    def read_wav(self, wavpath):
        wf = wave.open(wavpath, 'r')

        # waveファイルが持つ性質を取得
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        fr = wf.getframerate()
        fn = wf.getnframes()

        x = wf.readframes(wf.getnframes()) #frameの読み込み
        x = np.frombuffer(x, dtype= "int16") #numpy.arrayに変換

        return x

    def get_last_ipu(self, turn):
        ipu_label = np.zeros(len(turn))
        sub = turn[1:]-turn[:-1]    
        if 1 in sub:
            idx = np.where(sub==1)[0][-1]
            ipu_label[idx+1:] = 1

        return ipu_label
    
    def get_data(self):
        
        df = self.df
        
        # NaNの削除
        df = df[df['phoneme']==df['phoneme']]
        df = df[df['text']==df['text']]
        
        # 英数字・記号の含まれるものの削除
        df = df[df['is_removed']==False]
        
        # speakerの指定
        if self.spk_list is not None:
            for spk in self.spk_list:
                df = df[df['spk']==spk]              
            
        return df
        
    def __getitem__(self, index):
        batch = {}
        batch['text'] = self.data['text'].iloc[index]
        batch['kana'] = self.data['kana'].iloc[index]
        idx_str = self.data['kana_idx'].iloc[index]
        idx = [int(i) for i in idx_str.replace('[', '').replace(']', '').replace(',', '').split()]
#         if not self.is_use_eou:
#             idx = idx[:-1] 
        
        batch['idx'] = idx
        feat_path = self.data['feat_path'].iloc[index]
        
        feat = np.load(feat_path)
        
        if self.is_use_specaug and self.split == 'train':
            feat = spec_augment(feat)            
#             feat = torch.tensor(feat).unsqueeze(0).unsqueeze(0)
#             feat = self.spec_augmenter(feat)
#             feat = feat.squeeze(0).squeeze(0).numpy()
        
        batch['feat'] = feat
        batch['indices'] = index
        
        return list(batch.values())

    def __len__(self):
        # raise NotImplementedError
        return len(self.data)
    

def collate_fn(batch, pad_idx=127):
    texts, kanas, idxs, feats, indices = zip(*batch)
    
    batch_size = len(indices)
    
    max_len = max([len(f) for f in feats])
    max_id_len = max([len(i) for i in idxs])
    
    feat_dim = feats[0].shape[-1]     
    feat_ = torch.zeros(batch_size, max_len, feat_dim)
    
    idxs_ = []
    input_lengths = []
    target_lengths = []
    for i in range(batch_size):        
        
        l1 = len(feats[i])
        l2 = len(idxs[i])
                
        input_lengths.append(l1) 
        feat_[i, :l1] = torch.tensor(feats[i])        
        
        target_lengths.append(l2) 
        idxs_.append(idxs[i]+[pad_idx]*(max_id_len-l2))

    idxs_ = torch.tensor(idxs_).long()
    input_lengths = torch.tensor(input_lengths).long()
    target_lengths = torch.tensor(target_lengths).long()
        
    return feat_, input_lengths, idxs_, target_lengths, indices, texts, kanas
    
    
def create_dataloader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn= lambda x: collate_fn(x),
    )
    return loader

def get_dataset(config, split="train", speaker_list=None):
    dataset = ATRDataset(config, split, speaker_list)
    return dataset


def get_dataloader(dataset, config, split="train"):
    if split=="train":
        shuffle = True
    else:
        shuffle = False
    dataloader = create_dataloader(dataset, config.optim_params.batch_size, shuffle=shuffle)
    return dataloader




