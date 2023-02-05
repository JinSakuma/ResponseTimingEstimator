import os
import glob
import json
import wave
import struct
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from tqdm import tqdm


name_mapping = {'F1(伊藤)': 'F1',
                'F2(不明)': 'F2',
                'F3(中川)': 'F3',
                'F4(川村)': 'F4',
                'M1(黒河内)': 'M1',
                'M2(平林)': 'M2',
                'M3(浜田)': 'M3',
                'M4(不明)': 'M4'
               }

# 直前の発話のみ
# 出力: CNN-AE feature, VAD出力ラベル, 最後のIPU=1ラベル
class ATRDataset(Dataset):
    def __init__(self, config, split='train', speaker_list=None):
        self.config = config
        self.data_dir = self.config.data_params.data_dir
        
        name_path = os.path.join(self.data_dir, "asr/names/{}.txt".format(split))
        with open(name_path) as f:
            lines = f.readlines()
    
        self.file_names = [line.replace('\n', '') for line in lines]
        
        spk_file_path = os.path.join(self.data_dir, 'speaker_ids.csv')
        df_spk=pd.read_csv(spk_file_path, encoding="shift-jis")
        df_spk['operator'] =  df_spk['オペレータ'].map(lambda x: name_mapping[x])
        filenames = df_spk['ファイル名'].to_list()
        spk_ids = df_spk['operator'].to_list()
        spk_dict  =dict(zip(filenames, spk_ids))
        if speaker_list is not None:
            self.file_names = [name for name in self.file_names if spk_dict[name+'.wav'] in speaker_list]
        
        self.frame_length = config.data_params.frame_size  # 1frame=50ms
        self.sample_rate = config.data_params.sampling_rate
        self.max_positive_length = config.data_params.max_positive_length # システム発話のターゲットの最大長(0/1の1の最大長) [frame]
        self.text_dir = config.data_params.text_dir
        
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
    
    def get_turn_info(self, file_name):
        # 各種ファイルの読み込み
        df_turns_path = os.path.join(self.data_dir, 'dataframe/{}.csv'.format(file_name))
        df_vad_path = os.path.join(self.data_dir,'vad/{}.csv'.format(file_name))
        json_turn_path = os.path.join(self.data_dir, 'samples/json/{}_samples.json'.format(file_name))
        feat_list = os.path.join(self.data_dir, 'samples/CNN_AE/{}/*_spec.npy'.format(file_name))
        wav_list = os.path.join(self.data_dir, 'samples/wav/{}/*.wav'.format(file_name))
        feat_list = sorted(glob.glob(feat_list))
        wav_list = sorted(glob.glob(wav_list))
        
        df = pd.read_csv(df_turns_path)
        df_vad = pd.read_csv(df_vad_path)

        with open(json_turn_path, 'r') as jf:
            turn_info = json.load(jf)

        # 対話全体の長さ確認
        N = turn_info['length']//self.sample_rate*1000

        # vadの結果
        uttr_user = np.zeros(N//self.frame_length)
        uttr_agent = np.zeros(N//self.frame_length)      
        for i in range(len(df_vad)):
            spk = df_vad['spk'].iloc[i]
            start = (df_vad['start'].iloc[i]) // self.frame_length
            end = (df_vad['end'].iloc[i]) // self.frame_length

            if spk==0:
                uttr_user[start:end]=1
            else:
                uttr_agent[start:end]=1

        batch_list = []
        num_turn = len(turn_info['speaker'])
        for t in range(num_turn): 
            feat_path = feat_list[t]
            wav_path = wav_list[t]
            feat_name_part = feat_path.split('/')[-1].split('_')[:-1]
            feat_file_name = '_'.join(feat_name_part[:-1])
            wav_name_part = wav_path.split('/')[-1].split('_')[:-1]
            wav_file_name = '_'.join(wav_name_part)           
            
            assert feat_file_name == wav_file_name, "file name mismatch! check the feat-file and wav-file!"
            
            ch = df['spk'].iloc[t]
            offset = df['offset'].iloc[t]
            next_ch = df['next_spk'].iloc[t]
            start=turn_info['start'][t]//self.frame_length
            end = turn_info['end'][t]//self.frame_length
            cur_usr_uttr_end = turn_info['cur_usr_uttr_end'][t]//self.frame_length
            timing = turn_info['next_sys_uttr_start'][t]//self.frame_length

            if end - timing > self.max_positive_length:  # システム発話をどれくらいとるか
                end = timing + self.max_positive_length

            vad_user = uttr_user[start:end]
            vad_agent = uttr_agent[start:end]

            turn_label = np.zeros(N//self.frame_length)
            turn_label[start:cur_usr_uttr_end] = 1
            turn_label = turn_label[start:end]

            timing_target = np.zeros(N//self.frame_length)
            timing_target[timing:] = 1

            turn_timing_target = timing_target[start:end]

            last_ipu_user = self.get_last_ipu(vad_user)
            last_ipu_agent = self.get_last_ipu(vad_agent)

            if ch == 0 and next_ch != 0:
                vad_label = vad_user
                last_ipu = last_ipu_user
            else:
                continue                

            batch = {"ch": ch,
                     "offset": offset,
                     "feat_path": feat_path,
                     "wav_path": wav_path,
                     "vad": vad_label,
                     "turn": turn_label,
                     "last_ipu": last_ipu,
                     "target": vad_label,
                    }

            batch_list.append(batch)
            
        return batch_list
    
    def get_data(self):
        data = []
        for file_name in tqdm(self.file_names):  
            data += self.get_turn_info(file_name)
            
        return data            
        
    def __getitem__(self, index):
        batch = self.data[index]
        feat = np.load(batch['feat_path'])
        #wav = self.read_wav(batch['wav_path'])
        vad = batch['vad']
        turn = batch['turn']
        last_ipu = batch['last_ipu']
        target = batch['target']
        
        length = min(len(feat), len(vad), len(turn), len(target))
        batch['vad'] = vad[:length]
        batch['turn'] = turn[:length]
        batch['last_ipu'] = last_ipu[:length]
        batch['target'] = target[:length]
        batch['feat'] = feat[:length]
        batch['indices'] = index
        
        wav_len = int(length * self.sample_rate * self.frame_length / 1000)
        
        assert len(batch['feat'])==len(batch['vad']), "error"
        
        return list(batch.values())

    def __len__(self):
        return len(self.data)
    

def collate_fn(batch):
    chs, offsets, feat_paths, wav_paths, vad, turn, last_ipu, targets, feats, indices = zip(*batch)
    
    batch_size = len(chs)
    
    max_len = max([len(f) for f in feats])
    feat_dim = feats[0].shape[-1]
    
    vad_ = torch.zeros(batch_size, max_len).long()
    turn_ = torch.zeros(batch_size, max_len).float()
    last_ipu_ = torch.zeros(batch_size, max_len).long()
    target_ = torch.ones(batch_size, max_len).long()*(-100)
    feat_ = torch.zeros(batch_size, max_len, feat_dim)
    
    input_lengths = []
    for i in range(batch_size):
        l1 = len(feats[i])
        input_lengths.append(l1)
            
        vad_[i, :l1] = torch.tensor(vad[i]).long()               
        turn_[i, :l1] = torch.tensor(turn[i]).float()       
        last_ipu_[i, :l1] = torch.tensor(last_ipu[i]).long()
        target_[i, :l1] = torch.tensor(targets[i]).long()       
        feat_[i, :l1] = torch.tensor(feats[i])
        
    input_lengths = torch.tensor(input_lengths).long()
        
    return chs, vad_, turn_, last_ipu_, target_, feat_, input_lengths, offsets, indices, #wav_, wav_lengths, wav_paths
    
    
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




