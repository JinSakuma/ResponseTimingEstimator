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


DATAROOT="/mnt/aoni04/jsakuma/data/ATR-Fujie"

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


path = '/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr/asr1/data/jp_token_list/char/tokens.txt'
with open(path) as f:
    lines =f.readlines()
    
tokens = [line.split()[0] for line in lines]

def token2idx(token): 
    if token != token or token == '':
        return [0]
    
    token = token.replace('<eou>', '')
    idxs = [tokens.index(t) for t in token]#+[len(tokens)-1]
    
    return idxs

def idx2token(idxs): 
    token = [tokens[idx] for idx in idxs]
    
    return token


# 直前の発話のみ
# 出力: CNN-AE feature, VAD出力ラベル, 最後のIPU=1ラベル
class ATRDataset(Dataset):
    def __init__(self, config, split='train', speaker_list=None, is_use_eou=True):
        self.config = config
        
        name_path = "/mnt/aoni04/jsakuma/data/ATR2022/asr/names/{}.txt".format(split)
        with open(name_path) as f:
            lines = f.readlines()
    
        self.file_names = [line.replace('\n', '') for line in lines]
        if speaker_list is not None:
            self.file_names = [name for name in self.file_names if spk_dict[name+'.wav'] in speaker_list]
        
        self.offset = 200  # VADのhang over
        self.frame_length = 50  # 1frame=50ms
        self.sample_rate = 16000
        self.max_positive_length = 20 # システム発話のターゲットの最大長(0/1の1の最大長) [frame]
        self.N = 10  # 現時刻含めたNフレーム先の発話状態を予測
        self.context_num = 3
        
        self.asr_delay1 = 200 # 実際のASRの遅延 [ms]
        self.asr_delay2 = 200 # 実験用の仮のASRの遅延 [ms]
        
        self.is_use_eou = is_use_eou    
        self.eou_id = len(tokens)-1
        
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
        df_turns_path = os.path.join(DATAROOT, 'dataframe3/{}.csv'.format(file_name))
        #df_turn_text_path = os.path.join(DATAROOT, 'dataframe0526/{}.csv'.format(file_name))
#         df_vad_path = os.path.join(DATAROOT,'vad/{}.csv'.format(file_name))
        json_turn_path = os.path.join(DATAROOT, 'samples/json/{}_samples.json'.format(file_name))
        feat_list = os.path.join(DATAROOT, 'samples/CNN_AE/{}/*_spec.npy'.format(file_name))
        wav_list = os.path.join(DATAROOT, 'samples/wav/{}/*.wav'.format(file_name))
        feat_list = sorted(glob.glob(feat_list))
        wav_list = sorted(glob.glob(wav_list))
        
        df = pd.read_csv(df_turns_path)
#         df_turn_text = pd.read_csv(df_turn_text_path)
#         df_vad = pd.read_csv(df_vad_path)

        with open(json_turn_path, 'r') as jf:
            turn_info = json.load(jf)

        # 対話全体の長さ確認
        N = turn_info['length']//self.sample_rate*1000

#         # vadの結果
#         uttr_user = np.zeros(N//self.frame_length)
#         uttr_agent = np.zeros(N//self.frame_length)      
#         for i in range(len(df_vad)):
#             spk = df_vad['spk'].iloc[i]
#             start = (df_vad['start'].iloc[i]) // self.frame_length
#             end = (df_vad['end'].iloc[i]) // self.frame_length

#             if spk==0:
#                 uttr_user[start:end]=1
#             else:
#                 uttr_agent[start:end]=1

        batch_list = []
        num_turn = len(turn_info['speaker'])
        for t in range(num_turn): 
#             feat_path = feat_list[t]
            wav_path = wav_list[t]
#             feat_name_part = feat_path.split('/')[-1].split('_')[:-1]
#             feat_file_name = '_'.join(feat_name_part[:-1])
#             wav_name_part = wav_path.split('/')[-1].split('_')[:-1]
#             wav_file_name = '_'.join(wav_name_part)
            
#             assert feat_file_name == wav_file_name, "file name mismatch! check the feat-file and wav-file!"
            
            ch = df['spk'].iloc[t]
            offset = df['offset'].iloc[t]
            next_ch = df['next_spk'].iloc[t]
#             # text = df['text'].iloc[t]
            start=turn_info['start'][t]//self.frame_length
            end = turn_info['end'][t]//self.frame_length
            cur_usr_uttr_end = turn_info['cur_usr_uttr_end'][t]//self.frame_length
            timing = turn_info['next_sys_uttr_start'][t]//self.frame_length
            
            timing = timing-start
            
            if offset > 3000 or offset < -500:
                continue
                
#             if df['next_sys_uttr_start'].iloc[t]-df['offset'].iloc[t] == df['end'].iloc[t]:
#                 is_barge_in = False
#             else:
#                 is_barge_in = True

#             if end - timing > self.max_positive_length:  # システム発話をどれくらいとるか
#                 end = timing + self.max_positive_length

#             vad_user = uttr_user[start:end]
#             vad_agent = uttr_agent[start:end]

#             turn_label = np.zeros(N//self.frame_length)
#             turn_label[start:cur_usr_uttr_end] = 1
#             turn_label = turn_label[start:end]

#             timing_target = np.zeros(N//self.frame_length)
#             timing_target[timing:] = 1

#             turn_timing_target = timing_target[start:end]

#             last_ipu_user = self.get_last_ipu(vad_user)
#             last_ipu_agent = self.get_last_ipu(vad_agent)

    #             turn_user = np.ones(len(turn_user)) - turn_user
    #             turn_agent = np.ones(len(turn_agent)) - turn_agent

            if ch == 0 and next_ch != 0:
                pass
#                 vad_label = vad_user
#                 last_ipu = last_ipu_user
            else:
                continue
                
            # text
#             contexts = df_turn_text['text'].iloc[max(0, t-self.context_num):t].tolist()
#             spks = df_turn_text['spk'].iloc[max(0, t-self.context_num):t].tolist()
#             context = ''
#             for s in range(len(spks)):
#                 context += '<spk{}>'.format(spks[s]+1)+str(contexts[s])
            
            text_path = ('_'.join(wav_path.split('_')[:-1])+'.csv').replace('wav', 'text_10_4_2')
            df_text = pd.read_csv(text_path)
            df_text[pd.isna(df_text['asr_recog'])] = ''
            text = df_text['asr_recog'].tolist()[-1]            
            
#             kana_path = '_'.join(wav_path.replace('wav', 'kana2').split('_')[:-1])+'.csv'
#             df_kana = pd.read_csv(kana_path)
#             df_kana[pd.isna(df_kana['asr_recog'])] = ''
#             kana = df_kana['asr_recog'].tolist()[-1]
            kana = None
            
            idx = token2idx(text)
            
            if text == '':
                continue
            
            if self.is_use_eou:
                idx = idx+[self.eou_id]
                text += '<eou>'            
            
            # Delayの調節
            # tmp = text[-1]
            # n = self.asr_delay1//self.frame_length-self.asr_delay2//self.frame_length
            # text = text[n:]+[tmp]*n
                
#                 vad_label = vad_agent
#                 last_ipu = last_ipu_agent
                
#             pad = self.offset//self.frame_length
#             turn[:pad] = 0
#             #turn[-pad:] = 0
            
#             length = len(turn)
#             target = np.zeros([length-pad, self.N])

#             for i in range(length-pad):
#                 n = len(turn[i:i+self.N])
#                 tmp = np.zeros(self.N)
#                 tmp[:n] = turn[i:i+self.N]
#                 target[i] = tmp

            batch = {
                     "text": text,
                     "kana": kana,
                     "idx": idx,
                     "wav_path": wav_path
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
        batch['indices'] = index        
        
        return list(batch.values())

    def __len__(self):
        # raise NotImplementedError
        return len(self.data)
    

def collate_fn(batch, pad_idx=0):
    texts, kanas, idxs, paths, indices = zip(*batch)
    
    batch_size = len(indices)
    max_id_len = max([len(i) for i in idxs])   
    
    idxs_ = []
    target_lengths = []
    for i in range(batch_size):        
        
        l = len(idxs[i])        
        target_lengths.append(l-1)
        idxs_.append(idxs[i]+[pad_idx]*(max_id_len-l))

    idxs_ = torch.tensor(idxs_).long()
    target_lengths = torch.tensor(target_lengths).long()
        
    return texts, kanas, idxs_, target_lengths, indices
    
    
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




