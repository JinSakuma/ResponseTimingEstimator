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
import random

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
    def __init__(self, config, cv_id, split='train', subsets=['M1_all'], speaker_list=None, is_use_eou=True):    
        self.config = config
        self.data_dir = self.config.data_params.data_dir
        
        self.file_names = []
        for sub in subsets:
            name_path = os.path.join(self.data_dir, "asr/names/{}.txt".format(sub))
            with open(name_path) as f:
                lines = f.readlines()
                
            self.file_names += [line.replace('\n', '') for line in lines]

        spk_file_path = os.path.join(self.data_dir, 'speaker_ids.csv')
        df_spk=pd.read_csv(spk_file_path, encoding="shift-jis")
        df_spk['operator'] =  df_spk['オペレータ'].map(lambda x: name_mapping[x])
        filenames = df_spk['ファイル名'].to_list()
        spk_ids = df_spk['operator'].to_list()
        spk_dict  =dict(zip(filenames, spk_ids))
        if speaker_list is not None:
            self.file_names = [name for name in self.file_names if spk_dict[name+'.wav'] in speaker_list]
        
        path = self.config.data_params.token_list_path
        with open(path) as f:
            lines =f.readlines()
        self.tokens = [line.split()[0] for line in lines]
        
        self.frame_length = config.data_params.frame_size  # 1frame=50ms
        self.sample_rate = config.data_params.sampling_rate
        self.max_positive_length = config.data_params.max_positive_length # システム発話のターゲットの最大長(0/1の1の最大長) [frame]
        self.asr_delay = config.data_params.asr_decoder_delay # 実際のASRの遅延 [ms]        
        self.context_num = config.data_params.n_context
        self.max_timing = config.data_params.max_timing
        self.min_timing = config.data_params.min_timing
        self.text_dir = config.data_params.text_dir
        
        self.is_use_eou = is_use_eou    
        self.eou_id = len(self.tokens)-1
        
        data = self.get_data() 
        
        random.seed(0)
        random.shuffle(data)
        
        NUM = len(data)//5
        sub1 = data[NUM*0:NUM*1]
        sub2 = data[NUM*1:NUM*2]
        sub3 = data[NUM*2:NUM*3]
        sub4 = data[NUM*3:NUM*4]
        sub5 = data[NUM*4:]
        
        if cv_id == 1:
            trainset = sub1+sub2+sub3+sub4
            testset = sub5
        elif cv_id == 2:
            trainset = sub5+sub1+sub2+sub3
            testset = sub4
        elif cv_id == 3:
            trainset = sub4+sub5+sub1+sub2
            testset = sub3
        elif cv_id == 4:
            trainset = sub3+sub4+sub5+sub1
            testset = sub2
        elif cv_id == 5:
            trainset = sub2+sub3+sub4+sub5
            testset = sub1
        elif cv_id == 0:
            trainset = sub1+sub2+sub3+sub4+sub5
            testset = sub1+sub2+sub3+sub4+sub5
        else:
            NotImplemented
            
        if split == 'train':
            self.data = trainset
        else:
            self.data = testset
        
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
    
    def clean(self, sentence):
        for remove in ["はいはいはいはい", "はいはいはいは", "はいはいはい", "はいはいは", "はいはい", "はいは", "はい", "は"]:  
        #for remove in ["はいはいはいはい", "はいはいはいは", "はいはいはい", "はいはいは", "はいはい", "はいは"]:  
            if sentence != remove and sentence[-len(remove):]==remove:
                sentence = sentence[:-len(remove)]
                break

        return sentence
    
    def token2idx(self, token): 
        if token != token or token == '':
            return [0]

        token = token.replace('<eou>', '')
        idxs = [self.tokens.index(t) for t in token]#+[len(tokens)-1]

        return idxs

    def idx2token(self, idxs): 
        token = [self.tokens[idx] for idx in idxs]

        return token

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
        json_turn_path = os.path.join(self.data_dir, 'samples/json/{}_samples.json'.format(file_name))
        feat_list = os.path.join(self.data_dir, 'samples/CNN_AE/{}/*_spec.npy'.format(file_name))
        wav_list = os.path.join(self.data_dir, 'samples/wav/{}/*.wav'.format(file_name))
        feat_list = sorted(glob.glob(feat_list))
        wav_list = sorted(glob.glob(wav_list))
        
        df = pd.read_csv(df_turns_path)

        with open(json_turn_path, 'r') as jf:
            turn_info = json.load(jf)

        # 対話全体の長さ確認
        N = turn_info['length']//self.sample_rate*1000

        batch_list = []
        num_turn = len(turn_info['speaker'])
        for t in range(num_turn): 
            wav_path = wav_list[t]
            wav_name_part = wav_path.split('/')[-1].split('_')[:-1]
            wav_file_name = '_'.join(wav_name_part)
            
            ch = df['spk'].iloc[t]
            offset = df['offset'].iloc[t]
            next_ch = df['next_spk'].iloc[t]
            start=turn_info['start'][t]//self.frame_length
            end = turn_info['end'][t]//self.frame_length
            cur_usr_uttr_end = turn_info['cur_usr_uttr_end'][t]//self.frame_length
            timing = turn_info['next_sys_uttr_start'][t]//self.frame_length
            
            timing = timing-start
            
            if offset > self.max_timing or offset < self.min_timing:
                continue                

            if end - timing > self.max_positive_length:  # システム発話をどれくらいとるか
                end = timing + self.max_positive_length

            if ch == 0 and next_ch != 0:
                pass
            else:
                continue
                
            # text
            text_path = os.path.join(self.text_dir, f'{file_name}/{wav_file_name}.csv')
            df_text = pd.read_csv(text_path)
            df_text[pd.isna(df_text['asr_recog'])] = ''
            texts = df_text['asr_recog'].tolist()
            eou = cur_usr_uttr_end-start            
            eou = min(len(texts)-1, eou+6)
            text = texts[eou]
            text = self.clean(text)
            
            kana = None
            
            idx = self.token2idx(text)
            
            if text == '':
                continue
            
            if self.is_use_eou:
                idx = idx+[self.eou_id]
                text += '<eou>'                       

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

def get_dataset(config, cv_id, split='train', subsets=["M1_all"], speaker_list=None):
    dataset = ATRDataset(config, cv_id, split, subsets, speaker_list)
    return dataset


def get_dataloader(dataset, config, shuffle=True):
    dataloader = create_dataloader(dataset, config.optim_params.batch_size, shuffle=shuffle)
    return dataloader
