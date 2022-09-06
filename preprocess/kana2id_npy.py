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


path = '/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr_kana/asr1/data/jp_token_list/word/tokens.txt'
with open(path) as f:
    lines =f.readlines()
    
tokens = [line.split()[0] for line in lines]

def kana2idx(kana): 
    if kana != kana or kana == '':
        return [0]
    
    idxs = [tokens.index(k) for k in kana.split()]
    
    return idxs


out_dict = {}

for split in ['valid', 'train', 'test']:
    speaker_list=['M1']
    

    name_path = "/mnt/aoni04/jsakuma/data/ATR2022/asr/names/{}.txt".format(split)
    with open(name_path) as f:
        lines = f.readlines()

    file_names = [line.replace('\n', '') for line in lines]
    if speaker_list is not None:
        file_names = [name for name in file_names if spk_dict[name+'.wav'] in speaker_list]

    frame_length = 50
    sample_rate = 16000
    max_positive_length = 20
    context_num = 0
    for file_name in tqdm(file_names):

        # 各種ファイルの読み込み    
        df_turns_path = os.path.join(DATAROOT, 'dataframe3/{}.csv'.format(file_name))
        df_turn_text_path = os.path.join(DATAROOT, 'dataframe0526/{}.csv'.format(file_name))
        df_vad_path = os.path.join(DATAROOT,'vad/{}.csv'.format(file_name))
        #df_vad_path = os.path.join(DATAROOT,'vad_offset200_wo_hang_over/{}.csv'.format(file_name))
        json_turn_path = os.path.join(DATAROOT, 'samples/json/{}_samples.json'.format(file_name))
        feat_list = os.path.join(DATAROOT, 'samples/CNN_AE/{}/*_spec.npy'.format(file_name))
        wav_list = os.path.join(DATAROOT, 'samples/wav/{}/*.wav'.format(file_name))
        feat_list = sorted(glob.glob(feat_list))
        wav_list = sorted(glob.glob(wav_list))

        df = pd.read_csv(df_turns_path)
        df_turn_text = pd.read_csv(df_turn_text_path)
        df_vad = pd.read_csv(df_vad_path)

        with open(json_turn_path, 'r') as jf:
            turn_info = json.load(jf)

        # 対話全体の長さ確認
        N = turn_info['length']//sample_rate*1000

        # vadの結果
        uttr_user = np.zeros(N//frame_length)
        uttr_agent = np.zeros(N//frame_length)      
        for i in range(len(df_vad)):
            spk = df_vad['spk'].iloc[i]
            start = (df_vad['start'].iloc[i]) // frame_length
            end = (df_vad['end'].iloc[i]) // frame_length

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
            # text = df['text'].iloc[t]
            start=turn_info['start'][t]//frame_length
            end = turn_info['end'][t]//frame_length
            cur_usr_uttr_end = turn_info['cur_usr_uttr_end'][t]//frame_length
            timing = turn_info['next_sys_uttr_start'][t]//frame_length

            if df['next_sys_uttr_start'].iloc[t]-df['offset'].iloc[t] == df['end'].iloc[t]:
                is_barge_in = False
            else:
                is_barge_in = True

            if end - timing > max_positive_length:  # システム発話をどれくらいとるか
                end = timing + max_positive_length

            vad_user = uttr_user[start:end]
            vad_agent = uttr_agent[start:end]

            turn_label = np.zeros(N//frame_length)
            turn_label[start:cur_usr_uttr_end] = 1
            turn_label = turn_label[start:end]

            timing_target = np.zeros(N//frame_length)
            timing_target[timing:] = 1

            turn_timing_target = timing_target[start:end]

        #     last_ipu_user = self.get_last_ipu(vad_user)
        #     last_ipu_agent = self.get_last_ipu(vad_agent)

        #             turn_user = np.ones(len(turn_user)) - turn_user
        #             turn_agent = np.ones(len(turn_agent)) - turn_agent

            if ch == 0 and next_ch != 0:
                vad_label = vad_user
        #         last_ipu = last_ipu_user
            else:
                continue

            # text
            contexts = df_turn_text['text'].iloc[max(0, t-context_num):t].tolist()
            spks = df_turn_text['spk'].iloc[max(0, t-context_num):t].tolist()
            context = ''
            for s in range(len(spks)):
                context += '<spk{}>'.format(spks[s]+1)+str(contexts[s])

            text_path = '_'.join(wav_path.replace('wav', 'text2').split('_')[:-1])+'.csv'
            df_text = pd.read_csv(text_path)
            df_text[pd.isna(df_text['asr_recog'])] = ''
            text = df_text['asr_recog'].tolist()
            text = [txt+context for txt in text]

            kana_path = '_'.join(wav_path.replace('wav', 'kana2').split('_')[:-1])+'.csv'
            
            out_path = kana_path.replace('kana2', 'kana_ids').replace('kana2', 'kana_ids').replace('.csv', '')
            out_name = out_path.split('/')[-1]            
            
            df_kana = pd.read_csv(kana_path)
            df_kana[pd.isna(df_kana['asr_recog'])] = ''
            kana = df_kana['asr_recog'].tolist()            
            idx = [kana2idx(k) for k in kana]

#             max_length = max([len(ii) for ii in idx])
#             new_idx = np.zeros([len(idx), max_length])
#             for i, ii in enumerate(idx):
#                 new_idx[i][:len(ii)] = ii           

            out_dict[out_name] = idx
            
    json_path = '/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/kana_ids/{}.json'.format(split)
    with open(json_path, 'w') as f:
        json.dump(out_dict, f, indent=4)