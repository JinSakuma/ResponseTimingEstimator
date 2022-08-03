import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import fromstring, int16
import os
import glob


offset = 0
speaker_list = ['M1']

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

file_names = os.listdir('/mnt/aoni04/jsakuma/data/ATR-Fujie/vad/')
file_names = [name.replace('.csv', '') for name in file_names]
if speaker_list is not None:
    file_names = [name for name in file_names if spk_dict[name+'.wav'] in speaker_list]

turn_out_path = '/mnt/aoni04/jsakuma/data/ATR-Fujie/annotations/M1/turn'
ipu_out_path = '/mnt/aoni04/jsakuma/data/ATR-Fujie/annotations/M1/ipu'

for file_name in tqdm(file_names):    
    df_out_path = '/mnt/aoni04/jsakuma/data/ATR-Fujie/dataframe0526/{}.csv'.format(file_name)
    
    if not os.path.isfile(df_out_path):
        print(df_out_path)
        continue
        
    df1 = pd.read_csv(df_out_path)

    df_out_path = '/mnt/aoni04/jsakuma/data/ATR-Fujie/vad/{}.csv'.format(file_name)
    df2 = pd.read_csv(df_out_path)

    start_turn = df1['start'].tolist()
    end_turn = df1['end'].tolist()
    spk_turn = df1['spk'].tolist()
    text_turn = df1['text'].tolist()

    start_vad = df2['start'].values+offset
    end_vad = df2['end'].values-offset
    spk_vad = df2['spk'].tolist()

    df_turn_user = pd.DataFrame({'start': start_turn,
                                 'end': end_turn,
                                 'spk': spk_turn,
                                 'text_user': text_turn,
                                })

    df_turn_agent = pd.DataFrame({'start': start_turn,
                                  'end': end_turn,
                                  'spk': spk_turn,
                                  'text_agent': text_turn,
                                 })



    df_vad_user = pd.DataFrame({'start': start_vad,
                                'end': end_vad,
                                'spk': spk_vad,
                                'turn_id_user': ['']*len(start_vad),
                                'dialog_acts_user': ['']*len(start_vad),
                                'notes_user': ['']*len(start_vad),
                                })

    df_vad_agent = pd.DataFrame({'start': start_vad,
                                'end': end_vad,
                                'spk': spk_vad,
                                'turn_id_agent': ['']*len(start_vad),
                                'dialog_acts_agent': ['']*len(start_vad),
                                'notes_agent': ['']*len(start_vad),
                                })

    df_turn_user = df_turn_user[df_turn_user['spk']==0]
    df_turn_agent = df_turn_agent[df_turn_agent['spk']==1]
    df_vad_user = df_vad_user[df_vad_user['spk']==0]
    df_vad_agent = df_vad_agent[df_vad_agent['spk']==1]
    
    df_turn_user.to_csv(os.path.join(turn_out_path, '{}_turn_user.csv'.format(file_name)), encoding='utf-8', index=False)
    df_turn_agent.to_csv(os.path.join(turn_out_path, '{}_turn_agent.csv'.format(file_name)), encoding='utf-8', index=False)
    df_vad_user.to_csv(os.path.join(ipu_out_path, '{}_ipu_user.csv'.format(file_name)), encoding='utf-8', index=False)
    df_vad_agent.to_csv(os.path.join(ipu_out_path, '{}_ipu_agent.csv'.format(file_name)), encoding='utf-8', index=False)