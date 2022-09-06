import json
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm


DATAROOT="/mnt/aoni04/jsakuma/data/ATR-Fujie"
split='valid'
speaker_list = ['F1', 'F2', 'M1', 'M2', 'M3', 'M4']

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
    name_path = "/mnt/aoni04/jsakuma/data/ATR2022/asr/names/{}.txt".format(split)
    with open(name_path) as f:
        lines = f.readlines()

    file_names = [line.replace('\n', '') for line in lines]
    if speaker_list is not None:
        file_names = [name for name in file_names if spk_dict[name+'.wav'] in speaker_list]


    for file_name in tqdm(file_names):
        df_out_path = '/mnt/aoni04/jsakuma/data/ATR-Fujie/dataframe0526/{}.csv'.format(file_name)
        if os.path.isfile(df_out_path):
            continue
        
        # 各種ファイルの読み込み
        df_turns_path = os.path.join(DATAROOT, 'dataframe3/{}.csv'.format(file_name))
        df_vad_path = os.path.join(DATAROOT,'vad/{}.csv'.format(file_name))
#         df_agent_path = os.path.join(DATAROOT,'julius2text/turn_agent/{}.csv'.format(file_name))
        json_turn_path = os.path.join(DATAROOT, 'samples/json/{}_samples.json'.format(file_name))
        feat_list = os.path.join(DATAROOT, 'samples/CNN_AE/{}/*_spec.npy'.format(file_name))
        wav_list = os.path.join(DATAROOT, 'samples/wav/{}/*.wav'.format(file_name))
        feat_list = sorted(glob.glob(feat_list))
        wav_list = sorted(glob.glob(wav_list))

        df = pd.read_csv(df_turns_path)
        df_vad = pd.read_csv(df_vad_path)
#         df_agent_text = pd.read_csv(df_agent_path)

        with open(json_turn_path, 'r') as jf:
            turn_info = json.load(jf)

#         text_list = []
#         num_turn = len(turn_info['speaker'])
#         for i in range(num_turn):
#             spk = df['spk'].iloc[i]
#             start = df['start'].iloc[i]

#             if spk==1:
#                 text = df_agent_text[df_agent_text['start']==start]['text'].iloc[0]
#             else:
#                 wav_path = wav_list[i]
# #                 text_path = '_'.join(wav_path.replace('wav', 'text').split('_')[:-1])+'.csv'
#                 text_path = ('_'.join(wav_path.split('_')[:-1])+'.csv').replace('wav', 'text_10_4_2')
#                 df_text = pd.read_csv(text_path)
#                 df_text[pd.isna(df_text['asr_recog'])] = ''
#                 text = df_text['asr_recog'].tolist()[-1]

#             text_list.append(text)

#         if len(df)==len(text_list)+1:
#             text_list.append('')
            
#         df['text'] = text_list
     
        df.to_csv(df_out_path, encoding='utf-8', index=False)