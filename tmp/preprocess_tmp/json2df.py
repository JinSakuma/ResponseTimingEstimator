import json
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

# for split in ['test']:
# #for split in ['train', 'valid', 'test']:
#     name_path = "/mnt/aoni04/jsakuma/data/ATR2022/asr/names/{}.txt".format(split)
#     with open(name_path) as f:
#         lines = f.readlines()

#     file_names = [line.replace('\n', '') for line in lines]
#     info = {'file_id': [], 'file_name': [], 'is_removed' :[], 'text': [], 'phoneme': [], 'idx': [], 'feat_path' : []}
#     for file_name in tqdm(file_names):
#         names = os.listdir('/mnt/aoni04/jsakuma/data/ATR2022/asr/phoneme/{}'.format(file_name))

#         for name in names:
#             text_path = '/mnt/aoni04/jsakuma/data/ATR2022/asr/phoneme/{}/{}'.format(file_name, name)
#             feat_path = text_path.replace('phoneme', 'fbank').replace('.json', '_fbank.npy')

#             if not os.path.isfile(text_path) or not os.path.isfile(feat_path):
#                 continue

#             with open(text_path, 'r') as jf:
#                 text_dict = json.load(jf)

#             info['file_id'].append(file_name)
#             info['file_name'].append(name)
#             info['is_removed'].append(text_dict['is_removed'])
#             info['text'].append(text_dict['text'])
#             info['phoneme'].append(text_dict['phoneme'])
#             info['idx'].append(text_dict['idx'])
#             info['feat_path'].append(feat_path)

#     df = pd.DataFrame(info)

#     df_path = '/mnt/aoni04/jsakuma/data/ATR2022/asr/dataframe/{}.csv'.format(split)
#     df.to_csv(df_path, encoding='utf-8', index=False)

    
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

for split in ['test']:
#for split in ['train', 'valid', 'test']:
    df = pd.read_csv('/mnt/aoni04/jsakuma/data/ATR2022/asr/dataframe/{}.csv'.format(split))
    spk_list = []
    for i in tqdm(range(len(df))):
        file_name = df['file_id'].iloc[i]
        spk = spk_dict[file_name+'.wav']
        spk_list.append(spk)
        
    df['spk'] = spk_list
    df_path = '/mnt/aoni04/jsakuma/data/ATR2022/asr/dataframe/{}.csv'.format(split)
    df.to_csv(df_path, encoding='utf-8', index=False)