import json
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

file_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR-Fujie/duration_v2/*.txt'))
columns = ["start", "end", "spk", "next_sys_uttr", "next_usr_uttr", "offset", "length"]
df_list = []
name_list = []
for j, file_path in enumerate(tqdm(file_list)): 
    name = file_path.split('/')[-1].replace('.dur.txt', '')
    with open(file_path) as f:
        lines = f.readlines()

    ch0 = []
    ch1 = []
    for i, line in enumerate(lines):
        if 'CH 0' in line:
            ch0.append(i)
        elif 'CH 1' in line:
            ch1.append(i)

    vad_start0, vad_end0 = ch0[:2]
    vad_start1, vad_end1 = ch1[:2]

    turn_start0, turn_end0 = ch0[2:]
    turn_start1, turn_end1 = ch1[2:]

    vad_info0 = {'start': [], 'end': []}
    for line in lines[vad_start0+1:vad_end0]:
        line = line.replace('\n', '')
        start, end = line.split(',')
        vad_info0['start'].append(int(start))
        vad_info0['end'].append(int(end))

    vad_info1 = {'start': [], 'end': []}
    for line in lines[vad_start1+1:vad_end1]:
        line = line.replace('\n', '')
        start, end = line.split(',')
        vad_info1['start'].append(int(start))
        vad_info1['end'].append(int(end))

    df0_vad = pd.DataFrame(vad_info0)
    df1_vad = pd.DataFrame(vad_info1)


    turn_info0 = {'start': [], 'end': [], 'length': []}
    for line in lines[turn_start0+1:turn_end0]:
        line = line.replace('\n', '')
        start, end, length = line.split(',')
        turn_info0['start'].append(int(start))
        turn_info0['end'].append(int(end))
        turn_info0['length'].append(length)

    turn_info1 = {'start': [], 'end': [], 'length': []}
    for line in lines[turn_start1+1:turn_end1]:
        line = line.replace('\n', '')
        start, end, length = line.split(',')
        turn_info1['start'].append(int(start))
        turn_info1['end'].append(int(end))
        turn_info1['length'].append(length)

    df0_turn = pd.DataFrame(turn_info0)
    df1_turn = pd.DataFrame(turn_info1)

    df0_vad['spk'] = [0]*len(df0_vad)
    df1_vad['spk'] = [1]*len(df1_vad)

    df_vad = pd.concat([df0_vad, df1_vad]).sort_values('start')

#     df0 = df0_turn#[df0_turn['length']=='T']
#     df0['spk'] = [0]*len(df0)

#     df1 = df1_turn#[df1_turn['length']=='T']
#     df1['spk'] = [1]*len(df1)

#     df = pd.concat([df0, df1]).sort_values('start')
    
#     remove_list = []
#     pre_spk = -1
#     pre_end = -1
#     new_start = -1
#     for i in range(len(df)):
#         spk = df['spk'].iloc[i]
#         start = df['start'].iloc[i]
#         end = df['end'].iloc[i]
#         l_type = df['length'].iloc[i]
#         if l_type == 'S':
#             if pre_spk != spk and pre_end<start:
#                 remove_list.append(False)
#                 pre_spk = spk
#                 pre_end = end
#                 new_start = start
#             else:
#                 remove_list.append(True)
#         else:
#             remove_list.append(False)
#             if new_start != -1:
#                 df['start'].iloc[i] = new_start
#                 new_start = -1

#             pre_spk = spk
#             pre_end = end

#     df['remove'] = remove_list
#     df = df[df['remove']==False]
#     df = df[df['length']=='T']

#     # システム発話のオフセット取得
#     offsets = []
#     next_sys_uttr = []
#     next_usr_uttr = []
#     for i in range(1, len(df)):
#     #     eou = df['end'].iloc[i-1]
#     #     start = df['start'].iloc[i]
#     #     offset = start-eou
#         start = df['start'].iloc[i]
#         df_tmp = df_vad[(df_vad['end']>start) & (df_vad['start']<start)]
        
#         if len(df_tmp)>0:
#             offset = start-df_tmp['end'].iloc[0]
#             pre_spk = df_tmp['spk'].iloc[0]
#             pre_end = df_tmp['end'].iloc[0]
#         else:
#             offset=start-df_vad[df_vad['end']<start].iloc[-1]['end']
#             pre_spk = df_vad[df_vad['end']<start].iloc[-1]['spk']
#             pre_end = df_vad[df_vad['end']<start].iloc[-1]['end']

#         df_tmp = df_vad[df_vad['spk']==pre_spk]
#         df_next_usr = df_tmp[df_tmp['start']>pre_end]
#         if len(df_next_usr)>0:
#             next_usr = df_next_usr['start'].iloc[0]
#         else:
#             next_usr = 100000
        
#         next_sys_uttr.append(start)
#         next_usr_uttr.append(next_usr)
#         offsets.append(offset)

#     df['offset'] = offsets+[100000]
#     df['next_sys_uttr'] = next_sys_uttr+[100000]
#     df['next_usr_uttr'] = next_usr_uttr+[100000]
#     df = df[columns]

    out_path = '/mnt/aoni04/jsakuma/data/ATR-Fujie/vad-fujie/{}.csv'.format(name)
    df_vad.to_csv(out_path, encoding='utf-8', index=False)