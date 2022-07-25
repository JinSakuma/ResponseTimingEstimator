import json
import wave
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm


def get_text_file(name):
    text_user_path = '/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_User/{}_L.xls'.format(name)
    text_agent_path = '/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_Agent/{}.csv'.format(name)
    
    input_book = pd.ExcelFile(text_user_path)
    input_sheet_name = input_book.sheet_names
    df_text_user = input_book.parse(input_sheet_name[0])

    df_text_user = df_text_user.rename(columns={'Unnamed: 0': 'start1',
                                                'Unnamed: 1': 'start2',
                                                'Unnamed: 2': 'end1',
                                                'Unnamed: 3': 'end2',
                                                'Unnamed: 4': 'file_name',          
                                                'Unnamed: 5': 'transcript',
                                               }
                  )

    columns = ['start1', 'start2', 'end1', 'end2', 'transcript']
    df_text_user = df_text_user[columns]
    df_text_user = df_text_user[df_text_user['transcript']!='-']
    df_text_user = df_text_user[df_text_user['transcript']==df_text_user['transcript']]
    
    df_text_agent = pd.read_csv(text_agent_path)

    return (df_text_user, text_user_path), (df_text_agent, text_agent_path)

# speaker取得
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

file_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/turn_info/*_sorted.turn.txt'))
user_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_User/*_L.xls'))
agent_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_Agent/*.csv'))

file_names1 = [file_path.split('/')[-1].replace('_sorted.turn.txt', '') for file_path in file_list] 
file_names2 = [file_path.split('/')[-1].replace('_L.xls', '') for file_path in user_list] 
file_names3 = [file_path.split('/')[-1].replace('.csv', '') for file_path in agent_list] 

filenames = list(set(file_names1) & set(file_names2) & set(file_names3))
spk_ids = df_spk[df_spk['ファイル名'].isin([f+'.wav' for f in filenames])]['operator'].to_list()
spk_dict  =dict(zip(filenames, spk_ids))


df_list = [] # dfを格納
meta_dict = {} # ファイルごとの情報を格納


# 全ファイルのpath
file_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/turn_info/*_sorted.turn.txt'))

for file_name in tqdm(filenames):
    file_path = '/mnt/aoni04/jsakuma/data/ATR/turn_info/{}_sorted.turn.txt'.format(file_name)
    spk_id = spk_dict[file_name]
    
    df=pd.read_csv(file_path)
    
    # ユーザのテキスト取得
    (df_text_user, text_user_path), (df_text_agent, text_agent_path) = get_text_file(file_name)
    text_list = []
    try:
        for i in range(len(df)):
            text = ''
            spk=df['ch'].iloc[i]
            if spk==0:
                # start=(df['start'].iloc[i]-200)/1000
                start=df['start'].iloc[i]/1000
                end=df['end'].iloc[i]/1000
                df_tmp = df_text_user[(df_text_user['end2']>start) & (df_text_user['start2']<end)]
                text = ''.join(df_tmp['transcript'].to_list())

            else:
                start=df['start'].iloc[i]
                end=df['end'].iloc[i]
                df_tmp = df_text_agent[(df_text_agent['end']>start) & (df_text_agent['start']<end)]
                text = ''.join(df_tmp['text'].to_list()) 

            text_list.append(text)
    except:
        print(file_name)
        print(aaaa)
        
    df['text'] = text_list
    
    # turn交代しない場合は結合する
    ch_list, start_list, end_list, text_list = [], [], [], []
    pre = df['ch'].iloc[0]
    start = df['start'].iloc[0]
    end = df['end'].iloc[0]
    text = df['text'].iloc[0]

    for i in range(1, len(df)):
        spk = df['ch'].iloc[i]
        if spk!=pre:
            ch_list.append(pre)
            start_list.append(start)
            end_list.append(end)
            text_list.append(text)

            pre = df['ch'].iloc[i]
            start = df['start'].iloc[i]
            end = df['end'].iloc[i]
            text = df['text'].iloc[i]
        else:
            end = df['end'].iloc[i]
            text += df['text'].iloc[i]

    if spk==pre:
        end = df['end'].iloc[i]
        text += df['text'].iloc[i]
    
    ch_list.append(pre)
    start_list.append(start)
    end_list.append(end)
    text_list.append(text)

    df2 = pd.DataFrame({'ch': ch_list, 'start': start_list, 'end': end_list, 'text': text_list})
    
    
    # システム発話のオフセット取得
    offsets = [-100000]
    for i in range(1, len(df2)):
        spk=df2['ch'].iloc[i]
        eou = df2['end'].iloc[i-1]
        start = df2['start'].iloc[i]
        offset = start-eou

        offsets.append(offset)

    df2['offset'] = offsets
    
    df_path = '/mnt/aoni04/jsakuma/data/ATR2022/dataframe/{}.csv'.format(file_name)
    wavpath = '/mnt/aoni04/jsakuma/data/ATR/wav_safia/{}.wav'.format(file_name)
    wavpath_user = '/mnt/aoni04/jsakuma/data/ATR2022/wav_mono/{}_user.wav'.format(file_name)
    wavpath_agent = '/mnt/aoni04/jsakuma/data/ATR2022/wav_mono/{}_agent.wav'.format(file_name)
    
    df_list.append(df2)
    json_dict = {'name': file_name,
                 'spker': spk_id,
                 'df_path': df_path,
                 'wav_path': wavpath,
                 'wav_path_user': wavpath_user,
                 'wav_path_agent': wavpath_agent,
                 'text_path_user': text_user_path,
                 'text_path_agent': text_agent_path,
         }
    
    # 出力
    columns = ['ch', 'start', 'end', 'text', 'offset']
    df2 = df2[columns]
    
    df_path = json_dict['df_path']
    json_path = '/mnt/aoni04/jsakuma/data/ATR2022/json/{}.json'.format(file_name)
    
    df2.to_csv(df_path, encoding='utf-8', index=False)
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)