import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


DATAROOT="/mnt/aoni04/jsakuma/data/ATR-Trek"
VADDIR  = os.path.join(DATAROOT, "vad")
OFFSET = 400  # vad hangover


def get_text_csv(name):
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
                                                'Unnamed: 5': 'text',
                                                '継続': 'conti_flg',
                                                '相槌/聞き返し/復唱': 'da',
                                               }
                  )

    columns = ['start1', 'start2', 'end1', 'end2', 'text', 'conti_flg', 'da']
    df_text_user = df_text_user[columns]
    df_text_user = df_text_user[df_text_user['text']!='-']
    df_text_user = df_text_user[df_text_user['text']==df_text_user['text']]
    
    df_text_agent = pd.read_csv(text_agent_path)

    return df_text_user, df_text_agent

def get_vad_csv(name):
    vad_path = os.path.join(VADDIR, name+'.csv')
    df_vad = pd.read_csv(vad_path)
    
    return df_vad

def get_turn_df(file_name):
    new_dict = {'start': [], 'end': [], 'text': [], 'offset': [], 'next_start': [], 'next_end': []}#, 'overlap': []}

    df_text_user, df_text_agent = get_text_csv(file_name)
    df_vad = get_vad_csv(file_name)

    df_vad_user = df_vad[df_vad['spk']==0]
    df_vad_agent = df_vad[df_vad['spk']==1]

    flg = False
    text = ''
    tmp = ''
    for i in range(len(df_text_user)):

        if df_text_user['start1'].iloc[i] != df_text_user['start1'].iloc[i]:
            continue

        start=int(df_text_user['start1'].iloc[i]*1000)#+OFFSET
        end=int(df_text_user['end2'].iloc[i]*1000)#-OFFSET

        if df_text_user['da'].iloc[i] in ['BP', 'BN']:
            continue

        if df_text_user['conti_flg'].iloc[i] == 1:
            if not flg:
                ostart = start

            tmp += df_text_user['text'].iloc[i]
            #tmp = tmp.replace('　', '').replace('、', '').replace('。', '')
            text += tmp
            flg = True
            tmp = ''
            continue            

        tmp = df_text_user['text'].iloc[i]
        #tmp = tmp.replace('　', '').replace('、', '').replace('。', '')

        if flg:
            start = ostart
            text += tmp
        else:
            text = tmp

        flg = False
        if end-start<1000:
            text = ''
            tmp = ''
            continue

        #if text.replace('　', '').replace('、', '').replace('。', '') in ['はい', 'はいはい', '']:
        if text.replace('　', '').replace('、', '').replace('。', '') in ['']:
            text = ''
            tmp = ''
            continue

        if len(df_vad_agent[(df_vad_agent['start']<end-OFFSET) & (df_vad_agent['end']>end-OFFSET)])>0:
            agent_start, agent_end = df_vad_agent[(df_vad_agent['start']<end-OFFSET) & (df_vad_agent['end']>end-OFFSET)][['start', 'end']].iloc[0]
#             overlap = True
        else: 
            if len(df_vad_agent[df_vad_agent['start']>end-OFFSET])<1:
                continue
            agent_start, agent_end = df_vad_agent[df_vad_agent['start']>end-OFFSET][['start', 'end']].iloc[0]            
#             overlap = False

        new_dict['start'].append(start)
        new_dict['end'].append(end)
        new_dict['text'].append(text)
        new_dict['offset'].append(agent_start-(end-OFFSET))
        new_dict['next_start'].append(agent_start)
        new_dict['next_end'].append(agent_end)
#         new_dict['overlap'].append(overlap)

        text = ''
        tmp = ''
        
    df_turn = pd.DataFrame(new_dict)
    return df_turn


if __name__ == '__main__':
    spk_file_path = os.path.join(DATAROOT, 'speaker_ids.csv')
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
    
    speaker_list = None #['M1']
    for split in ['train', 'valid', 'test']:
        name_path = os.path.join(DATAROOT, "names/{}.txt".format(split))
        with open(name_path) as f:
            lines = f.readlines()

        file_names = [line.replace('\n', '') for line in lines]
        if speaker_list is not None:
            file_names = [name for name in file_names if spk_dict[name+'.wav'] in speaker_list]
        
        for file_name in tqdm(file_names):
            df_turn = get_turn_df(file_name)                       
        
            out_path = os.path.join(DATAROOT, 'turn_csv2/{}.csv'.format(file_name))
            df_turn.to_csv(out_path)