import MeCab
import jaconv
import re
import wave
import struct
import numpy as np
import pandas as pd
import os
import json
import glob
from tqdm import tqdm

from hiragana2julius import hiragana2julius

DATAROOT="/mnt/aoni04/jsakuma/data/ATR-Fujie"
file_names = sorted(os.listdir("/mnt/aoni04/jsakuma/data/ATR2022/julius/M1/wav"))

tag = MeCab.Tagger()

error_dic = {'file': [], 'idx': []}

def julius2text(file_name):
    df_turns_path = os.path.join(DATAROOT, 'dataframe3/{}.csv'.format(file_name))
    df_vad_path = os.path.join(DATAROOT,'vad/{}.csv'.format(file_name))
    json_turn_path = os.path.join(DATAROOT, 'samples/json/{}_samples.json'.format(file_name))
    feat_list = os.path.join(DATAROOT, 'samples/CNN_AE/{}/*_spec.npy'.format(file_name))
    wav_list = os.path.join(DATAROOT, 'samples/wav/{}/*.wav'.format(file_name))
    text_agent_path = os.path.join("/mnt/aoni04/jsakuma/data/ATR2022", 'Text_Agent/{}.csv'.format(file_name))
    feat_list = sorted(glob.glob(feat_list))
    wav_list = sorted(glob.glob(wav_list))

    df = pd.read_csv(df_turns_path)
    df_vad = pd.read_csv(df_vad_path)
    df_agent_text = pd.read_csv(text_agent_path)

    filename1 = file_names[j] #'20131106-4_01'
    filename2 = filename1.replace('-', '_')

    dic = {'start': [], 'end': [], 'word': [], 'error_code': []}
    for idx in range(len(df_agent_text)):

        text_org = df_agent_text['text'].iloc[idx]
        text = re.sub(r"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", "", text_org)
        text = re.sub(r"[a-zA-Z]", "", text)    
        code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠？！｀＋￥％]')
        text = code_regex.sub('', text)
        word_list = tag.parse(text).replace(' ', '').split()[::2][:-1]
        yomi_kata_list = [out.split(',')[-2] for out in tag.parse(text).replace(' ', '').split()[1:][::2]]
        assert len(word_list)==len(yomi_kata_list), 'num mismatch'

        phons = []
        for yomi in yomi_kata_list:
            if yomi == '。' or yomi == '、':
                phons.append('')
            else:        
                phons.append(hiragana2julius(jaconv.kata2hira(yomi)))

        dic_wp = {'word': word_list, 'phon': phons}

        path = '/mnt/aoni04/jsakuma/data/ATR2022/julius/M1/wav/{}/{}_{:03}.lab'.format(filename1, filename2, idx+1)
        with open(path) as f:
            lines = f.readlines()

        output=[]
        for line in lines:
            s, e, phon = line.split()
            if phon == 'silB' or phon == 'silE':
                continue        

            output.append(phon)

        julius = ' '.join(output)


        phon_list = []
        for phon in dic_wp['phon']:
            if phon != '':
                phon_list.append(phon)

        try:
            point = 0
            out_list = []
            for i, phon in enumerate(phon_list):

                n = len(phon)
                if julius[point:point+n] == phon:
                    point += n+1
                    out_list.append(phon)
                else:
                    if i == len(phon_list)-1: # 間違っていた単語が最後の時
                        out_list.append(julius[point:])
                    else:
                        nxt = phon_list[i+1]            
                        cnt = 0
                        while julius[point+cnt:point+cnt+len(nxt)]!=nxt:
                            cnt += 1
                            assert cnt < len(julius), 'loop error:{}'.format(text_org)

                        out_list.append(julius[point:point+cnt-1])
                        point += len(julius[point:point+cnt-1])+1
        except:
            # print(idx, julius)
            error_dic['file'].append(file_name)
            error_dic['idx'].append(idx)
            dic['start'].append(int(df_agent_text.iloc[idx]['start']))
            dic['end'].append(int(df_agent_text.iloc[idx]['end']))
            dic['word'].append(df_agent_text.iloc[idx]['text'])
            dic['error_code'].append(1)
            continue

        new_phon = []
        cnt = 0
        for phon in dic_wp['phon']:
            if phon == '':
                new_phon.append('')
            else:
                new_phon.append(out_list[cnt])
                cnt += 1

        dic_wp['new_phon'] = new_phon


        assert len(dic_wp['new_phon'])==len(dic_wp['word']), 'length, mismatch'

        start = df_agent_text['start'].iloc[idx]
        tmp = ''
        w_tmp = ''
        s_tmp = 0
        cnt = 0
        #dic = {'start': [], 'end': [], 'word': []}
        for line in lines:

            s, e, phon = line.split()
            if phon == 'silB' or phon == 'silE' or phon == 'sp':
                continue

            s = float(s)*1000
            e = float(e)*1000

            while dic_wp['new_phon'][cnt]=='': # そのほか(英語・数字・記号の場合)、次の単語更新時につける
                if dic_wp['word'][cnt] == '、' or dic_wp['word'][cnt] == '。': # 句読点はこのタイミングでつけてしまう
                    dic['word'][-1]+=dic_wp['word'][cnt]
                    cnt += 1
                else:
                    w_tmp += dic_wp['word'][cnt]
                    cnt += 1

                if cnt >= len(dic_wp['new_phon']):
                    break

            if tmp == '':
                tmp += phon
                s_tmp = s
            else:
                tmp += (' '+phon)

            if tmp == dic_wp['new_phon'][cnt]:
                dic['start'].append(s_tmp+start)
                dic['end'].append(e+start)
                dic['word'].append(w_tmp+dic_wp['word'][cnt])
                dic['error_code'].append(0)
                tmp = ''
                w_tmp = ''
                cnt += 1

    df_julius = pd.DataFrame(dic)
    
    df_spk1 = df[df['spk']==1]
    text_list = []
    for i in range(len(df_spk1)):
        start = df_spk1['start'].iloc[i]
        end = df_spk1['end'].iloc[i]

        text = ''.join(df_julius[(df_julius['start']>start) & (df_julius['end']<end)]['word'].tolist())
        text_list.append(text)

    df_spk1['text'] = text_list
    
    
    df_path1 = '/mnt/aoni04/jsakuma/data/ATR-Fujie/julius2text/turn_agent/{}.csv'.format(file_name)
    df_path2 = '/mnt/aoni04/jsakuma/data/ATR-Fujie/julius2text/julius/{}.csv'.format(file_name)
    df_spk1.to_csv(df_path1, encoding='utf-8', index=False)
    df_julius.to_csv(df_path2, encoding='utf-8', index=False)
    
    
for j, file_name in enumerate(tqdm(file_names)): 
    julius2text(file_name)
    
df_error = pd.DataFrame(error_dic)
df_path3 = '/mnt/aoni04/jsakuma/data/ATR-Fujie/julius2text/errors/{}.csv'.format(file_name)
df_error.to_csv(df_path3, encoding='utf-8', index=False)