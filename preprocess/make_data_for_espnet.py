import torch
import torchaudio
import codecs
import wave
import numpy as np
import pandas as pd
import os
import glob
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_text_file(name):
    text_user_path = '/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_User/{}_L.xls'.format(name)
    text_agent_path = '/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_Agent/{}.csv'.format(name)
    
#     input_book = pd.ExcelFile(text_user_path)
#     input_sheet_name = input_book.sheet_names
#     df_text_user = input_book.parse(input_sheet_name[0])

#     df_text_user = df_text_user.rename(columns={'Unnamed: 0': 'start1',
#                                                 'Unnamed: 1': 'start2',
#                                                 'Unnamed: 2': 'end1',
#                                                 'Unnamed: 3': 'end2',
#                                                 'Unnamed: 4': 'file_name',          
#                                                 'Unnamed: 5': 'transcript',
#                                                }
#                   )

#     columns = ['start1', 'start2', 'end1', 'end2', 'transcript']
#     df_text_user = df_text_user[columns]
#     df_text_user = df_text_user[df_text_user['transcript']!='-']
#     df_text_user = df_text_user[df_text_user['transcript']==df_text_user['transcript']]
    
    df_text_agent = pd.read_csv(text_agent_path)

    return None, df_text_agent


def save_turn_wav(wavepath, outpath, start, end, rate=16000, bits_per_sample=16):
    
    wf = wave.open(wavepath, 'r')

    # waveファイルが持つ性質を取得
    ch = wf.getnchannels()
    width = wf.getsampwidth()
    fr = wf.getframerate()
    fn = wf.getnframes()
    
    x = wf.readframes(wf.getnframes()) #frameの読み込み
    x = np.frombuffer(x, dtype= "int16") #numpy.arrayに変換
    
    turn = x[start*(rate//1000):end*(rate//1000)]

    wf.close()
    
    torchaudio.save(filepath=outpath, src=torch.tensor([turn]), sample_rate=rate, encoding="PCM_S", bits_per_sample=bits_per_sample)
    


TRAIN_SIZE=0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
SEED = 0
root='/mnt/aoni04/jsakuma/data/ATR2022/asr'

user_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_User/*_L.xls'))
agent_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_Agent/*.csv'))

file_names1 = [file_path.split('/')[-1].replace('_L.xls', '') for file_path in user_list] 
file_names2 = [file_path.split('/')[-1].replace('.csv', '') for file_path in agent_list]
file_names = sorted(list(set(file_names1) & set(file_names2)))

file_names_train, file_names_val_test = train_test_split(file_names, test_size=TEST_SIZE+VAL_SIZE, random_state=SEED)
file_names_val, file_names_test = train_test_split(file_names_val_test, test_size=TEST_SIZE/(VAL_SIZE+TEST_SIZE), random_state=SEED)
file_names_train = sorted(file_names_train)
file_names_val = sorted(file_names_val)
file_names_test = sorted(file_names_test)
file_dict = {'train': file_names_train, 'valid': file_names_val, 'test': file_names_test}

for split in ['train', 'valid', 'test']:
    path_names = os.path.join(root, 'names', split+'.txt')
    for j in tqdm(range(len(file_dict[split]))): 
        with open(path_names, mode='a') as f:
            f.write(file_dict[split][j]+'\n')

if __name__ == "__main__":
    for split in ['train', 'valid', 'test']:
        text_list = []
        utt2spk_list = []
        scp_list = []
        uttrs = []
        
        path_text = os.path.join(root, 'data/{}/text'.format(split))
        path_wavscp = os.path.join(root, 'data/{}/wav.scp'.format(split))
        path_utt2spk = os.path.join(root, 'data/{}/utt2spk'.format(split))
        
        #print(path_text)
        
        file_names_list = file_dict[split]
        for idx, file_name in enumerate(tqdm(file_names_list)):
            _, df_text_agent = get_text_file(file_name) 
            wavpath = os.path.join(root.replace('/asr_agent', ''), 'wav_mono/{}_agent.wav'.format(file_name))
            cnt = 0
            for i in range(len(df_text_agent)):
                if df_text_agent['start'].iloc[i] != df_text_agent['start'].iloc[i]:
                    continue
                
                start=int(df_text_agent['start'].iloc[i])
                end=int(df_text_agent['end'].iloc[i])
                
                if end-start<2000:
                    continue
                    
                cnt += 1
                
                text = df_text_agent['text'].iloc[i]
                text = text.replace('　', '').replace('、', '').replace('。', '')

                spk_id = 'U{:04}'.format(idx)
                uttr_id = spk_id+'_'+file_name.replace('-', '_')+'_{:03}'.format(cnt)
                name = spk_id+'_'+file_name.replace('-', '_')+'_{:03}.wav'.format(cnt)

                wav_out_dir = os.path.join(root, 'wav', file_name)
                os.makedirs(wav_out_dir, exist_ok=True)

                wav_out_path = os.path.join(wav_out_dir, name)
                save_turn_wav(wavpath, wav_out_path, start, end)
                
                text_list.append(uttr_id+' '+text+'\n')
                utt2spk_list.append(uttr_id+' '+spk_id+'\n')
                scp_list.append(uttr_id+' '+wav_out_path+'\n')
                uttrs.append(uttr_id)
        
        idxs = np.argsort(uttrs)
        for j in tqdm(idxs):                 
            with codecs.open(path_text,"a","utf-8") as f:
                f.write(text_list[j])
            
            with open(path_wavscp, mode='a') as f:
                f.write(scp_list[j])

            with open(path_utt2spk, mode='a') as f:
                f.write(utt2spk_list[j])

            # with open(path_spk2utt, mode='a') as f:
            #     f.write(spk2utt_list[j])         