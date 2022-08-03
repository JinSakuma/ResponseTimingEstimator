import os
import pandas as pd
import glob
import subprocess
from tqdm import tqdm

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


def run_julius(path):
    cmd = ["perl segment_julius2.pl {}".format(path)]
    subprocess.check_call(cmd, shell=True)
    
if __name__ == "__main__":
    path_list = glob.glob('/mnt/aoni04/jsakuma/data/ATR2022/julius/M1/wav/*')
    #path_list = []
    #files = df_spk[df_spk['operator']=='M1']['ファイル名'].tolist()
    #for path in tqdm(paths):
    #    if path.split('/')[-1]+'.wav' in files:
    #        path_list.append(path)

    #path_ = '/mnt/aoni04/jsakuma/data/ATR2022/asr_agent/wav_julius/20131101-2_02'
    for path in tqdm(path_list[:1]):
        path = "/mnt/aoni04/jsakuma/data/ATR2022/julius/M1/wav/20131111-2_01"        
        run_julius(path)
