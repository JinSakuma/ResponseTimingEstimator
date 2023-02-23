import os
import numpy as np
import pandas as pd
from tqdm import tqdm


DATAROOT="/mnt/aoni04/jsakuma/data/ATR_Annotated"
MIN=-500
MAX=2000

def get_csv(name):
    path = os.path.join(DATAROOT, "csv", name+'.csv')
    df = pd.read_csv(path)
    
    return df

def get_vad_csv(name):
    vad_path = os.path.join(DATAROOT, "vad", name+'.csv')
    df_vad = pd.read_csv(vad_path)
    
    return df_vad

if __name__ == '__main__':
    names1 = list(set([name.replace('.wav', '').replace('_user', '').replace('_agent', '') for name in os.listdir(os.path.join(DATAROOT, "wav_mono"))]))
    names2 = [name.replace('.csv', '') for name in os.listdir(os.path.join(DATAROOT, "csv"))]
    file_names = [name for name in names2 if name in names1]
    
    for file_name in tqdm(file_names):
        df = get_csv(file_name)
        
        # 発話タイミングの範囲
        df = df[(df['offset']>MIN) & (df['offset']<MAX)]        
        # 現話者: User, 次話者: エージェントの場合
        df = df[(df['spk']==1) & (df['nxt_spk']==0)]
        
        out_dir = os.path.join(DATAROOT, 'data_{}_{}/csv'.format(MIN, MAX))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(DATAROOT, 'data_{}_{}/csv/{}.csv'.format(MIN, MAX, file_name))
        df.to_csv(out_path)