import os
import glob
import numpy as np
import pandas as pd

from tqdm import tqdm


OUTDIR = "/mnt/aoni04/jsakuma/data/ATR2022/vad"


def get_df(vad_out, role=0):
    starts, ends = [], []
    for line in vad_out:
        line_ = line.replace('\n', '').split(' ')
        starts.append(int(float(line_[-3])))
        ends.append(int(float(line_[-1])))
        
    roles = [role]*len(starts)

    return pd.DataFrame({'start': starts, 'end': ends, 'role': roles})


def get_vad_results(name):
    vad_out_path = "./outputs/{}_vad_output.txt".format(name)
    with open(vad_out_path) as f:
        lines = f.readlines()

    id_list = []
    for i, line in enumerate(lines):
        if 'VAD' not in line:
            id_list.append(i)
        
    user_vad_out = lines[id_list[0]+1:id_list[1]]
    agent_vad_out = lines[id_list[1]+1:-1]

    df_user = get_df(user_vad_out, role=0)
    df_agent = get_df(agent_vad_out, role=1)

    df = pd.concat([df_user, df_agent]).sort_values('start')
    return df


def save_df(df, name):
    out_path = os.path.join(OUTDIR, "{}.csv".format(name))
    df.to_csv(out_path, encoding='utf-8', index=False) 
        

name='20131120-1_02'
df = get_vad_results(name)    
if __name__ == "__main__":
    file_list = sorted(glob.glob('/mnt/aoni04/jsakuma/development/ATR2022/VAD/outputs/*.txt'))
    file_names = [file_path.split('/')[-1].replace('_vad_output.txt', '') for file_path in file_list] 
    
    for name in tqdm(file_names):        
        df = get_vad_results(name)
        save_df(df, name)
