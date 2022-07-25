import numpy as np
#import pandas as pd
import os
import glob
import subprocess
from tqdm import tqdm

PWD = "."
WAVPATH = '/mnt/aoni04/jsakuma/data/ATR2022/wav_mono2/'

def get_file_names():
    file_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/turn_info/*_sorted.turn.txt'))
    user_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_User/*_L.xls'))
    agent_list = sorted(glob.glob('/mnt/aoni04/jsakuma/data/ATR/ATR-Trek/Text_Agent/*.csv'))

    file_names1 = [file_path.split('/')[-1].replace('_sorted.turn.txt', '') for file_path in file_list] 
    file_names2 = [file_path.split('/')[-1].replace('_L.xls', '') for file_path in user_list] 
    file_names3 = [file_path.split('/')[-1].replace('.csv', '') for file_path in agent_list] 

    file_names = sorted(list(set(file_names1) & set(file_names2) & set(file_names3)))
    return file_names
    
def prepare_input_lists(name):
    path_w = os.path.join(PWD, 'inputs', '{}_vad_input.lst'.format(name))
    user = os.path.join(WAVPATH, '{}_user.wav'.format(name))
    agent = os.path.join(WAVPATH, '{}_agent.wav'.format(name))
    file_list = [user, agent]
    with open(path_w, mode='w') as f:
        f.write('\n'.join(file_list))

def run_vad(name):
    cmd = ["./vad.sh $0", "{}".format(name)]
    subprocess.check_call(cmd, shell=True)
    
if __name__ == "__main__":
    file_names = get_file_names()
    # file_names = ["20131101-1_01"]
    for name in tqdm(file_names):        
        prepare_input_lists(name)
        run_vad(name)
