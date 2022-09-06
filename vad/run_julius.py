import os
import glob
import subprocess
from tqdm import tqdm


# def run_julius(path):
#     cmd = ["./run.sh $0", "{}".format(path)]
#     subprocess.check_call(cmd, shell=True)

def run_julius(path):
    cmd = ["./test.sh $0", "{}".format(path)]
    subprocess.check_call(cmd, shell=True)
    
if __name__ == "__main__":
    paths = glob.glob('/mnt/aoni04/jsakuma/data/ATR2022/asr_agent/wav_julius/*')
    run_julius(paths[0])
#     for path in tqdm(paths):        
#         run_julius(path)
