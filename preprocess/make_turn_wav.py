import numpy as np
import pandas as pd
import os
import wave
import glob
from tqdm import tqdm


DATAROOT="/mnt/aoni04/jsakuma/data/ATR_Annotated/data_-500_2000"
WAVDIR="/mnt/aoni04/jsakuma/data/ATR_Annotated/wav_mono"
CSVDIR  = os.path.join(DATAROOT, "csv")
MAX_AGENT_LENGTH = 2000  # [ms]
OFFSET = 300  


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
    
    waveFile = wave.open(outpath, 'wb')
    waveFile.setnchannels(ch)
    waveFile.setsampwidth(width)
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(turn))
    waveFile.close()

if __name__ == '__main__':
    file_paths = sorted(glob.glob(os.path.join(CSVDIR, '*.csv')))
    for file_path in tqdm(file_paths):
        df = pd.read_csv(file_path)
        file_name = file_path.split('/')[-1].replace('.csv', '')
                
        for i in range(len(df)):
            start, end, nxt_start, nxt_end, offset = df[['start', 'end', 'nxt_start', 'nxt_end', 'offset']].iloc[i]
            wav_start = start - OFFSET
            wav_end = max(end, nxt_start+MAX_AGENT_LENGTH) + OFFSET
            
            wavpath = os.path.join(WAVDIR, '{}_user.wav'.format(file_name))
            wav_out_dir = os.path.join(DATAROOT, 'wav', file_name)
            os.makedirs(wav_out_dir, exist_ok=True)
            
            name = file_name+'_{:03}.wav'.format(i+1)
            wav_out_path = os.path.join(wav_out_dir, name)
            
            save_turn_wav(wavpath, wav_out_path, wav_start, wav_end)