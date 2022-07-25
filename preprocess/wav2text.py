import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import wave
import sys
from dotmap import DotMap
from tqdm import tqdm

sys.path.append('../')
import espnet
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
# from src.datasets.dataset_asr_inference_reverse import get_dataloader, get_dataset
from src.datasets.dataset_asr_inference import get_dataloader, get_dataset
from src.utils.utils import load_config


speech2text = Speech2TextStreaming(
    asr_base_path="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr/asr1",
    asr_train_config="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr/asr1/exp/asr_train_asr_streaming_conformer_blk10_hop4_la2_raw_jp_char_sp/config.yaml",
    asr_model_file="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr/asr1/exp/asr_train_asr_streaming_conformer_blk10_hop4_la2_raw_jp_char_sp/valid.acc.ave_10best.pth",
    lm_train_config="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr/asr1/exp/lm_train_lm2_jp_char/config.yaml",
    lm_file="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr/asr1/exp/lm_train_lm2_jp_char/valid.loss.best.pth",
    token_type=None,
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=2,
    ctc_weight=0.5,
    lm_weight=0.0,
    penalty=0.0,
    nbest=1,
    device = "cuda:1", # "cpu",
    disable_repetition_detection=True,
    decoder_text_length_limit=0,
    encoded_feat_length_limit=0
)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_recognize_frame(data):
    output_list = []
    speech = data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    sim_chunk_length = 800 # 640

    if sim_chunk_length > 0:
        for i in range(len(speech)//sim_chunk_length):
            results = speech2text(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=False)
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                output_list.append(text)
            else:
                output_list.append("")

        results = speech2text(speech[(i+1)*sim_chunk_length:len(speech)], is_final=True)
    else:
        results = speech2text(speech, is_final=True)
    
    nbests = [text for text, token, token_int, hyp in results]
    if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                output_list.append(text)
    else:
        output_list.append("")
        
    return output_list
    

def run(args):
    config = load_config(args.config)
    seed_everything(config.seed)
    if args.gpuid >= 0:
        config.gpu_device = args.gpuid
        
    if config.cuda:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    #train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'valid')
    test_dataset = get_dataset(config, 'test')
    
    #train_loader = get_dataloader(train_dataset, config, 'train')
    val_loader = get_dataloader(val_dataset, config, 'valid')
    test_loader = get_dataloader(test_dataset, config, 'test')
    
    #loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    loader_dict = {'val': val_loader, 'test': test_loader}
    
    #for split in ['train']:
    #for split in ['val', 'test']:
    for split in ['test']:
        for batch in tqdm(loader_dict[split]):
            chs = batch[0]
            vad = batch[1]
            turn = batch[2].to(device)
            #last_ipu = batch[3].to(self.device)
            targets = batch[4].to(device)
            #feats = batch[5].to(device)
            input_lengths = batch[6] #.to(self.device)
            wavs = batch[7]#.to(device)
            wav_lengths = batch[8] #.to(self.device)
            wav_paths = batch[9] #.to(self.device)
            batch_size = int(len(chs))

            for i in range(batch_size):
                name_part = wav_paths[i].split('/')[-1].split('_')[:-1]
                dir_name = '_'.join(name_part[:-1])
                file_name = '_'.join(name_part)
                
                os.makedirs('/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/text_10_4_2/{}'.format(dir_name), exist_ok=True)
                df_path = '/mnt/aoni04/jsakuma/data/ATR-Fujie/samples/text_10_4_2/{}/{}.csv'.format(dir_name, file_name)
                
                if os.path.isfile(df_path):
                    continue
                
                wav = wavs[i][:wav_lengths[i]].detach().cpu().numpy()
                
                L=input_lengths[i]
                if vad[i][:L].numpy().sum()>0:
                    idx = np.where(vad[i][:L].numpy()==1)[0][-1]
                    N = len(vad[i][:L].numpy())-idx
                else:
                    idx = len(vad[i][:L].numpy())
                    N = 0
                
                wav = wav[:(idx+1)*800]
                
                text = get_recognize_frame(wav)
                tmp = [text[-1]]*N
                text += tmp
                
                df = pd.DataFrame({'asr_recog': text})
                
                df.to_csv(df_path, encoding='utf-8', index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    run(args)

