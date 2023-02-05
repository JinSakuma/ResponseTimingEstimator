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
from espnet2.bin.asr_parallel_transducer_inference import Speech2Text
from src.datasets.dataset_asr_inference import get_dataloader, get_dataset
from src.utils.utils import load_config

PARALLEL=True

speech2text = Speech2Text(
    asr_base_path="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr6/asr1",
    asr_train_config="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr6/asr1/exp/asr_train_asr_cbs_transducer_848_finetune_raw_jp_char_sp/config.yaml",
    asr_model_file="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr6/asr1/exp/asr_train_asr_cbs_transducer_848_finetune_raw_jp_char_sp/valid.loss_transducer.ave_10best.pth",
    token_type=None,
    bpemodel=None,
    beam_size=5,
    beam_search_config={"search_type": "maes"},
    lm_weight=0.0,
    nbest=1,
    #device = "cuda:0", # "cpu",
    device = "cpu",
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
    sim_chunk_length = 2048 # 640
    
#     if len(speech) <= speech2text._raw_ctx*4:
#         add = speech2text._raw_ctx*4 - len(speech) + 1
#         pad = np.zeros(speech2text._raw_ctx*3+add)
#     else:
#         pad = np.zeros(speech2text._raw_ctx*3)
        
    if len(speech) <= speech2text._raw_ctx*4:
        add = speech2text._raw_ctx*4 - len(speech) + 1
        pad = np.zeros(speech2text._raw_ctx*1+add)
    else:
        pad = np.zeros(speech2text._raw_ctx*1)
        
    speech = np.concatenate([speech, pad])
    speech2text.reset_inference_cache()
    if sim_chunk_length > 0:
        for i in range(len(speech)//sim_chunk_length):
            hyps = speech2text.streaming_decode(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=False)
            if not PARALLEL:
                results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps))
            elif hyps[2] is None:               
                results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[1]))
            else:
                results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[2]))
                #results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[0]))
                        
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                output_list.append(text)
            else:
                output_list.append("")

#         hyps = speech2text.streaming_decode(speech[(i+1)*sim_chunk_length:len(speech)], is_final=True)
#         if hyps[2] is None:
#             results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[1]))
#         else:
#             results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[2]))
#             #results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[0]))
#     else:
#         hyps = speech2text(speech)
#         if hyps[2] is None:
#             results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[1]))
#         else:
#             results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[2]))
            #results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[0]))
            
    #nbests = [text for text, token, token_int, hyp in results]
#     if results is not None and len(results) > 0:
#         nbests = [text for text, token, token_int, hyp in results]
#         text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
#         output_list.append(text)
#     else:
#         output_list.append("")
        
    return output_list
    

def run(args):
    config = load_config(args.config)
    seed_everything(config.seed)
    #if args.gpuid >= 0:
    config.gpu_device = args.gpuid
        
    if config.gpu_device>=0:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    #train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'valid')
    test_dataset = get_dataset(config, 'test')
    
#     val_dataset = get_dataset(config, 'valid', ['M1'])
#     test_dataset = get_dataset(config, 'test', ['M1'])
    #test_dataset = get_dataset(config, 'test')
    
    #train_loader = get_dataloader(train_dataset, config, 'train')
    val_loader = get_dataloader(val_dataset, config, 'valid')
    test_loader = get_dataloader(test_dataset, config, 'test')
    
    #loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    loader_dict = {'val': val_loader, 'test': test_loader}
    #loader_dict = {'test': test_loader}
    
    #for split in ['train']:
    for split in ['val', 'test']:
    #for split in ['test']:
        for j, batch in enumerate(tqdm(loader_dict[split])):            
#             wavs = batch[0]#.to(device)
#             wav_lengths = batch[1] #.to(self.device)
#             wav_paths = batch[2] #.to(self.device)
#             batch_size = int(len(wavs))    
            chs = batch[0]           
            wavs = batch[7]#.to(device)
            wav_lengths = batch[8] #.to(self.device)
            wav_paths = batch[9] #.to(self.device)
            batch_size = int(len(chs))


            for i in range(batch_size):
#                 name_part = wav_paths[i].split('/')[-1].split('_')
#                 dir_name = '_'.join(name_part[:-1])
#                 file_name = '_'.join(name_part).replace('.wav', '')
                name_part = wav_paths[i].split('/')[-1].split('_')[:-1]
                dir_name = '_'.join(name_part[:-1])
                file_name = '_'.join(name_part)
                
#                 os.makedirs('/mnt/aoni04/jsakuma/data/ATR-Trek/texts/text_cbs-t_non_parallel/{}'.format(dir_name), exist_ok=True)
#                 df_path = '/mnt/aoni04/jsakuma/data/ATR-Trek/texts/text_cbs-t_non_parallel/{}/{}.csv'.format(dir_name, file_name)

                os.makedirs('/mnt/aoni04/jsakuma/data/ATR-Trek/texts/text_cbs-t/{}'.format(dir_name), exist_ok=True)
                df_path = '/mnt/aoni04/jsakuma/data/ATR-Trek/texts/text_cbs-t/{}/{}.csv'.format(dir_name, file_name)
                
                if os.path.isfile(df_path):
                    continue 
                                 
                wav = wavs[i][:wav_lengths[i]].detach().cpu().numpy()
                #wav = wavs[i].detach().cpu().numpy()
                
#                 L=input_lengths[i]
#                 if vad[i][:L].numpy().sum()>0:
#                     idx = np.where(vad[i][:L].numpy()==1)[0][-1]
#                     N = len(vad[i][:L].numpy())-idx
#                 else:
#                     idx = len(vad[i][:L].numpy())
#                     N = 0
                
#                 wav = wav[:(idx+1)*800]
                text = get_recognize_frame(wav)                                   
#                 tmp = [text[-1]]*N
#                 text += tmp
                
                df = pd.DataFrame({'asr_recog': text})
                df.to_csv(df_path, encoding='utf-8', index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    run(args)