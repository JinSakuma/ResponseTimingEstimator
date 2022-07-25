import os
import json
import torch
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotmap import DotMap
from tqdm import tqdm

from src.datasets.dataset_char_recog import get_dataloader, get_dataset
from src.utils.utils import load_config
from src.models.lm.model import TransformerLM
from src.models.lm.model import LSTMLM


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def run(args):
    config_path = args.config  # '../../configs/lm/kana_lstm_l1.json'
    config = load_config(config_path)
    seed_everything(config.seed)
    # if args.gpuid >= 0:
    config.gpu_device = 0

    if config.cuda:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')

    # train_dataset = get_dataset(config, 'train')
    # val_dataset = get_dataset(config, 'valid')

    #train_dataset = get_dataset(config, 'train', ['M1'])
    #val_dataset = get_dataset(config, 'valid', ['M1'])
    #test_dataset = get_dataset(config, 'test', ['M1'])
    
    val_dataset = get_dataset(config, 'valid', ['F3', 'F4', 'M1', 'M2', 'M3', 'M4'])
    test_dataset = get_dataset(config, 'test', ['F3', 'F4', 'M1', 'M2', 'M3', 'M4'])

    #train_loader = get_dataloader(train_dataset, config, 'train')
    val_loader = get_dataloader(val_dataset, config, 'valid')
    test_loader = get_dataloader(test_dataset, config, 'test')
    
    if 'transformer' in config_path:
        self = TransformerLM(config, device)
    else:
        self = LSTMLM(config, device)
    #path = "../../exp/lm/kana/M1/lstm_l1_lr5/best_val_loss_model.pth"
    path = args.model
    self.load_state_dict(torch.load(path), strict=False)
    self.to(device)
    
    EOU=1314
    
    tpr, tnr, bAcc = 0, 0, 0
    total = 0
    for batch in tqdm(val_loader):
        texts = batch[0]
        phons = batch[1]
        idxs = batch[2]
        input_lengths = batch[3]
        indices = batch[4]
        batch_size = len(indices)

        inputs = idxs[:, :-1].to(self.device)
        targets = idxs[:, 1:].to(self.device)

        with torch.no_grad():
            #outputs = self.lm(inputs)    
            outputs = self.lm(inputs, input_lengths)    
            _, preds = torch.max(outputs.data, -1)

        for i in range(batch_size):
            prd = preds[i][:input_lengths[i]].detach().cpu()
            trt = targets[i][:input_lengths[i]].detach().cpu()

            TP, FP, FN, TN = 0, 0, 0, 0
            for i in range(len(prd)):
                if trt[i]==EOU and prd[i]==EOU:
                    TP+=1
                elif trt[i]==EOU and prd[i]!=EOU:
                    FN+=1
                elif trt[i]!=EOU and prd[i]==EOU:
                    FP+=1
                else:
                    TN+=1

            tpr += TP / (TP+FN)
            tnr += TN / (TN+FP)
            bAcc += (tpr+tnr)/2

        total+=batch_size

    tpr = tpr / total
    tnr = tnr / total
    bAcc = (tpr+tnr)/2
    print('TPR:{:.4f}'.format(tpr))
    print('TNR:{:.4f}'.format(tnr))
    print('bACC:{:.4f}'.format(bAcc)) 
    
    tpr, tnr, bAcc = 0, 0, 0
    total = 0
    for batch in tqdm(test_loader):
        texts = batch[0]
        phons = batch[1]
        idxs = batch[2]
        input_lengths = batch[3]
        indices = batch[4]
        batch_size = len(indices)

        inputs = idxs[:, :-1].to(self.device)
        targets = idxs[:, 1:].to(self.device)

        with torch.no_grad():
            outputs = self.lm(inputs, input_lengths)    
            _, preds = torch.max(outputs.data, -1)

        for i in range(batch_size):
            prd = preds[i][:input_lengths[i]].detach().cpu()
            trt = targets[i][:input_lengths[i]].detach().cpu()

            TP, FP, FN, TN = 0, 0, 0, 0
            for i in range(len(prd)):
                if trt[i]==EOU and prd[i]==EOU:
                    TP+=1
                elif trt[i]==EOU and prd[i]!=EOU:
                    FN+=1
                elif trt[i]!=EOU and prd[i]==EOU:
                    FP+=1
                else:
                    TN+=1

            if TP+FN>0:
                tpr += TP / (TP+FN)
            if TN+FP>0:
                tnr += TN / (TN+FP)
            bAcc += (tpr+tnr)/2

        total+=batch_size

    tpr = tpr / total
    tnr = tnr / total
    bAcc = (tpr+tnr)/2
    print('TPR:{:.4f}'.format(tpr))
    print('TNR:{:.4f}'.format(tnr))
    print('bACC:{:.4f}'.format(bAcc)) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--model', type=str, help='path to model weight')
    args = parser.parse_args()
    run(args)

