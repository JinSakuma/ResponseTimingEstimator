import os
import json
import torch
import argparse
import numpy as np
from dotmap import DotMap

from dataset import get_dataloader, get_dataset
from utils import load_config
from model import System
from trainer import trainer


def run(args):
    config = load_config(args.config)
    if args.gpuid >= 0:
        config.gpu_device = args.gpuid
        
    if config.cuda:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'val')
    # test_dataset = get_dataset(config, 'test')
    
    train_loader = get_dataloader(train_dataset, config, 'train')
    val_loader = get_dataloader(val_dataset, config, 'val')
    # test_loader = get_loader(test_dataset, config, 'test')
    
    loader_dict = {'train': train_loader, 'val': val_loader} #, 'test': test_loader}
    
    del train_dataset
    del val_dataset
    
    model = System(config, device)
    model.to(device)
    
    parameters = model.configure_optimizer_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.optim_params.learning_rate,
        weight_decay=config.optim_params.weight_decay,
    )
    
    trainer(
        num_epochs=config.num_epochs,
        model=model,
        loader_dict=loader_dict,
        optimizer=optimizer,
        device=device,
        outdir=config.exp_dir,
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    run(args)
    