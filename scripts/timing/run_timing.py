import os
import json
import torch
import argparse
import numpy as np
from dotmap import DotMap

from src.datasets.dataset3 import get_dataloader, get_dataset
# from src.datasets.dataset_rtnet import get_dataloader, get_dataset
from src.utils.utils import load_config
from src.utils.trainer_timing import trainer
from src.models.timing.model import TimingEstimator


def run(args):
    config = load_config(args.config)
    if args.gpuid >= 0:
        config.gpu_device = args.gpuid
        
    if config.cuda:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    train_dataset = get_dataset(config, 'train', ['M1'])
    val_dataset = get_dataset(config, 'valid', ['M1'])
    #train_dataset = get_dataset(config, 'train')
    #val_dataset = get_dataset(config, 'valid')
    # test_dataset = get_dataset(config, 'test')
    
    train_loader = get_dataloader(train_dataset, config, 'train')
    val_loader = get_dataloader(val_dataset, config, 'valid')
    # test_loader = get_loader(test_dataset, config, 'test')
    
    loader_dict = {'train': train_loader, 'val': val_loader} #, 'test': test_loader}
    
    del train_dataset
    del val_dataset
    
    model = TimingEstimator(config, device)
    model.to(device)

    # model.vad.load_state_dict(torch.load(config.vad_continue_from_checkpoint), strict=False)
    
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
    
