import os
import json
import random
import torch
import argparse
import numpy as np
from dotmap import DotMap

from src.datasets.dataset_lm import get_dataloader, get_dataset
from src.utils.utils import load_config
from src.utils.trainer_lm import trainer
from src.models.lm.model import LSTMLM


SPEAKERS = ['M1']
FILES = ["M1_all"]

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def cross_validation(config, device, cv_id, out_name):
        
    train_dataset = get_dataset(config, cv_id, split='train', subsets=FILES, speaker_list=SPEAKERS)
    val_dataset = get_dataset(config, cv_id, split='test', subsets=FILES, speaker_list=SPEAKERS)
    
    train_loader = get_dataloader(train_dataset, config, shuffle=True)
    val_loader = get_dataloader(val_dataset, config, shuffle=False)
    
    loader_dict = {'train': train_loader, 'val': val_loader}
    
    del train_dataset
    del val_dataset
    
    model = LSTMLM(config, device)
    model.to(device)
    
    parameters = model.configure_optimizer_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.optim_params.learning_rate,
        weight_decay=config.optim_params.weight_decay,
    )
    
    os.makedirs(os.path.join(config.exp_dir, out_name), exist_ok=True)
    
    trainer(
        num_epochs=config.num_epochs,
        model=model,
        loader_dict=loader_dict,
        optimizer=optimizer,
        device=device,
        outdir=os.path.join(config.exp_dir, out_name),
        phasename=out_name
    )        

def run(args):
    config = load_config(args.config)
    seed_everything(config.seed)
    if args.gpuid >= 0:
        config.gpu_device = args.gpuid
        
    if config.cuda:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    cross_validation(config, device, 1, "cv1")
    cross_validation(config, device, 2, "cv2")
    cross_validation(config, device, 3, "cv3")
    cross_validation(config, device, 4, "cv4")
    cross_validation(config, device, 5, "cv5")    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    run(args)
    
