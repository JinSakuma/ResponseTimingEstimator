import os
import json
import torch
import random
import argparse
import numpy as np
from dotmap import DotMap

from src.utils.utils import load_config
from src.datasets.dataset_timing import get_dataloader, get_dataset
from src.utils.trainer_timing import trainer
from src.utils.tester_timing import tester
from src.models.timing.model_proposed import System
from src.models.timing.model_baseline import BaselineSystem


SPEAKERS = ['M1']
FILES = ['all']

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def cross_validation(config, model, device, cv_id, out_name):        
    train_dataset = get_dataset(config, cv_id, split='train', subsets=FILES, speaker_list=SPEAKERS)
    val_dataset = get_dataset(config, cv_id, split='test', subsets=FILES, speaker_list=SPEAKERS)
    
    train_loader = get_dataloader(train_dataset, config, shuffle=True)
    val_loader = get_dataloader(val_dataset, config, shuffle=False)
    
    loader_dict = {'train': train_loader, 'val': val_loader}
    #loader_dict = {'val': val_loader}
    
    del train_dataset
    del val_dataset
    
    if model == 'baseline':
        model = BaselineSystem(config, device)
    else:
        model = System(config, device)
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
    
    tester(
        config,
        device,
        loader_dict['val'],
        model,
        model_dir=os.path.join(config.exp_dir, out_name),
        out_dir=config.exp_dir,
        resume_name=out_name,
        resume=True,
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
        
    config.model_params.lm_model_path = os.path.join(config.model_params.lm_model_path, "cv{}".format(str(args.cv_id)), "best_val_bacc_model.pth")
        
    if args.cv_id == 1:
        trainset = ["M1_sub1", "M1_sub2", "M1_sub3", "M1_sub4"]
        testset = ["M1_sub5"]
    elif args.cv_id == 2:
        trainset = ["M1_sub5", "M1_sub1", "M1_sub2", "M1_sub3"]
        testset = ["M1_sub4"]
    elif args.cv_id == 3:
        trainset = ["M1_sub4", "M1_sub5", "M1_sub1", "M1_sub2"]
        testset = ["M1_sub3"]
    elif args.cv_id == 4:
        trainset = ["M1_sub3", "M1_sub4", "M1_sub5", "M1_sub1"]
        testset = ["M1_sub2"]
    elif args.cv_id == 5:
        trainset = ["M1_sub2", "M1_sub3", "M1_sub4", "M1_sub5"]
        testset = ["M1_sub1"]
    else:
        NotImplemented
                
    cross_validation(config, args.model, device, int(args.cv_id), "cv{}".format(int(args.cv_id)))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/config.path', help='path to config file')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'proposed'], help='model type')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    parser.add_argument('--cv_id', type=int, default=0, help='id for the cross validation settings')
    args = parser.parse_args()
    run(args)
    
