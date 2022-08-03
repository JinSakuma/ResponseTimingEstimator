import os
import json
import random
import torch
import argparse
import numpy as np
from dotmap import DotMap

from src.datasets.dataset_timing_char3 import get_dataloader, get_dataset
#from src.datasets.dataset5_ import get_dataloader, get_dataset
from src.utils.utils import load_config
from src.utils.trainer_turn_shift import trainer
from src.models.gmf.model2 import GMFModel
#from src.models.gmf.model2_early import GMFModel
#from src.models.gmf.baseline import BaselineModel


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def run(args):
    config = load_config(args.config)
    seed_everything(config.seed)

    if args.gpuid >= 0:
        config.gpu_device = args.gpuid
        
    if config.cuda:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    train_dataset = get_dataset(config, 'train', ['M1'])
    val_dataset = get_dataset(config, 'valid', ['M1'])
    #train_dataset = get_dataset(config, 'train', ['M1'], task='turn-shift')
    #val_dataset = get_dataset(config, 'valid', ['M1'], task='turn-shift')
    # test_dataset = get_dataset(config, 'test')
    
    train_loader = get_dataloader(train_dataset, config, 'train')
    val_loader = get_dataloader(val_dataset, config, 'valid')
    # test_loader = get_loader(test_dataset, config, 'test')
    
    loader_dict = {'train': train_loader, 'val': val_loader} #, 'test': test_loader}
    
    del train_dataset
    del val_dataset
    
    model = GMFModel(config, device)
    #model = BaselineModel(config, device)
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
    
