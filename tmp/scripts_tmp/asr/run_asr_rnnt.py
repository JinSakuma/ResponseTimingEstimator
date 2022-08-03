import os
import json
import torch
import argparse
import numpy as np
from dotmap import DotMap

from src.datasets.dataset_asr_kana import get_dataloader, get_dataset
from src.utils.utils import load_config
from src.utils.trainer_asr import trainer
from src.models.asr.transducer.model import RNNTModel
from distutils.util import strtobool


def run(args):
    config = load_config(args.config)
    if args.gpuid >= 0:
        config.gpu_device = args.gpuid
        
    if config.cuda:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'valid')
    #train_dataset = get_dataset(config, 'train', ['M1'])
    #val_dataset = get_dataset(config, 'valid', ['M1'])
    # test_dataset = get_dataset(config, 'test')
    
    train_loader = get_dataloader(train_dataset, config, 'train')
    val_loader = get_dataloader(val_dataset, config, 'valid')
    # test_loader = get_loader(test_dataset, config, 'test')
    
    loader_dict = {'train': train_loader, 'val': val_loader} #, 'test': test_loader}
    
    del train_dataset
    del val_dataset
    
    model = RNNTModel(config, args, device)
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
    
    parser.add_argument(
        "--report-cer",
        default=False,
        action="store_true",
        help="Compute CER on development set",
    )
    parser.add_argument(
        "--report-wer",
        default=True,
        action="store_true",
        help="Compute WER on development set",
    )
    parser.add_argument(
        "--transducer-weight",
        default=1.0,
        type=float,
        help="Weight of main Transducer loss.",
    )
    parser.add_argument(
        "--use-ctc-loss",
        type=strtobool,
        nargs="?",
        default=True,
        help="Whether to compute auxiliary CTC loss.",
    )
    parser.add_argument(
        "--ctc-loss-weight",
        default=0.5,
        type=float,
        help="Weight of auxiliary CTC loss.",
    )
    parser.add_argument(
        "--use-lm-loss",
        type=strtobool,
        nargs="?",
        default=True,
        help="Whether to compute auxiliary LM loss (label smoothing).",
    )
    parser.add_argument(
        "--lm-loss-weight",
        default=0.5,
        type=float,
        help="Weight of auxiliary LM loss.",
    )

    parser.add_argument("--sym-space", default="[PAD]", type=str, help="Space symbol")
    parser.add_argument("--sym-blank", default="<blank>", type=str, help="Blank symbol")
    args = parser.parse_args()
    
    run(args)   


