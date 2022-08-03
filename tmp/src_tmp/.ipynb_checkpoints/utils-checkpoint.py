import os
import json
import torch
import shutil
import functools
import numpy as np
from dotmap import DotMap
from collections import Counter, OrderedDict


def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)


def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)
        

def load_config(config_path):
    config_json = load_json(config_path)
    config = DotMap(config_json)
    
    exp_base = config.exp_base
    exp_dir = os.path.join(exp_base, config.exp_name)
    config.exp_dir = exp_dir

    os.makedirs(exp_dir, exist_ok=True)
    
    return config