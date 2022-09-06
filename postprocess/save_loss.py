import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse




def run(args):
    
    exp_path = args.exp
    
    loss_list = []
    model_paths = os.listdir(exp_path)
    for path in model_paths:
        if not '.pth' in path:
            continue

        loss = float(path.split('_')[-1].replace('loss', '').replace('.pth', ''))
        loss_list.append(loss)

    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(exp_path, 'loss.jpg'))
    
    path_text = os.path.join(exp_path, 'loss.txt')
    for loss in loss_list: 
        with open(path_text, "a") as f:
            f.write(str(loss)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='path to exp dir')
    args = parser.parse_args()
    run(args)