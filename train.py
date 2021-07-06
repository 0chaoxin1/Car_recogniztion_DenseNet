"""
Usage:

New training using default settings:
    python train.py "~/.torch/datasets/flowers" --gpu

New training using custom settings:
    python train.py "~/.torch/datasets/flowers" --gpu --save_dir=models --arch=vgg13 --learning_rate=0.001 --hidden_units 512 256 --epochs 10 

Resume training:
    python train.py "~/.torch/datasets/flowers" --gpu --ckp_file "models\ckp_Flowers_vgg13_512_256_0.0001_20_best.pth" --epoch 10
"""

import argparse
import os
import json
import time
import argparse
import numpy as np
# import matplotlib.pyplot as plt

from collections import OrderedDict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Local imports
from ImageClassifier import ImageClassifier

def parse_input():
    parser = argparse.ArgumentParser(
        description='Train an image classifier and saves the model'
    )
    
    parser.add_argument('--data_dir',
                        help='Path to dataset to train')

    parser.add_argument('--ckp_file', action='store',
                        dest='ckp_file',
                        help='Path to checkpoint to continue training. This ignores arch and hidden_units')

    parser.add_argument('--arch', action='store',
                        dest='arch',
                        help='Architecture to use')

    parser.add_argument('--hidden_units', action='store',
                        dest='hidden_units', nargs='+', type=int,
                        help='Number of hidden units')

    parser.add_argument('--learning_rate', action='store',
                        dest='learning_rate', type=float,
                        help='Learning rate for training')

    parser.add_argument('--epochs', action='store',
                        dest='epochs', type=int,
                        help='Number of epochs for training')

    parser.add_argument('--print_every', action='store',
                        dest='print_every', type=int,
                        help='Print every x steps in the training')

    parser.add_argument('--save_dir', action='store',
                        dest='save_dir',
                        help='Directory to save model checkpoints')
    # 设置是否使用 gpu 使用：default=True 不使用：default=False
    # parser.add_argument('--gpu', action='store_true',
    #                     dest='gpu', default=True,
    #                     help='Train using CUDA:0')

    results = parser.parse_args()
    return results

if __name__ == "__main__":
    print("Hola!")
    # Get cmd args
    args = parse_input()
    
    # Instanciate Image Classifier Class
    ic = ImageClassifier()
    # ic = densenet161()

    # Request GPU if available
    # ic.use_gpu(args.gpu)

    # Load Dataset
    if not ic.load_data(args.data_dir):
        exit()
      
    if args.ckp_file is None:
        # Create Model
        print('#' * 30)
        print(ic.create_model(args.arch, args.hidden_units))

        # Create Optimizer
        # print('#' * 30)
        # print('Optimizer:\n', ic.create_optimizer(args.learning_rate))
        print('#' * 30)
    else:
        # Load checkpoint to resume training
        # 加载检查点恢复训练
        if not ic.load_checkpoint(args.ckp_file):
            exit()

    # Train and save network
    ic.train(args.epochs, args.save_dir, args.print_every)