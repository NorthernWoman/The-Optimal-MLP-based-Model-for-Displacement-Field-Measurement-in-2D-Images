import torch
import torch.nn as nn
from .common import *
def conv_layers(num_input_channels=2, num_output_channels=1, num_hidden=200,need_sigmoid = True, need_tanh = False):
    model = nn.Sequential(
        nn.Conv2d(
            num_input_channels,
            num_hidden,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(num_hidden),

        nn.Conv2d(
            num_hidden,
            num_hidden,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        
        nn.Conv2d(
            num_hidden,
            num_hidden,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        
        nn.Conv2d(
            num_hidden,
            num_hidden,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        
        nn.Conv2d(
            num_hidden,
            num_hidden,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        nn.Conv2d(
            num_hidden,
            num_output_channels,
            kernel_size=1,
            padding=0),
    )
    if need_sigmoid:
        model.add(nn.Sigmoid())
    elif need_tanh:
        model.add(nn.Tanh())
    return model
