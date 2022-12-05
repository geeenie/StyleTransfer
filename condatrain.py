# 0. Package Installation
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import time
from utils.imageloader import load_images
from utils.dataloader import load_data
from utils.normalize import batch_normalize
from utils.gram_matrix import gram_matrix
from model.VGG16 import VGG16
from model.TransformerNet import TransformerNet

def train():
    print("did it")

train()