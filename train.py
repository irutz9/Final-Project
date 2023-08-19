#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports here
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import Classifier


# Argparser Arguments
parser = argparse.ArgumentParser()
parser.add_argument('data', action='store')
parser.add_argument('--save_dir', action='store', default='checkpoints/')
parser.add_argument('--arch', action='store', choices=['vgg13','vgg19'], default='vgg13')
parser.add_argument('--learning_rate', action='store', type=float, default=0.01)
parser.add_argument('--epochs', action='store', type=int, default=20)
parser.add_argument('--hidden_units', action='store', type=int, default=512)
parser.add_argument('--gpu', action='store_true', default=False)
args = parser.parse_args()


data_dir = args.data
checkpoint_path = args.save_dir
arch = args.arch
hidden_units = args.hidden_units
epochs = args.epochs
lr = args.learning_rate
gpu = args.gpu

device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    
# Loading Dataset
dataloaders, image_datasets  = Classifier. load_data(data_dir)
class_to_idx = image_datasets['train'].class_to_idx

# Network Setup
model = Classifier.model_setup(arch, hidden_units, lr)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

# Training Model
Classifier.train_model(model, dataloaders, device, criterion, optimizer, epochs)

# Testing Model
Classifier.test_model(model, dataloaders,device)

# Saving Checkpoint
Classifier.save_check(model, arch, lr, epochs, hidden_units, class_to_idx, checkpoint_path)

