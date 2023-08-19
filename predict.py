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
parser.add_argument('--save_dir', action='store', default='checkpoint/')
parser.add_argument('--top_k', action='store',type=int, default=5)
parser.add_argument('--label_map', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default=False)
args = parser.parse_args()
print("Data file path: ", args.data)


img_dir = args.data
checkpoint_path = args.save_dir
top_k = args.top_k
label_map = args.label_map
gpu = args.gpu

device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

#Loding the checkpoint
model = Classifier.load_checkpoint(checkpoint_path)

#Process the image
processed_data = Classifier.process_image(img_dir)

#Predict the image
probs, classes = Classifier.predict(processed_data,model,top_k, device)

#Label map
cat_to_name = Classifier.label_mapping(label_map)

labels = []

for class_index in classes:
    labels.append(cat_to_name[str(class_index)])
    
print("Flower Name: ",labels[0])
print("The Probability: ",probs)



