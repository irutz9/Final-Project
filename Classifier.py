#!/usr/bin/env python
# coding: utf-8


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.models as models
import json




def load_data(data_dir):
#     data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # TODO: Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }
    
    return dataloaders, image_datasets


def label_mapping(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name
    


def model_setup(arch,hidden_layers, lr):
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        print("we use the default arch which is vgg13")
        model = models.vgg13(pretrained=True)

        

        # Freeze the pre-trained network parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained feed-forward network as a classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),
        nn.LogSoftmax(dim=1)
    )

    # Replace the pre-trained model's classifier with the newly defined classifier
    model.classifier = classifier

    return model



# Train the classifier layers using backpropagation
def train_model(model, dataloaders, device, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate validation loss and accuracy after each epoch
        validation_loss = 0.0
        accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                validation_loss += val_loss.item()

                probabilities = torch.exp(outputs)
                equality = (labels.data == probabilities.max(dim=1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_val_loss = validation_loss / len(dataloaders['valid'])
        epoch_acc = accuracy / len(dataloaders['valid'])

        print(f"Epoch: {epoch + 1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Validation Loss: {epoch_val_loss:.4f} | "
              f"Validation Accuracy: {epoch_acc:.4f}")






def test_model(model,dataloaders,device):
    # Set the model to evaluation mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloaders['test']:
            # Move images and labels to the available device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate the accuracy
        accuracy = 100 * correct / total

        # Print the accuracy
        print(f"Test accuracy: {accuracy:.2f}%")
        


def save_check(model, arch, lr, epochs, hidden_units, class_to_idx, checkpoint_path):
    model.class_to_idx = class_to_idx
#     # Define the checkpoint file path
#     checkpoint_path = checkpoint_path

    # Save the checkpoint
    checkpoint = {
        'arch':arch,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'learning_rate': lr,
        'epochs': epochs,
        'hidden_layers':hidden_units,
    }
    torch.save(checkpoint, checkpoint_path)
    print("Checkpoint saved successfully.")



def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    hidden_layers = checkpoint['hidden_layers']
    lr = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    
    model = model_setup(arch,hidden_layers,lr)
    # Load the saved class-to-index mapping
    model.class_to_idx = checkpoint['class_to_idx']
    # Load the saved model state dictionary
    model.load_state_dict(checkpoint['state_dict'])
    
    return model



def process_image(image):
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(Image.open(image))
    return img


def predict(image_path, model, topk, device):
    
    model.eval()
    model.to(device)
    image = image_path.unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        top_probabilities, top_indices = torch.topk(probabilities, k=topk)
    
    top_probabilities = top_probabilities.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_probabilities, top_classes


