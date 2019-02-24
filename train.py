import numpy as np 
import torch
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn 
from torch import optim
from PIL import Image
import pandas as pd
import argparse
import time
import copy


# Defining Arguement Parser
arg_parser = argparse.ArgumentParser()

# Adding Arguements to Parser
arg_parser.add_argument('--data_dir', type=str, help='Dataset Filepath')
arg_parser.add_argument('--gpu', action='store_true', help='Use GPU')
arg_parser.add_argument('--epochs', type=int, help='Number of epochs')
arg_parser.add_argument('--arch', type=str, help='Network Architecture')
arg_parser.add_argument('--learning_rate', type=float, help='Learning Rate')
arg_parser.add_argument('--hidden_units', type=int, help='Number of Hidden Units')
arg_parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')

# Parsing Arguements
args,_ = arg_parser.parse_known_args()

# Function to Load Model
def load_model(arch='densenet161', output_size = 102, hidden_units = 500):
    
    # Loading a Pretrained Model
    if arch == 'densenet161':
        model = models.densenet161(pretrained = True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'alexnet': 
        model = models.alexnet(pretrained=True)
    else: 
        raise ValueError('Unknown network architecture:  ', arch)
        
    input_size = model.classifier.in_features
    
    for param in model.parameters(): 
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(input_size, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(p=0.5)),
                                        ('batch_normalization',nn.BatchNorm1d(hidden_units)),
                                        ('fc2', nn.Linear(hidden_units, output_size)),
                                        ('output', nn.LogSoftmax(dim = 1))]))
    model.classifier = classifier
    
    return model

# Function to Validate Model 
def validation(model, validloader, criterion): 
    total = 0
    correct = 0
    validation_loss = 0
    for inputs, labels in validloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs).to("cuda")
        _, predicted = torch.max(outputs.data,1)
        total += float(labels.size(0))
        correct += float((predicted == labels).sum().item())
        validation_loss += float(criterion(outputs, labels).item())
    return validation_loss, correct, total

# Function to Train Model 
def train_model(image_datasets, arch='densenet161', hidden_units = 500, epochs = 15, learning_rate = 0.0001, gpu = True, checkpoint = ' '): 
    
    # Use command line arguements if available
    # Architecture
    if args.arch:
        arch = args.arch     
    # Hidden Layer Size 
    if args.hidden_units:
        hidden_units = args.hidden_units
    # Learning Rate 
    if args.learning_rate:
        learning_rate = args.learning_rate
    # Use GPU
    if args.gpu:
        gpu = args.gpu
    # Checkpoint Availible
    if args.checkpoint:
        checkpoint = args.checkpoint
        
        
    # Defining the DataLoaders Using Image Datasets
    dataloaders = {
        i: data.DataLoader(image_datasets[i], batch_size=32, shuffle = True)
        for i in list(image_datasets.keys())
    }
    
    # Dataset Sizes
    dataset_sizes = {
        i: len(dataloaders[i].dataset) 
        for i in list(image_datasets.keys())
    }    
    
    # Loading the model
    output_size = len(image_datasets['train'].classes)
    model = load_model(arch, output_size, hidden_units)
    
    # Use GPU 
    model.to('cuda')
    
    # Defining Criterion and optimizer
    crtierion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    steps = 0
    print_every = 40
    
    for e in range(epochs): 
        running_loss = 0

        for inputs, labels in dataloaders['train']: 
            steps += 1 
            
            # Moving inputs, labels to GPU 
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            # Forward Pass 
            outputs = model.forward(inputs)
            # Calculating loss after Forward pass
            loss = criterion(outputs, labels)
            # Backpropagating 
            loss.backward()
            # Updating weights
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0: 
                # Model in Eval Mode for inference
                model.eval()
            
                # Turning off gradients for validation
                with torch.no_grad():
                    validation_loss, correct, total = validation(model, dataloader['valid'], criterion)
                
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(validation_loss/dataset_sizes['valid']),
                          "Test Accuracy: {:.3f}".format((correct/total)*100))
                
                    running_loss = 0
                
                # Putting Model back in Training mode
                model.train()
                
                # Saving class_to_idx 
                model.class_to_idx = image_datasets['train'].class_to_idx
                
                # Save if Checkpoint Requested
    if checkpoint: 
        print ('Saving checkpoint to:', checkpoint) 
        checkpoint_dict = {
        'arch': arch,
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict(),
        'hidden_units': hidden_units   }
        # Saving model using torch
        torch.save(checkpoint_dict, checkpoint)
     # Returning Model
    return model
    
# Train model if invoked from command line
if args.data_dir:    
    # Default transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomVerticalFlip(p=0.2), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224,224)), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
         ]),
        'valid': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
         ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
         ])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }
        
    train_model(image_datasets) 
                
                
                
                
    
    

    
    
    
        
    
        