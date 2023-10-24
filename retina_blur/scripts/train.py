import os
import csv
import cv2
import time
import glob
import random
import shutil
import numpy as np
import pandas as pd
from skimage import io
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import torch
import torchvision
import torchmetrics
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import *
from config import *
from test import run_test

# Function to train model for given epochs, lr, and task (diabetic retinopathy or good vs bad)

def run_train(net, trainloader, valloader, criterion, optimizer,
              scheduler, num_epochs, decay_epochs, init_lr, task, print_freq):

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        # Running metrics, reset once printed after every few mini batches (defined below)
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        
        # Total metrics
        loss_final = 0.0
        avg_loss_final = 0.0
        total_final = 0.0
        correct_final = 0.0
        
        start_time = time.time()
        
        net.train()

        for i, data in enumerate(trainloader, 0):
        
            # adjust_learning_rate(optimizer, epoch, init_lr, decay_epochs)
            images, labels = data['image'].to(device), data['label'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Getting predicted results
            _, predicted = torch.max(outputs.data, 1)
            
            # Print statistics
            # print_freq = 100
            running_loss += loss.item()
            # loss_final += loss.item()
            avg_loss_final += loss.item()  / len(trainloader)

            # Calculate acc
            running_total += labels.size(0)  
            total_final += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            correct_final += (predicted == labels).sum().item()
            
            if i % print_freq == (print_freq - 1):    # print every 100 mini-batches
                print(f'[epoch {epoch + 1}, batch {i + 1:5d}] loss: {running_loss / print_freq:.3f} acc: {100*running_correct / running_total:.2f} time: {time.time() - start_time:.2f}')
                running_loss, running_correct, running_total = 0.0, 0.0, 0.0
                start_time = time.time()

        # Append train loss and accuracy to list
        # train_loss.append(loss_final/total_final)
        train_loss.append(avg_loss_final)
        train_acc.append(np.round((100*correct_final)/total_final,2))
        
        # Running the run_test() function after each epoch; Setting the model to the evaluation mode.
        net.eval()
        _, _, val_loss_epoch, val_acc_epoch = run_test(net, valloader, criterion, task)

        # Append val loss and accuracy to list        
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)
        
    print('Finished Training')
    
    return train_loss, train_acc, val_loss, val_acc