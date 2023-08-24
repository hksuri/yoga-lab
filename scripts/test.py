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

from config import *
from utils import *

# Function to evaluate updated model after each epoch
def run_test(net, testloader, criterion):
    
    correct = 0
    loss_final = 0
    total = 0
    avg_test_loss = 0.0
    labels = torch.empty(0).cuda()
    predicted = torch.empty(0).cuda()
    

    with torch.no_grad():
        
        for data in testloader:
            images, labels_batch = data['image'].to(device), data['label'].to(device)
            
            outputs = net(images)
            _, predicted_batch = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted_batch == labels_batch).sum().item()

            # loss
            avg_test_loss += criterion(outputs, labels_batch)  / len(testloader)

            # concatenate preds and labels
            labels = torch.cat((labels,labels_batch))
            predicted = torch.cat((predicted,predicted_batch))
            
    print('TESTING:')
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f} %')
    print(f'Average loss on the test images: {avg_test_loss:.3f}\n')
    
    test_acc = np.round((100*correct)/total,2)
    
    return labels, predicted, avg_test_loss.item(), test_acc